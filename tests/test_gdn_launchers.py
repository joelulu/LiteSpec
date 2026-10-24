"""Execute the real shell chain with fake GPU workers; no hardware/data needed."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('entrypoint', ['train_gdn_ablation.sh', 'run_gdn_20k.sh'])
@pytest.mark.parametrize('fail_arm', ['', 'dlite_gdn'])
def test_four_card_training_success_gates_auto_evaluation(tmp_path, fail_arm, entrypoint):
    python = tmp_path/'python'
    python.write_text('#!'+sys.executable+'\n'+r'''
import json, os
from pathlib import Path
import sys
args=sys.argv[1:]
log=Path(os.environ['CHAIN_LOG'])
def record(value):
    with log.open('a') as f: f.write(json.dumps(value)+'\n')
if args[:1] == ['-c'] or args[:1] == ['tests/check_gdn_reader_cuda.py'] or args[:2] == ['-', '4']:
    sys.exit(0)
if args[:1] == ['tests/prepare_gdn_subset.py']:
    output=Path(args[args.index('--output-dir')+1]); output.mkdir(parents=True)
    count=int(args[args.index('--count')+1])
    record(dict(prepare=count,gpus=os.environ['CUDA_VISIBLE_DEVICES']))
    (output/'train.jsonl').write_text(''.join(json.dumps(dict(id=str(i),question='q',answer='a'))+'\n' for i in range(count)))
    sys.exit(0)
if args[:2] == ['-m','torch.distributed.run']:
    output=Path(args[args.index('--output-dir')+1]); arm=output.name
    record(dict(train=arm,gpus=os.environ['CUDA_VISIBLE_DEVICES'],workers=os.environ['NPROC_PER_NODE'],
        data=args[args.index('--train-data')+1], reuse=args[args.index('--reuse-validation-from')+1] if '--reuse-validation-from' in args else None))
    if arm == os.environ.get('FAIL_ARM'): sys.exit(13)
    import yaml
    cfg=yaml.safe_load(Path(args[args.index('--config')+1]).read_text())
    cfg.update(target_model=os.environ['TARGET_MODEL'],train_data=os.environ['TRAIN_DATA'],data_sha256='same',
        eligible_indices_sha256='same',world_size=4)
    (output/'effective_config.json').write_text(json.dumps(cfg))
    p=output/'last';p.mkdir()
    (p/'config.json').write_text('{}')
    (p/'checkpoint.json').write_text(json.dumps(dict(cursor=dict(global_step=12,optimizer_step=2))))
    sys.exit(0)
if args[:1] == ['tests/run_vlm_matrix.py']:
    record(dict(evaluate=args,gpus=os.environ['CUDA_VISIBLE_DEVICES'],image_cap=os.environ['IMAGE_MAX_TOKENS']))
    sys.exit(0)
os.execv(sys.executable,[sys.executable,*args])
''')
    python.chmod(0o755)
    manifest=tmp_path/'train.jsonl';manifest.write_text('{}\n')
    target=tmp_path/'target';target.mkdir()
    env=dict(os.environ,PYTHON_BIN=str(python),DLITE_WORKSPACE=str(tmp_path),TARGET_MODEL=str(target),
             TRAIN_DATA=str(manifest),RUN_GROUP=str(tmp_path/'runs'),CHAIN_LOG=str(tmp_path/'chain.jsonl'),
             FAIL_ARM=fail_arm,CUDA_VISIBLE_DEVICES='0,1,2,3',NPROC_PER_NODE='8')
    # Inherited debug settings from an unrelated run must not cap this run.
    env.update(MAX_STEPS='7',MAX_SAMPLES='100')
    result=subprocess.run(['bash','tests/'+entrypoint],cwd=ROOT,env=env,text=True,capture_output=True)
    assert (tmp_path/'chain.jsonl').exists(), result.stdout+result.stderr
    rows=[json.loads(line) for line in (tmp_path/'chain.jsonl').read_text().splitlines()]
    trained=[r for r in rows if 'train' in r]
    if entrypoint == 'run_gdn_20k.sh':
        assert rows[0] == dict(prepare=20000,gpus='')
        assert 'valid20k' in trained[0]['data']
        assert trained[0]['reuse'] == str(Path(trained[0]['data']).parent)
        assert all(r['data'] == trained[0]['data'] for r in trained)
    assert [r['train'] for r in trained] == ['dflash','dlite','dflash_gdn','dlite_gdn']
    assert all(r['gpus']=='4,5,6,7' and r['workers']=='4' for r in trained)
    if fail_arm:
        assert result.returncode != 0
        assert not any('evaluate' in r for r in rows)
        assert not (tmp_path/'runs'/fail_arm/'training_complete.json').exists()
    else:
        assert result.returncode == 0,result.stdout+result.stderr
        evaluations=[r for r in rows if 'evaluate' in r]
        assert len(evaluations)==1
        args=evaluations[0]['evaluate']
        assert args[args.index('--gpus')+1]=='4,5,6,7'
        assert args.count('--checkpoint')==4
        assert evaluations[0]['image_cap']=='1024'
        for arm in ('dflash','dlite','dflash_gdn','dlite_gdn'):
            cfg=json.loads((tmp_path/'runs'/arm/'effective_config.json').read_text())
            assert cfg['batch_size']*cfg['accumulation_steps']*cfg['world_size']==64
            assert cfg['num_epochs']==3 and cfg['num_draft_layers']==5 and cfg['num_anchors']==512
            assert cfg['gdn_enabled']==arm.endswith('_gdn')
            assert (tmp_path/'runs'/arm/'training_complete.json').is_file()


def test_explicit_four_checkpoint_matrix_manifest(monkeypatch, tmp_path):
    import run_vlm_matrix as matrix
    target = tmp_path/'target'
    target.mkdir()
    (target/'config.json').write_text('{}')
    data = tmp_path/'gsm.jsonl'
    data.write_text(''.join(json.dumps(dict(turns=[f'question {i}']))+'\n' for i in range(2)))
    argv = ['matrix','--workspace',str(tmp_path),'--output-dir',str(tmp_path/'eval'),
        '--gpus','4,5,6,7','--tests','text-only','--samples','2','--batch-sizes','1',
        '--state-modes','per_token','--dry-run']
    for arm in ('dflash','dlite','dflash_gdn','dlite_gdn'):
        path = tmp_path/arm
        path.mkdir()
        (path/'config.json').write_text(json.dumps(dict(architectures=['DLiteDraftModel' if arm.startswith('dlite') else 'DFlashDraftModel'],
            block_size=8,num_hidden_layers=5,gdn_reader_config=dict(layer_ids=[14,30]) if arm.endswith('_gdn') else None)))
        argv += ['--checkpoint',arm,str(path)]
    monkeypatch.setenv('TARGET_MODEL',str(target))
    monkeypatch.setenv('TEXT_DATA',str(data))
    monkeypatch.setenv('IMAGE_MAX_TOKENS','1024')
    monkeypatch.delenv('IMAGE_MAX_PIXELS',raising=False)
    monkeypatch.delenv('IMAGE_MIN_PIXELS',raising=False)
    monkeypatch.setattr(sys,'argv',argv)
    matrix.main()
    manifest = json.loads((tmp_path/'eval/manifest.json').read_text())
    assert list(manifest['models']) == ['dflash','dlite','dflash_gdn','dlite_gdn']
    assert manifest['gpus'] == ['4','5','6','7']
    assert manifest['models']['dlite_gdn']['gdn_reader_config']['layer_ids'] == [14,30]
    assert manifest['image_options'] == {'--image-max-tokens':'1024'}
