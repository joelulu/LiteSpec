"""Exact usable count, native preparation, deduplication and training reuse."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

import prepare_gdn_subset as subset
from litespec.vlm.train_aligned import reuse_validation


def source_fixture(tmp_path):
    source_dir = tmp_path/'source'
    source_dir.mkdir()
    rows = [dict(id=str(i), image=f'images/{i}.png', question=f'q{i}', answer='target answer') for i in range(12)]
    rows[1]['id'] = rows[0]['id']
    rows[2]['question'] = rows[0]['question']
    rows[2]['image'] = rows[0]['image']
    for i in (3,5,7):
        rows[i]['answer'] = 'invalid'
    source = source_dir/'train.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    target = tmp_path/'target'
    target.mkdir()
    (target/'config.json').write_text('{}')
    return rows, source, target


def fake_validator(row):
    return dict(reason='short_answer' if row['answer']=='invalid' else None,
                input_tokens=48, image_tokens=16, answer_tokens=24)


def test_collect_fills_exact_count_after_filtering_and_reuses_eligibility(tmp_path):
    rows, source, target = source_fixture(tmp_path)
    directory = tmp_path/'subset'
    result = subset.collect(source,directory,target,count=6,workers=1,validator=fake_validator)
    output = [json.loads(s) for s in (directory/'train.jsonl').read_text().splitlines()]
    assert len(output)==len({r['id'] for r in output})==6
    assert all(r['answer'] != 'invalid' for r in output)
    assert all(Path(r['image']).is_absolute() for r in output)
    assert result['stats']['scanned']>6
    assert result['stats']['selected']==6
    # Selection is reproducible and independent of file order (seeded permutation).
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    ids, prompts, expected = set(),set(),[]
    for index in order:
        r=rows[index]
        key=(r['image'],r['question'])
        if r['answer']=='invalid' or r['id'] in ids or key in prompts:
            continue
        ids.add(r['id']);prompts.add(key);expected.append(r['id'])
        if len(expected)==6:break
    assert [r['id'] for r in output]==expected
    cfg=json.loads((directory/'effective_config.json').read_text())
    packet=reuse_validation(output,cfg,directory)
    assert packet['valid']==list(range(6)) and packet['skipped']==[]
    assert (directory/'skipped_training.jsonl').read_text()==''
    calls=[]
    again=subset.collect(source,directory,target,count=6,workers=1,validator=lambda r:calls.append(r))
    assert again==result and calls==[]
    # Changing source/cap/template invalidates an existing selection, never overwrites it.
    before=(directory/'train.jsonl').read_bytes()
    with pytest.raises(ValueError,match='changed'):
        subset.collect(source,directory,target,count=6,workers=1,image_max_tokens=512,validator=fake_validator)
    (target/'chat_template.jinja').write_text('a new template')
    with pytest.raises(ValueError,match='changed'):
        subset.collect(source,directory,target,count=6,workers=1,validator=fake_validator)
    assert (directory/'train.jsonl').read_bytes()==before


def test_insufficient_rows_never_publish_a_training_manifest(tmp_path):
    _, source, target=source_fixture(tmp_path)
    directory=tmp_path/'not_enough'
    with pytest.raises(ValueError,match='need exactly 20'):
        subset.collect(source,directory,target,count=20,workers=1,validator=fake_validator)
    assert not (directory/'train.jsonl').exists()
    assert not (directory/'selection.json').exists()
    assert not (directory/'effective_config.json').exists()
    assert json.loads((directory/'incomplete.json').read_text())['selected']<20
    with pytest.raises(ValueError,match='separate'):
        subset.collect(source,source.parent,target,count=2,workers=1,validator=fake_validator)


def test_real_image_and_anchor_validation_uses_processor_config_only(monkeypatch,tmp_path):
    from PIL import Image
    import transformers
    from test_vlm import tiny_qwen35
    from test_vlm_aligned import Processor
    target_config=tiny_qwen35().config
    processor=Processor()
    processor.image_processor=SimpleNamespace(size=dict(shortest_edge=16,longest_edge=65536))
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**k:target_config)
    monkeypatch.setattr(transformers.AutoProcessor,'from_pretrained',lambda *a,**k:processor)
    subset.init_validator('local-config-only',4096,4,1024)
    image=tmp_path/'image.png';Image.new('RGB',(4,4)).save(image)
    record=dict(id='real',image=str(image),question='describe',answer='a square')
    result=subset.validate_record(record)
    assert result['reason'] is None and result['answer_tokens']==9
    assert result['image_tokens']==4
    subset._OPTIONS['block_size']=8
    assert subset.validate_record(record)['reason']=='insufficient_supervised_answer_or_full_block_anchor'
    subset._OPTIONS.update(block_size=4,max_length=12)
    assert subset.validate_record(record)['reason']=='ValueError'
    subset._OPTIONS['max_length']=4096
    assert subset.validate_record(dict(record,image=str(tmp_path/'missing.png')))['reason']=='FileNotFoundError'
    broken=tmp_path/'broken.png';broken.write_text('not an image')
    assert subset.validate_record(dict(record,image=str(broken)))['reason']=='UnidentifiedImageError'
    assert subset.validate_record(dict(record,image=''))['reason']=='missing_single_image'
    assert subset.validate_record(dict(record,split='val'))['reason']=='non_train_split'
    assert subset.validate_record(dict(record,answer=''))['reason']=='ValueError'
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 1)
    assert subset.validate_record(record)['reason']=='DecompressionBombError'


def test_bounded_parallel_results_follow_seeded_order(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    # Exercise production scheduling without requiring a saved HF tokenizer.
    # Each task still has an independent Future and initialization step.
    class Pool(ThreadPoolExecutor):
        def __init__(self,max_workers,mp_context,initializer,initargs):
            super().__init__(max_workers=max_workers)
    monkeypatch.setattr(subset,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(subset,'validate_record',lambda r:dict(reason=None,id=r['id']))
    rows=[dict(id=i) for i in range(30)]
    order=list(reversed(range(30)))
    results=list(subset.ordered_checks(rows,order,workers=4,init_args=()))
    assert [i for i,_ in results]==order
    assert [r['id'] for _,r in results]==order


@pytest.mark.parametrize('fail',[False,True])
def test_prepare_only_and_failed_preparation_cannot_start_training(tmp_path,fail):
    import os
    import subprocess
    import sys
    shim=tmp_path/'python'
    shim.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args[:1] != ['tests/prepare_gdn_subset.py']:
    raise RuntimeError('training should not have started')
Path(os.environ['PREP_LOG']).write_text(json.dumps(args))
sys.exit(17 if os.environ['FAIL_PREP']=='1' else 0)
''')
    shim.chmod(0o755)
    env=dict(os.environ,PYTHON_BIN=str(shim),PREPARE_ONLY='1',PREP_LOG=str(tmp_path/'prepare.json'),
             FAIL_PREP='1' if fail else '0')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash','tests/run_gdn_20k.sh'],cwd=root,env=env,text=True,capture_output=True)
    assert result.returncode==(17 if fail else 0)
    args=json.loads((tmp_path/'prepare.json').read_text())
    assert args[args.index('--count')+1]=='20000'
    assert args[args.index('--workers')+1]=='4'
