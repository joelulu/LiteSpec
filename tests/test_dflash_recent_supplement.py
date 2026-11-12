"""Recipe provenance, mixed checkpoints and actual shell orchestration."""
from contextlib import redirect_stdout
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare_dflash_recent_supplement as prep
import evaluate_dflash_longctx as longctx
from evaluate_dflash_all import PLAN,fingerprint,write_json


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.workspace=self.root/'workspace'
        self.baseline=self.workspace/'runs/dflash_exp1_10w_5ep_fixture';self.baseline.mkdir(parents=True)
        target=self.workspace/'target';target.mkdir();(target/'config.json').write_text(json.dumps(dict(max_position_embeddings=131072)))
        (target/'model.safetensors').write_bytes(b'target')
        data=self.workspace/'data/train.jsonl';data.parent.mkdir();data.write_text('{"id":"x"}\n')
        self.cfg=dict(upstream_config_sha256=prep.digest(HERE.parent/'scripts/config/dflash.yaml'),
            algorithm='dflash',world_size=8,batch_size=4,accumulation_steps=8,num_epochs=5,
            block_size=8,num_draft_layers=5,num_anchors=512,image_max_tokens=1024,seed=42,sampler_seed=0,
            max_length=4096,learning_rate=.0006,weight_decay=0.,warmup_ratio=.04,max_grad_norm=1.,use_muon=False,
            loss_decay_gamma=4.,objective_chunk_blocks=512,train_data=str(data),target_model=str(target),
            data_sha256=prep.digest(data),eligible_samples=1,eligible_indices_sha256='fixture',debug_max_samples=0,
            debug_max_steps=None,report_to='tensorboard')
        old={}
        for name in prep.OLD:
            directory=self.baseline/name;directory.mkdir()
            cfg=dict(self.cfg,draft_context=name,output_dir=str(directory))
            write_json(directory/'effective_config.json',cfg)
            self.export(directory/'epoch_5',name);old[name]=fingerprint(directory/'epoch_5')
        write_json(self.baseline/'experiment.json',dict(suite='exp1_10w_5ep',global_batch=256))
        self.reuse=self.workspace/'preflight';self.reuse.mkdir()
        write_json(self.reuse/'effective_config.json',self.cfg);(self.reuse/'skipped_training.jsonl').write_text('')
        self.source=self.baseline/'eval_all_b1_n32';self.source.mkdir()
        settings=dict(samples=32,video_samples=32,batch_size=1,max_new_tokens=256,seed=42,
            image_max_tokens=1024,frames=16,max_pixels_per_frame=200704)
        jobs=[]
        for name,kind in PLAN:
            records=[dict(id=f'{name}:{i}',question='question') for i in range(32)]
            snapshot=self.source/(name+('.json' if kind=='video' else '.jsonl'))
            if kind=='video':write_json(snapshot,{name:records})
            else:snapshot.write_text(''.join(json.dumps(row)+'\n' for row in records))
            jobs.append(dict(dataset=name,kind=kind,samples=32,snapshot=str(snapshot),input_sha256=prep.digest(snapshot)))
        write_json(self.source/'manifest.json',dict(run_group=str(self.baseline),target=fingerprint(target),
            checkpoints=old,settings=settings,jobs=jobs))
        self.args=SimpleNamespace(baseline_run_group=self.baseline,output_dir=self.workspace/'runs/supplement',
            reuse_validation_from=self.reuse,source_eval_dir=None)
        self.old_bytes={p:p.read_bytes() for p in self.baseline.rglob('*') if p.is_file()}

    def export(self,path,name):
        path.mkdir(parents=True)
        write_json(path/'config.json',dict(architectures=['DFlashDraftModel'],block_size=8,
            max_position_embeddings=131072,vlm_config=dict(draft_context=name)))
        (path/'model.safetensors').write_bytes(b'weights-'+name.encode())

    def prepare(self):
        with patch.object(prep.subprocess,'check_output',return_value='fixture-sha'),redirect_stdout(io.StringIO()):
            return prep.prepare(self.args)

    def finish(self):
        for name in prep.NEW:
            directory=self.args.output_dir/name
            cfg=yaml.safe_load((directory/'config.yaml').read_text())
            write_json(directory/'effective_config.json',cfg);self.export(directory/'epoch_5',name)
        with redirect_stdout(io.StringIO()):return prep.finalize(self.args.output_dir)

    def test_clones_old_hyperparameters_and_only_prepares_three_fresh_configs(self):
        result=self.prepare()
        self.assertEqual(result['new_modes'],list(prep.NEW))
        for name in prep.NEW:
            cfg=yaml.safe_load((self.args.output_dir/name/'config.yaml').read_text())
            for key in prep.MATCH_KEYS:self.assertEqual(cfg.get(key),self.cfg.get(key),key)
            self.assertEqual(cfg['draft_context'],name);self.assertNotIn('init_checkpoint',cfg)
            self.assertTrue(cfg['save_every_epoch']);self.assertFalse(cfg['final_only'])
        for path,content in self.old_bytes.items():self.assertEqual(path.read_bytes(),content)
        with self.assertRaises(FileExistsError):self.prepare()

    def test_rejects_changed_training_data_before_creating_output(self):
        Path(self.cfg['train_data']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Training JSONL differs'):self.prepare()
        self.assertFalse(self.args.output_dir.exists())

    def test_real_training_config_loader_accepts_all_new_modes_and_keeps_recipe(self):
        self.prepare()
        path=HERE.parent/'litespec/vlm/train_aligned.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_config')
        namespace=dict(yaml=yaml,Path=Path,__file__=str(path),hashlib=hashlib)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        for name in prep.NEW:
            cfg=namespace['load_config'](self.args.output_dir/name/'config.yaml')
            self.assertEqual(cfg['draft_context'],name)
            for key in ('learning_rate','num_epochs','batch_size','accumulation_steps','seed','block_size'):
                self.assertEqual(cfg[key],self.cfg[key])

    def test_rejects_wrong_world_size_and_mismatching_preflight(self):
        config=self.baseline/'full/effective_config.json';old=prep.read(config)
        write_json(config,dict(old,world_size=4))
        with self.assertRaisesRegex(ValueError,'world_size'):self.prepare()
        write_json(config,old)
        write_json(self.reuse/'effective_config.json',dict(self.cfg,image_max_tokens=512))
        with self.assertRaisesRegex(ValueError,'Preflight mismatch: image_max_tokens'):self.prepare()

    def test_rejects_changed_frozen_samples_before_training(self):
        (self.source/'textvqa.jsonl').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Frozen dataset preparation failed'):self.prepare()

    def test_finalizer_checks_all_new_recipes_and_mixed_epoch5_models(self):
        self.prepare();result=self.finish()
        self.assertEqual(list(result['all_checkpoints']),list(prep.ALL))
        args=SimpleNamespace(workspace=self.workspace,source_eval_dir=self.source,models=list(prep.ALL),
            supplement_run_group=self.args.output_dir,lengths=[0,8192,32768,65536],max_new_tokens=256)
        _,models,_=longctx.resolve_models(args)
        self.assertEqual(list(models),list(prep.ALL))
        for name in prep.ALL:
            source=self.args.output_dir if name in prep.NEW else self.baseline
            self.assertEqual(models[name]['path'],str(source/name/'epoch_5'))
        # Never substitute an unrecorded or changed export.
        (self.args.output_dir/'recent0/epoch_5/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'finalized with matching epoch-5'):longctx.resolve_models(args)

    def test_finalizer_rejects_changed_effective_training_recipe(self):
        self.prepare();self.finish()
        path=self.args.output_dir/'recent64/effective_config.json'
        write_json(path,dict(prep.read(path),learning_rate=.01))
        with self.assertRaisesRegex(ValueError,'Completed training recipe mismatch'):prep.finalize(self.args.output_dir)

    def exercise_shell(self,fail_check=False,fail_mode=''):
        repo=self.root/'repo';(repo/'tests').mkdir(parents=True);(repo/'scripts/dlite').mkdir(parents=True)
        for name in ('train_dflash_recent_supplement.sh','evaluate_dflash_recent_supplement.sh',
                     'prepare_dflash_recent_supplement.py','evaluate_dflash_all.py'):
            shutil.copy2(HERE/name,repo/'tests'/name)
        (repo/'scripts/config').mkdir()
        shutil.copy2(HERE.parent/'scripts/config/dflash.yaml',repo/'scripts/config/dflash.yaml')
        bindir=self.root/'bin';bindir.mkdir();git=bindir/'git';git.write_text('#!/bin/sh\necho fixture-sha\n');git.chmod(0o755)
        events=self.root/'events.jsonl'
        proxy=self.root/'python'
        proxy.write_text(f'''#!{sys.executable}
import json,os,sys
argv=sys.argv[1:]
with open({str(events)!r},'a') as f:f.write(json.dumps(dict(event='python',args=argv,env=dict(os.environ)))+'\\n')
if argv[0]=='-':sys.stdin.read();sys.exit(0)
if argv[0].endswith('check_dflash_recent.py') or '--fsdp' in argv:
    sys.exit(8 if os.environ.get('FAIL_CHECK')=='1' else 0)
os.execv({sys.executable!r},[{sys.executable!r},*argv])
''');proxy.chmod(0o755)
        writer=self.root/'fake_train.py'
        writer.write_text('''import json,os
from pathlib import Path
import yaml
root=Path(os.environ['OUTPUT_DIR']);cfg=yaml.safe_load(Path(os.environ['CONFIG']).read_text())
mode=cfg['draft_context']
with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps(dict(event='train',mode=mode,env=dict(os.environ)))+'\\n')
if os.environ.get('FAIL_MODE')==mode:raise SystemExit(9)
(root/'effective_config.json').write_text(json.dumps(cfg))
export=root/'epoch_5';export.mkdir()
(export/'config.json').write_text(json.dumps(dict(architectures=['DFlashDraftModel'],max_position_embeddings=131072,vlm_config=dict(draft_context=mode))))
(export/'model.safetensors').write_bytes(b'weights')
(root/'last').symlink_to('epoch_5',target_is_directory=True)
''')
        train=repo/'scripts/dlite/run_training_vlm.sh';train.write_text(f'#!/bin/bash\n{sys.executable} {writer}\n')
        for script,event in [('evaluate_dflash_all.sh','evaluate_all'),('evaluate_dflash_longctx.sh','evaluate_long')]:
            (repo/'tests'/script).write_text(f'''#!/bin/bash
{sys.executable} - "$@" <<'PY'
import json,os,sys
with open(os.environ['EVENTS'],'a') as f:f.write(json.dumps(dict(event={event!r},args=sys.argv[1:],env=dict(os.environ)))+'\\n')
PY
''')
        longdata=self.workspace/'data/longctx_mathvista100';longdata.mkdir();(longdata/'manifest.json').write_text('{}')
        env=dict(os.environ,DLITE_WORKSPACE=str(self.workspace),PYTHON_BIN=str(proxy),BASELINE_RUN_GROUP=str(self.baseline),
            RECENT_RUN_GROUP=str(self.args.output_dir),REUSE_VALIDATION_FROM=str(self.reuse),EVENTS=str(events),
            PATH=str(bindir)+':'+os.environ['PATH'],FAIL_CHECK='1' if fail_check else '0',FAIL_MODE=fail_mode,
            INIT_CHECKPOINT='/stale',RESUME='/stale',MAX_STEPS='1',EPOCHS='99',TRAIN_DATA='/stale',
            MICRO_BATCH_SIZE='99',ACCUMULATION_STEPS='99',NPROC_PER_NODE='4',MAX_SAMPLES='16')
        result=subprocess.run(['bash',str(repo/'tests/train_dflash_recent_supplement.sh')],env=env,capture_output=True,text=True)
        rows=[json.loads(line) for line in events.read_text().splitlines()]
        for path,content in self.old_bytes.items():self.assertEqual(path.read_bytes(),content)
        return result,rows

    def test_real_launcher_runs_only_three_new_models_then_compares_all_eight(self):
        result,rows=self.exercise_shell()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        trains=[r for r in rows if r['event']=='train'];self.assertEqual([r['mode'] for r in trains],list(prep.NEW))
        for row in trains:
            env=row['env']
            self.assertEqual([env[k] for k in ('NPROC_PER_NODE','EPOCHS','MICRO_BATCH_SIZE','ACCUMULATION_STEPS')],['8','5','4','8'])
            self.assertTrue(all(k not in env for k in ('INIT_CHECKPOINT','RESUME','MAX_STEPS')))
        ev=next(r for r in rows if r['event']=='evaluate_all')
        self.assertEqual([ev['args'][i+1] for i,x in enumerate(ev['args']) if x=='--checkpoint'],list(prep.ALL))
        self.assertEqual(ev['env']['EVAL_REUSE_SELECTION_FROM'],str(self.source))
        long=next(r for r in rows if r['event']=='evaluate_long')
        self.assertEqual(long['args'][:4],['--samples','32','--repeats','1'])
        self.assertEqual(long['args'][-8:],list(prep.ALL))

    def test_check_failure_prevents_training_and_evaluation(self):
        result,rows=self.exercise_shell(fail_check=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(r['event'] in ('train','evaluate_all','evaluate_long') for r in rows))

    def test_training_failure_stops_before_remaining_models_and_evaluation(self):
        result,rows=self.exercise_shell(fail_mode='recent64')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual([r['mode'] for r in rows if r['event']=='train'],['recent256','recent64'])
        self.assertFalse(any(r['event'].startswith('evaluate') for r in rows))


if __name__=='__main__':unittest.main()
