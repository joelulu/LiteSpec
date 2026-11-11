"""Exercise real suite launchers with lightweight model-process stand-ins."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_variants_keep_context_gamma_and_epoch_exports(self):
        tree = ast.parse((ROOT/'tests/dflash_context_config.py').read_text())
        build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build')
        namespace = dict(os=os, Path=Path, yaml=yaml, CONTEXTS=('full','recent128','pivot1','pivot5','pivot12'))
        exec(compile(ast.Module(body=[build], type_ignores=[]), 'actual_config', 'exec'), namespace)
        for mode in ('full','recent128','pivot1','pivot5','pivot12','pivot5_B','pivot5_C'):
            with patch.dict(os.environ, dict(EPOCHS='5', SAVE_EVERY_EPOCH='1', MICRO_BATCH_SIZE='4', ACCUMULATION_STEPS='8'), clear=True):
                previous = os.getcwd()
                try:
                    os.chdir(ROOT)
                    cfg = namespace['build'](mode)
                finally:
                    os.chdir(previous)
            self.assertEqual(cfg['draft_context'], 'pivot5' if mode.endswith(('_B','_C')) else mode)
            self.assertEqual(cfg['context_projection'], 'per_layer' if mode.endswith(('_B','_C')) else 'shared')
            self.assertEqual(cfg['position_layer_bias'], mode == 'pivot5_C')
            self.assertEqual((cfg['block_size'],cfg['loss_decay_gamma'],cfg['num_epochs']), (8,4.0,5))
            self.assertTrue(cfg['save_every_epoch'])
            self.assertFalse(cfg['final_only'])
            self.assertNotIn('init_checkpoint',cfg)


class LaunchTests(unittest.TestCase):
    def exercise(self, preset, micro='4', fail=False, retry=False, fail_check=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo, workspace = root/'repo', root/'workspace'
            (repo/'tests').mkdir(parents=True)
            (repo/'scripts/dlite').mkdir(parents=True)
            for name in ('train_dflash_research_suite.sh','train_dflash_exp1.sh','train_dflash_exp2.sh','retry_dflash_exp2_c.sh'):
                shutil.copy2(ROOT/'tests'/name, repo/'tests'/name)
            for relative in ('data/finevision_qwen35_10w_img1024_relaxed/train.jsonl',
                             'data/finevision_qwen35_gdn_valid20k_img1024_b8_s42/train.jsonl'):
                path=workspace/relative;path.parent.mkdir(parents=True);path.write_text('{}\n')
            for relative in ('runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb',
                             'runs/dflash_context_20k_1ep_20261110_041837_1299222/full'):
                path=workspace/relative;path.mkdir(parents=True)
                for filename in ('effective_config.json','skipped_training.jsonl'):
                    (path/filename).touch()
            target=root/'target';target.mkdir()
            old_group=workspace/'runs/completed'
            old_files={}
            if retry:
                for mode in ('pivot5','pivot5_B','pivot5_C'):
                    path=old_group/mode/'last/config.json'
                    path.parent.mkdir(parents=True)
                    path.write_text('original '+mode)
                    old_files[path]=path.read_bytes()
            events=root/'events.jsonl'
            recorder=root/'record.py'
            recorder.write_text(
                'import json,os,sys\n'
                f'with open({str(events)!r},"a") as f: f.write(json.dumps(dict(event=sys.argv[1],args=sys.argv[2:],env=dict(os.environ)))+"\\n")\n')
            python=root/'python'
            python.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
argv=sys.argv[1:]
subprocess.run([sys.executable,{str(recorder)!r},'python',*argv],check=True)
if '--fsdp' in argv and os.environ.get('FAIL_DISTRIBUTED_CHECK')=='1': sys.exit(9)
if argv[0]=='-':
    code=sys.stdin.read()
    if 'import os, torch' not in code:
        exec(compile(code,'stdin','exec'))
elif argv[0].endswith('dflash_context_config.py'):
    pathlib.Path(argv[2]).write_text(argv[1])
''')
            python.chmod(0o755)
            bindir=root/'bin';bindir.mkdir()
            git=bindir/'git';git.write_text('#!/bin/sh\necho test-commit\n');git.chmod(0o755)
            train=repo/'scripts/dlite/run_training_vlm.sh'
            train.write_text(f'''#!/bin/bash
set -eu
{sys.executable} '{recorder}' train "$(cat "$CONFIG")"
if [[ "{str(fail)}" == True ]]; then exit 7; fi
mkdir -p "$OUTPUT_DIR/last" "$OUTPUT_DIR/epoch_$EPOCHS"
echo '{{}}' > "$OUTPUT_DIR/last/config.json"
''')
            evaluate=repo/'tests/evaluate_dflash_all.sh'
            evaluate.write_text(f'''#!/bin/bash
set -eu
{sys.executable} '{recorder}' evaluate "$@"
mkdir -p "$RESULT_DIR"
touch "$RESULT_DIR/comparison.md"
''')
            env=dict(os.environ, DLITE_WORKSPACE=str(workspace),PYTHON_BIN=str(python),TARGET_MODEL=str(target),
                     MICRO_BATCH_SIZE=micro,PATH=str(bindir)+':'+os.environ['PATH'],
                     # Deliberately polluted inputs must not change the preset.
                     INIT_CHECKPOINT='/stale',RESUME='/stale',MAX_STEPS='1',EPOCHS='99',
                     TRAIN_DATA='/stale',MAX_SAMPLES='16',ACCUMULATION_STEPS='99',
                     SAVE_EVERY_EPOCH='0',VIDEO_SAMPLES='16',EVAL_BATCH_SIZE='64',
                     FAILED_RUN_GROUP=str(old_group),FAIL_DISTRIBUTED_CHECK='1' if fail_check else '0')
            script='retry_dflash_exp2_c.sh' if retry else f'train_dflash_exp{preset}.sh'
            result=subprocess.run(['bash',str(repo/'tests'/script)],env=env,capture_output=True,text=True)
            rows=[json.loads(line) for line in events.read_text().splitlines()]
            trains=[r for r in rows if r['event']=='train']
            evaluations=[r for r in rows if r['event']=='evaluate']
            for path,content in old_files.items(): self.assertEqual(path.read_bytes(),content)
            if fail_check:
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(trains,[])
                self.assertEqual(evaluations,[])
                return
            if fail:
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(len(trains),1)
                self.assertEqual(evaluations,[])
                return
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            modes=['full','recent128','pivot1','pivot5','pivot12'] if preset==1 else ['pivot5','pivot5_B','pivot5_C']
            self.assertEqual([r['args'][0] for r in trains],['pivot5_C'] if retry else modes)
            for row in trains:
                e=row['env']
                self.assertEqual(e['EPOCHS'],'5')
                self.assertEqual(e['MAX_SAMPLES'],'0')
                self.assertEqual(e['TRAIN_DATA'],str(workspace/'data/finevision_qwen35_10w_img1024_relaxed/train.jsonl'))
                self.assertEqual(e['REUSE_VALIDATION_FROM'],str(workspace/'runs/preflight_10w_img1024_parallel_20261110_092423_ccec48bb'))
                self.assertEqual(e['NPROC_PER_NODE'],'8' if preset==1 else '4')
                self.assertEqual(e['ACCUMULATION_STEPS'],str(256//((8 if preset==1 else 4)*int(micro))))
                self.assertEqual(int(e['MICRO_BATCH_SIZE'])*int(e['NPROC_PER_NODE'])*int(e['ACCUMULATION_STEPS']),256)
                self.assertEqual(e['SAVE_EVERY_EPOCH'],'1')
                self.assertTrue(all(key not in e for key in ['INIT_CHECKPOINT','RESUME','MAX_STEPS']))
            self.assertEqual(len(evaluations),1)
            e=evaluations[0]['env']
            self.assertEqual((e['EVAL_BATCH_SIZE'],e['EVAL_SAMPLES'],e['VIDEO_SAMPLES']),('1','32','32'))
            self.assertEqual(len(e['EVAL_GPUS'].split()),8 if preset==1 else 4)
            args=evaluations[0]['args']
            self.assertEqual([args[i+1] for i,a in enumerate(args) if a=='--checkpoint'],modes)
            checks=[r for r in rows if r['event']=='python' and r['args'][0].endswith('check_dflash_layer_adapters.py')]
            self.assertEqual(len(checks),0 if preset==1 else 1)
            distributed=[r for r in rows if r['event']=='python' and '--fsdp' in r['args']]
            self.assertEqual(len(distributed),0 if preset==1 else 1)
            if preset==2:
                self.assertIn('--nproc-per-node=4',distributed[0]['args'])

    def test_long_preset(self): self.exercise(1)
    def test_adapter_preset_includes_pivot5_control_and_uses_10w_five_epochs(self): self.exercise(2)
    def test_smaller_micro_preserves_global_batch(self): self.exercise(1,micro='2')
    def test_failure_stops_before_other_training_and_eval(self): self.exercise(2,fail=True)
    def test_c_retry_trains_only_c_and_reuses_unmodified_old_checkpoints(self): self.exercise(2,retry=True)
    def test_distributed_check_failure_prevents_training(self): self.exercise(2,retry=True,fail_check=True)


if __name__=='__main__': unittest.main()
