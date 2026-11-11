"""Frozen Full-only interventions, shared AR and per-GPU dependency queues."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_full_visual as cli


class VisualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'runs/dflash_exp1_10w_5ep_mb4_acc8_test/eval_all_b1_n32'
        (self.source/'inputs').mkdir(parents=True)
        full = self.source.parent/'full/step_0001108'
        full.mkdir(parents=True)
        (full/'config.json').write_text(json.dumps(dict(architectures=['DFlashDraftModel'], vlm_config=dict(draft_context='full'))))
        (full/'model.safetensors').write_bytes(b'full-weights')
        target = self.root/'target';target.mkdir()
        (target/'config.json').write_text('{}')
        (target/'model.safetensors').write_bytes(b'target-weights')
        jobs = []
        for name in cli.DATASETS:
            path = self.source/'inputs'/f'{name}.jsonl'
            path.write_text(''.join(json.dumps(dict(id=f'{name}:{i}', image=f'/local/{i}.jpg', prompt='q'))+'\n' for i in range(32)))
            jobs.append(dict(dataset=name, kind='image', samples=32, snapshot=str(path),
                             input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.manifest = dict(settings=dict(samples=32, batch_size=1, max_new_tokens=256, seed=42),
                             target=cli.fingerprint(target), checkpoints=dict(full=cli.fingerprint(full)),
                             jobs=jobs, image_max_pixels=1048576)
        cli.write_json(self.source/'manifest.json',self.manifest)
        self.args = argparse.Namespace(workspace=self.root, source_eval_dir=None, output_dir=self.root/'out', gpus=['4','5','6','7'], scheduling='dataset')

    def test_full_only_frozen_inputs_and_ar_reference_dependencies(self):
        root, jobs = cli.prepare(self.args)
        self.assertEqual(len(jobs),15)
        for index, name in enumerate(cli.DATASETS):
            cases = jobs[index*3:(index+1)*3]
            self.assertEqual([j['dataset'] for j in cases],[name+'/'+c for c in cli.CONDITIONS])
            self.assertEqual({j['gpu'] for j in cases},{self.args.gpus[index%4]})
            frozen = Path(cases[0]['snapshot'])
            self.assertEqual(frozen.read_bytes(),(self.source/'inputs'/f'{name}.jsonl').read_bytes())
            for case, condition in zip(cases,cli.CONDITIONS):
                command = case['command']
                self.assertEqual(command[command.index('--checkpoint')+1:command.index('--checkpoint')+3],
                                 ['full',self.manifest['checkpoints']['full']['path']])
                self.assertEqual(command[command.index('--visual-condition')+1],condition)
                self.assertEqual(command[command.index('--max-samples')+1],'32')
                self.assertEqual(command[command.index('--batch-sizes')+1],'1')
                self.assertEqual(case['models'],['ar','full'])
                if condition=='correct':
                    self.assertNotIn('--ar-reference',command)
                else:
                    self.assertEqual(command[command.index('--ar-reference')+1],cases[0]['output'])

    def test_changed_input_fails_before_writing_output(self):
        Path(self.manifest['jobs'][0]['snapshot']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'source inputs changed'):cli.prepare(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_changed_checkpoint_fails_before_writing_output(self):
        (Path(self.manifest['checkpoints']['full']['path'])/'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Full checkpoint files changed'):cli.prepare(self.args)
        self.assertFalse(self.args.output_dir.exists())

    def test_wrong_context_rejected(self):
        full=Path(self.manifest['checkpoints']['full']['path'])
        (full/'config.json').write_text(json.dumps(dict(architectures=['DFlashDraftModel'],vlm_config=dict(draft_context='pivot1'))))
        self.manifest['checkpoints']['full']=cli.fingerprint(full)
        cli.write_json(self.source/'manifest.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'Full DFlash'):cli.prepare(self.args)

    def test_explicit_selection_and_auto_latest(self):
        other=self.root/'runs/dflash_exp1_10w_5ep_other/eval_all_b1_n32'
        other.mkdir(parents=True)
        (other/'manifest.json').write_text('{}')
        os.utime(other/'manifest.json',ns=(1,1))
        self.assertEqual(cli.source_directory(self.root,None),self.source)
        self.assertEqual(cli.source_directory(self.root,other),other)

    def test_real_subprocess_order_and_three_condition_report(self):
        root,jobs=cli.prepare(self.args)
        code=("import fcntl,json,os,pathlib,sys; "
              "out=pathlib.Path(sys.argv[1]); condition=sys.argv[2]; "
              "lock=open(sys.argv[4]+'/gpu'+os.environ['CUDA_VISIBLE_DEVICES']+'.lock','w'); "
              "fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); "
              "assert condition=='correct' or pathlib.Path(sys.argv[3]+'/ready').is_file(); "
              "out.mkdir(); (out/'summary.json').write_text(sys.argv[5]); (out/'ready').touch()")
        for job in jobs:
            condition=job['dataset'].split('/')[1]
            rows=[]
            for label in ('ar','full'):
                rows.append(dict(model=label,batch_size=1,requests=32,matched=19,
                    mean_committed_length=None if label=='ar' else 2.5,first_candidate_hit_rate=None if label=='ar' else .7,
                    output_tokens_per_second=80,raw_throughput_speedup_vs_ar=2,peak_allocated_gib=10))
            job['command']=[sys.executable,'-c',code,job['output'],condition,
                            str(Path(job['output']).parent/'correct'),str(root),json.dumps(rows)]
        self.assertEqual(cli.execute(root,jobs,1),0)
        path=cli.visual_report(root)
        report=path.read_text()
        self.assertEqual(report.count('19/32'),15)
        self.assertIn('| charxiv_val | 4 | wrong |',report)
        self.assertIn('| hr_bench_8k | 4 | mask |',report)

    def test_cli_dry_run_needs_no_model_runtime(self):
        result=subprocess.run([sys.executable,str(HERE/'evaluate_full_visual.py'),
            '--workspace',str(self.root),'--output-dir',str(self.args.output_dir),'--dry-run'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        manifest=json.loads((self.args.output_dir/'manifest.json').read_text())
        self.assertEqual(len(manifest['jobs']),15)
        self.assertEqual(manifest['settings']['scheduling'],'condition')
        self.assertEqual(set(job['gpu'] for job in manifest['jobs']),set(map(str,range(8))))
        self.assertTrue(all('--ar-reference' not in job['command'] for job in manifest['jobs']))

    def test_eight_gpu_condition_schedule_has_no_cross_gpu_ar_dependencies(self):
        self.args.scheduling='condition'
        self.args.gpus=list(map(str,range(8)))
        root,jobs=cli.prepare(self.args)
        self.assertEqual([job['gpu'] for job in jobs],list(map(str,range(8)))+list(map(str,range(7))))
        for index,name in enumerate(cli.DATASETS):
            cases=jobs[index*3:(index+1)*3]
            self.assertEqual(len({job['snapshot'] for job in cases}),1)
            self.assertTrue(all('--ar-reference' not in job['command'] for job in cases))
        # The first job on every GPU must be running simultaneously before any
        # can complete. This fails if a future change silently caps workers at 5.
        code='''import fcntl,json,os,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); root=pathlib.Path(sys.argv[2])
gpu=os.environ['CUDA_VISIBLE_DEVICES']
lock=open(root/('gpu'+gpu+'.lock'),'w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(root/('started'+gpu)).touch()
deadline=time.monotonic()+5
while len(list(root.glob('started*')))<8 and time.monotonic()<deadline: time.sleep(.01)
assert len(list(root.glob('started*')))==8
out.mkdir(); (out/'summary.json').write_text(sys.argv[3])
'''
        rows=[dict(model=label,batch_size=1,requests=32,matched=32,
            mean_committed_length=None if label=='ar' else 2.5,first_candidate_hit_rate=.7,
            output_tokens_per_second=80,raw_throughput_speedup_vs_ar=2,peak_allocated_gib=10)
            for label in ('ar','full')]
        for job in jobs:
            job['command']=[sys.executable,'-c',code,job['output'],str(root),json.dumps(rows)]
        self.assertEqual(cli.execute(root,jobs,1),0)
        path=cli.visual_report(root)
        self.assertIn('own same-GPU AR',path.read_text())


if __name__=='__main__': unittest.main()
