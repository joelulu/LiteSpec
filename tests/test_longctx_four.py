"""CPU tests for four-GPU plans, frozen input reuse and multi-page routing."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_longctx_four as cli
from prepare_multimodal_longctx import lines
from benchmark_vlm_static_batch import prepare_benchmark_example


class LongContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        target=self.root/'target';target.mkdir()
        (target/'config.json').write_text(json.dumps(dict(vision_config=dict(patch_size=16,spatial_merge_size=2))))
        self.args=argparse.Namespace(target_model=target,run_group=self.root/'models',checkpoint_name='epoch_5',
            workspace=self.root,data_dir=self.root/'data',output_dir=self.root/'out',video_root=self.root/'videos',
            max_new_tokens=256,seed=42,samples=16,smoke_samples=1,image_max_tokens=1024,
            video_max_pixels=200704,worker='MMLongBench-Doc',worker_phase='smoke',
            tests=['MMLongBench-Doc'],lengths=[0,8192,32768,65536],frames=[32,64,128],reuse_inputs_from=None)
        for name in ('inputs','smoke','results'):(self.args.output_dir/name).mkdir(parents=True)

    def test_document_command_uses_explicit_multi_page_and_per_token(self):
        command=cli.case_command(self.args,'MMLongBench-Doc','all_pages',self.root/'base.jsonl',16,self.root/'result')
        self.assertIn('--document-input',command)
        self.assertIn('--cpu-inputs',command)
        self.assertEqual(command[command.index('--state-mode')+1],'per_token')
        self.assertEqual([command[i+1] for i,v in enumerate(command) if v=='--checkpoint'],['full','pivot1'])
        self.assertEqual(command[command.index('--batch-sizes')+1],'1')
        self.assertEqual(command[command.index('--max-samples')+1],'16')

    def test_video_smoke_and_full_use_same_prefix_and_explicit_frames(self):
        path=self.root/'video.jsonl'
        rows=[dict(id=f'v{i}',video=f'/video/{i}.mp4',question='describe') for i in range(16)]
        lines(path,rows)
        for count in (2,16):
            output=self.args.output_dir/'results'/f'v{count}'
            command=cli.case_command(self.args,'Video-MME','frames128',path,count,output)
            selection=Path(command[command.index('--selection-file')+1])
            actual=json.loads(selection.read_text())['Video-MME']
            self.assertEqual(actual,rows[:count])
            self.assertEqual(command[command.index('--frames')+1],'128')

    def test_worker_phases_share_formal_selection_and_smoke_uses_most_pages(self):
        path=self.root/'source.jsonl'
        rows=[dict(id=str(i),question='Q',images=['/page.png']*(i+1)) for i in range(16)]
        lines(path,rows)
        with patch.object(cli,'select_inputs',return_value=[('all_pages',path)]),patch.object(cli.subprocess,'run') as run,patch.object(cli,'validate_cell'),redirect_stdout(io.StringIO()):
            cli.worker(self.args)
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands),1)
        self.assertEqual(commands[0][commands[0].index('--max-samples')+1],'1')
        smoke_path=Path(commands[0][commands[0].index('--data')+1])
        self.assertEqual([r['id'] for r in cli.jsonl(smoke_path)],['15'])
        self.assertEqual(list((self.args.output_dir/'results').iterdir()),[])
        self.args.worker_phase='evaluation'
        with self.assertRaisesRegex(RuntimeError,'all-dataset smoke'):
            cli.worker(self.args)
        (self.args.output_dir/'smoke_passed.json').write_text('{}')
        with patch.object(cli,'select_inputs',side_effect=AssertionError('must not resample')),patch.object(cli.subprocess,'run') as run,patch.object(cli,'validate_cell'):
            cli.worker(self.args)
        command=run.call_args.args[0]
        self.assertEqual(command[command.index('--max-samples')+1],'16')
        full_path=Path(command[command.index('--data')+1])
        self.assertEqual(cli.jsonl(full_path),rows)

    def test_failed_global_smoke_never_starts_formal_stage(self):
        self.args.tests=list(cli.SETS)
        phases=[]
        def run(phase):
            phases.append(phase)
            return {name:dict(status='failed' if name=='Video-MME' else 'success') for name in self.args.tests}
        with self.assertRaisesRegex(RuntimeError,'NO 16-sample'):
            cli.run_gated(self.args,run)
        self.assertEqual(phases,['smoke'])
        self.assertFalse((self.args.output_dir/'smoke_passed.json').exists())

    def test_all_four_smokes_finish_before_formal_phase(self):
        self.args.tests=list(cli.SETS)
        phases=[]
        def run(phase):
            phases.append(phase)
            if phase=='smoke':self.assertFalse((self.args.output_dir/'smoke_passed.json').exists())
            else:self.assertTrue((self.args.output_dir/'smoke_passed.json').exists())
            return {name:dict(status='success') for name in self.args.tests}
        with patch.object(cli,'checked_plan',return_value=[]),redirect_stdout(io.StringIO()):
            cli.run_gated(self.args,run)
        self.assertEqual(phases,['smoke','evaluation'])
        self.assertEqual(json.loads((self.args.output_dir/'phase.json').read_text())['phase'],'complete')

    def test_reuse_previous_materialized_variants_without_resampling(self):
        source=self.root/'old';self.args.reuse_inputs_from=source
        for frames in self.args.frames:
            path=source/'prepared/Video-MME'/f'frames_{frames}.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            lines(path,[dict(id=str(i),question='Q',video='/video.mp4') for i in range(16)])
        with patch.object(cli,'select_inputs',side_effect=AssertionError('no resampling')),redirect_stdout(io.StringIO()):
            cli.freeze_inputs(self.args,'Video-MME')
        cases=cli.checked_plan(self.args,'Video-MME')
        self.assertEqual(len(cases),3)
        self.assertTrue(all(c['smoke_ids']==['0'] for c in cases))
        Path(cases[0]['path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError,'changed'):cli.checked_plan(self.args,'Video-MME')

    def test_gate_rejects_partial_or_wrong_count_results_but_allows_mismatches(self):
        output=self.root/'cell';output.mkdir()
        rows=[dict(model=name,batch_size=1,requests=1,state_mode='per_token',prefill_hidden_mode='selected',matched=0)
              for name in ('ar','full','pivot1')]
        (output/'summary.json').write_text(json.dumps(rows));cli.validate_cell(output,1)
        rows[-1]['requests']=16
        (output/'summary.json').write_text(json.dumps(rows))
        with self.assertRaises(ValueError):cli.validate_cell(output,1)

    def test_report_keeps_raw_ratios_despite_mismatch_and_skips_partial(self):
        case=self.args.output_dir/'results'/'doc_all';case.mkdir()
        data=[]
        for name in ('ar','full','pivot1'):
            data.append(dict(model=name,requests=16,output_tokens=160,wall_seconds=4,prefill_seconds=1,
                mean_input_tokens=12345,mean_committed_length=None if name=='ar' else 3,
                mean_accepted_draft_tokens=None if name=='ar' else 2,
                output_tokens_per_second=50,raw_throughput_speedup_vs_ar=2,
                raw_end_to_end_speedup_vs_ar=1.5,peak_allocated_gib=10,matched=8))
        (case/'summary.json').write_text(json.dumps(data))
        partial=self.args.output_dir/'results'/'partial';partial.mkdir()
        (partial/'summary.json').write_text('[{')
        rows=cli.report(self.args.output_dir)
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[1]['decode_speedup'],2)
        self.assertIn('8/16',(self.args.output_dir/'comparison.md').read_text())

    def test_multi_page_records_cannot_silently_become_text_only(self):
        row=dict(id='doc',question='Q',images=['/page.png'])
        with self.assertRaisesRegex(ValueError,'--document-input'):
            prepare_benchmark_example(None,None,row,device='cpu',base_dir=self.root)
        with self.assertRaisesRegex(ValueError,'nonempty images'):
            prepare_benchmark_example(None,None,dict(question='Q'),device='cpu',base_dir=self.root,document_input=True)

    def test_document_processor_route_keeps_all_images(self):
        import torch
        from litespec.vlm.runtime import PreparedExample
        example=PreparedExample(dict(input_ids=torch.tensor([[1,2]])),2,torch.tensor([False,True]),'doc')
        row=dict(id='doc',question='Q',images=['a.png','b.png'])
        with patch('encode_multimodal_longctx.document_example',return_value=example) as prepare:
            result=prepare_benchmark_example(None,None,row,device='cpu',base_dir=self.root,document_input=True)
        self.assertEqual(prepare.call_args.args[2]['images'],[str(self.root/'a.png'),str(self.root/'b.png')])
        self.assertEqual(result.prefill_length,2)


if __name__=='__main__':unittest.main()
