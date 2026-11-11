"""Preparation, exact-length fitting, model provenance and benchmark orchestration."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate_dflash_longctx as cli
import prepare_longctx_data as data


class Tokenizer:
    def encode(self,text,**kwargs):return list(text)
    def decode(self,ids,**kwargs):return ''.join(ids)


class DataTests(unittest.TestCase):
    def fixture(self):
        from PIL import Image
        png=io.BytesIO();Image.new('RGB',(2,3),'white').save(png,format='PNG')
        reference=json.loads((HERE/'longctx_reference.json').read_text())
        rows=[dict(pid=pid,query='Explain this image and give the answer.',answer='1',
                   decoded_image=dict(bytes=png.getvalue())) for pid in reference['mathvista_ids']]
        reports=[dict(report='A long report. '*100) for _ in range(max(reference['govreport_indices'])+1)]
        return rows,reports,reference

    def test_prepare_exactly_100_original_images_and_inspect_five(self):
        rows,reports,reference=self.fixture()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()) as output:
            root=data.write_dataset(Path(temp)/'dataset',rows,reports,reference,100,{'test':'fixture'})
            records,pool,manifest=data.verify_dataset(root,100)
            self.assertEqual(len(records),100);self.assertEqual(len(pool),50)
            self.assertEqual(len(list((root/'images').glob('*.png'))),100)
            self.assertEqual(len(json.loads((root/'inspection_first5.json').read_text())),5)
            self.assertIn('query -> question',output.getvalue())
            self.assertTrue(all(Path(r['image']).is_file() for r in records))
            self.assertEqual(records[0]['question'],rows[0]['query'])
            self.assertEqual(manifest['sample_ids'],[f'mathvista:{p}' for p in reference['mathvista_ids']])

    def test_existing_data_verified_instead_of_overwritten(self):
        rows,reports,reference=self.fixture()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=data.write_dataset(Path(temp)/'dataset',rows,reports,reference,100,{})
            args=argparse.Namespace(output_dir=root,samples=100)
            self.assertEqual(data.prepare(args),root)
            with self.assertRaises(FileExistsError):data.write_dataset(root,rows,reports,reference,100,{})
            (root/'images'/f"{reference['mathvista_ids'][0]}.png").write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed'):data.verify_dataset(root)

    def test_invalid_prompt_does_not_publish_partial_dataset(self):
        rows,reports,reference=self.fixture();rows[8]['query']=''
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp)/'dataset'
            with self.assertRaisesRegex(ValueError,'missing query'):data.write_dataset(root,rows,reports,reference,100,{})
            self.assertFalse(root.exists());self.assertEqual(list(Path(temp).glob('.longctx-*')),[])

    def test_uploaded_shards_prepare_100_without_importing_hub(self):
        rows,reports,_=self.fixture()
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            root=Path(temp)
            math_file=root/'mathvista.parquet';math_file.write_bytes(b'mathvista-test')
            report_file=root/'govreport.parquet';report_file.write_bytes(b'govreport-test')
            fake_pandas=types.ModuleType('pandas')
            def read_parquet(path,columns):
                selected=rows if path==math_file else reports
                return SimpleNamespace(to_dict=lambda orient:selected)
            fake_pandas.read_parquet=read_parquet
            args=argparse.Namespace(output_dir=root/'prepared',samples=100,cache_dir=root/'cache',
                mathvista_parquet=math_file,govreport_parquet=report_file)
            with patch.dict(sys.modules,{'pandas':fake_pandas,'huggingface_hub':None}):
                output=data.prepare(args)
            actual,_,manifest=data.verify_dataset(output,100)
            self.assertEqual(len(actual),100)
            for key,path in [('mathvista',math_file),('govreport',report_file)]:
                self.assertEqual(manifest['sources'][key]['resolved_revision'],'local-file')
                self.assertEqual(manifest['sources'][key]['sha256'],data.sha256(path))
            self.assertFalse(args.cache_dir.exists())


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.args=argparse.Namespace(samples=16,lengths=[0,256,1024,4096],seed=42,context_limit=131072,
            max_new_tokens=256,repeats=2,gpus=list(map(str,range(8))),profile_samples=2,
            workspace=self.root,source_eval_dir=None,models=list(cli.DEFAULT_MODELS))
        self.rows=[dict(id=f'q{i}',question='Explain this image.',image='/local/image.png') for i in range(100)]
        self.reports=[dict(id=f'r{i}',text=('Report '+str(i)+' ')*1000) for i in range(50)]
        self.encode=lambda row:(len(row['question'])+100,98)

    def test_fit_exact_physical_length_and_keep_question(self):
        text,length,cut=cli.fit_prefix(list('x'*10000),''.join,lambda q:len(q)+100,'Original question',8192)
        self.assertEqual(length,8192);self.assertTrue(text.endswith('Original question'))
        self.assertEqual(text[:cut],'x'*cut)
        with self.assertRaisesRegex(ValueError,'already uses'):
            cli.fit_prefix(list('x'*1000),''.join,lambda q:len(q)+100,'question',10)

    def test_build_only_16_common_ids_and_nested_filler_prefixes(self):
        with redirect_stdout(io.StringIO()):
            paths,ids=cli.build_inputs(self.args,self.root,self.rows,self.reports,Tokenizer(),self.encode)
        self.assertEqual(ids,[r['id'] for r in self.rows[:16]])
        records={length:data.jsonl(path) for length,path in paths.items()}
        for length,rows in records.items():
            self.assertEqual(len(rows),16);self.assertEqual([r['id'] for r in rows],ids)
            if length:self.assertTrue(all(self.encode(r)[0]==length for r in rows))
        for index in range(16):
            prefixes=[records[n][index]['question'].split(cli.SEPARATOR)[0] for n in self.args.lengths[1:]]
            self.assertTrue(prefixes[1].startswith(prefixes[0]));self.assertTrue(prefixes[2].startswith(prefixes[1]))

    def test_eight_jobs_same_inputs_reversed_model_order_and_two_sample_smoke(self):
        paths={n:self.root/f'{n}.jsonl' for n in self.args.lengths}
        models={label:dict(path=f'/weights/{label}') for label in cli.DEFAULT_MODELS}
        jobs=cli.make_jobs(self.args,self.root,paths,models,{'path':'/target'},1048576)
        self.assertEqual(len(jobs),8);self.assertEqual([j['gpu'] for j in jobs],self.args.gpus)
        self.assertTrue(all(j['samples']==16 for j in jobs))
        self.assertEqual(jobs[0]['models'],['ar','full','pivot1','pivot5','pivot12'])
        self.assertEqual(jobs[4]['models'],['ar','pivot12','pivot5','pivot1','full'])
        for job in jobs:
            self.assertEqual(job['command'].count('--checkpoint'),4)
            for label in cli.DEFAULT_MODELS:self.assertIn(f'/weights/{label}',job['command'])
        for a,b in zip(jobs[:4],jobs[4:]):self.assertEqual(a['snapshot'],b['snapshot'])
        smoke=cli.make_jobs(self.args,self.root,paths,models,{'path':'/target'},1048576,smoke=True)
        self.assertEqual([j['length_bin'] for j in smoke],[0,4096])
        self.assertTrue(all(j['samples']==2 and '--profile-decode' not in j['command'] for j in smoke))
        self.assertTrue(all(j['models']==['ar',*cli.DEFAULT_MODELS] for j in smoke))

    def model_fixture(self):
        group=self.root/'runs/dflash_exp1_10w_5ep_test'
        source=group/'eval_all_b1_n32';source.mkdir(parents=True)
        models={}
        for label in (*cli.DEFAULT_MODELS,'target'):
            path=group/label/'step_0001108';path.mkdir(parents=True)
            config=dict(architectures=['DFlashDraftModel'],max_position_embeddings=131072,
                        vlm_config=dict(draft_context=label))
            (path/'config.json').write_text(json.dumps(config));(path/'model.safetensors').write_bytes(b'weights')
            (path.parent/'epoch_5').symlink_to(path.name,target_is_directory=True)
            models[label]=cli.fingerprint(path)
        prior=dict(run_group=str(group),target=models.pop('target'),checkpoints=models)
        cli.write_json(source/'manifest.json',prior)
        return source,prior

    def test_models_come_from_matching_epoch5_exports(self):
        source,prior=self.model_fixture()
        chosen,models,target=cli.resolve_models(self.args)
        self.assertEqual(chosen,source);self.assertEqual(models,prior['checkpoints']);self.assertEqual(target,prior['target'])
        config=Path(models['pivot5']['path'])/'config.json'
        obj=json.loads(config.read_text());obj['max_position_embeddings']=2048;config.write_text(json.dumps(obj))
        prior['checkpoints']['pivot5']=cli.fingerprint(config.parent);cli.write_json(source/'manifest.json',prior)
        with self.assertRaisesRegex(ValueError,'configured context limit'):cli.resolve_models(self.args)

    def test_added_pivots_cannot_use_changed_or_wrong_context_exports(self):
        source,prior=self.model_fixture()
        for label in ('pivot1','pivot12'):
            with self.subTest(label=label):
                config=Path(prior['checkpoints'][label]['path'])/'config.json'
                original=config.read_text()
                obj=json.loads(original);obj['vlm_config']['draft_context']='pivot5'
                config.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError,'source evaluation must use the epoch-5 export'):
                    cli.resolve_models(self.args)
                prior['checkpoints'][label]=cli.fingerprint(config.parent)
                cli.write_json(source/'manifest.json',prior)
                with self.assertRaisesRegex(ValueError,'Wrong checkpoint architecture/context'):
                    cli.resolve_models(self.args)
                config.write_text(original)
                prior['checkpoints'][label]=cli.fingerprint(config.parent)
                cli.write_json(source/'manifest.json',prior)

    def test_missing_added_pivot_reports_source_instead_of_silently_omitting(self):
        source,prior=self.model_fixture()
        del prior['checkpoints']['pivot12'];cli.write_json(source/'manifest.json',prior)
        with self.assertRaisesRegex(ValueError,'pivot12: missing from source evaluation'):
            cli.resolve_models(self.args)

    def test_optional_subset_keeps_full_pivot5_comparison_available(self):
        self.model_fixture();self.args.models=['full','pivot5']
        _,models,target=cli.resolve_models(self.args)
        self.assertEqual(list(models),['full','pivot5'])
        jobs=cli.make_jobs(self.args,self.root,{n:self.root/f'{n}.jsonl' for n in self.args.lengths},models,target,1024)
        self.assertEqual(jobs[0]['models'],['ar','full','pivot5'])
        self.assertEqual(jobs[4]['models'],['ar','pivot5','full'])

    def test_smoke_failure_prevents_formal_jobs(self):
        source,prior=self.model_fixture()
        dataset=self.root/'data';dataset.mkdir();(dataset/'manifest.json').write_text('{}')
        calls=[]
        def execute(root,jobs,batch):calls.append(jobs);return 1
        argv=['--workspace',str(self.root),'--data-dir',str(dataset),'--output-dir',str(self.root/'out'),
              '--lengths','0','256','1024','4096']
        with patch.object(cli,'verify_dataset',return_value=(self.rows,self.reports,{})), \
             patch.object(cli,'processor_encoder',return_value=(Tokenizer(),self.encode,1048576,{})), \
             patch.object(cli,'execute',side_effect=execute),redirect_stdout(io.StringIO()):
            result=cli.main(argv)
        self.assertEqual(result,1);self.assertEqual(len(calls),1)
        self.assertTrue(all(j['samples']==2 for j in calls[0]))

    def test_comparison_reports_ratios_and_separate_profile_timings(self):
        job=dict(dataset='ctx4096_r1',length_bin=4096,repeat=1,gpu='0',output=str(self.root/'cell'))
        output=Path(job['output']);output.mkdir()
        summaries=[]
        for model,tokens in [('ar',50),('full',80),('pivot1',90),('pivot5',100),('pivot12',110)]:
            summaries.append(dict(model=model,mean_input_tokens=4096,output_tokens=4096,requests=16,
                mean_committed_length=None if model=='ar' else 3.0,first_candidate_hit_rate=.8,
                mean_accepted_draft_tokens=None if model=='ar' else 1.875,
                output_tokens_per_second=tokens,raw_throughput_speedup_vs_ar=tokens/50,
                prefill_seconds=10,decode_seconds=40,wall_seconds=50,peak_allocated_gib=20,matched=14))
        (output/'summary.json').write_text(json.dumps(summaries))
        profile=dict(first_stage_seconds={'draft':.1},later_stage_seconds={'draft':.2},later_stage_calls={'draft':2},
                     draft_cache=dict(projected_kv_storage_bytes=1048576,raw_hidden_storage_bytes=2097152))
        (output/'pivot5_b1_profile.jsonl').write_text(json.dumps(dict(result=dict(profile=profile)))+'\n')
        cli.write_json(self.root/'status.json',{'ctx4096_r1':{'status':'success'}})
        with redirect_stdout(io.StringIO()):cli.report(self.root,[job])
        rows=json.loads((self.root/'longctx_comparison.json').read_text())
        self.assertEqual([r['model'] for r in rows],['ar',*cli.DEFAULT_MODELS])
        self.assertTrue(all(r['requests']==16 for r in rows))
        self.assertEqual(next(r for r in rows if r['model']=='pivot12')['ratio_vs_full'],110/80)
        pivot=next(r for r in rows if r['model']=='pivot5')
        self.assertEqual(pivot['ratio_vs_full'],1.25)
        self.assertEqual(pivot['ac_len'],1.875)
        self.assertIsNone(next(r for r in rows if r['model']=='ar')['ac_len'])
        self.assertEqual((pivot['first_draft_ms'],pivot['later_draft_ms']),(100,100))
        self.assertEqual((pivot['final_draft_kv_mib'],pivot['final_raw_hidden_mib']),(1,2))
        self.assertTrue((self.root/'longctx_comparison.csv').is_file())
        self.assertIn('ac_len',(self.root/'longctx_comparison.csv').read_text().splitlines()[0])
        table=(self.root/'longctx_comparison.md').read_text()
        self.assertIn('| ac len | Committed |',table)
        self.assertIn('| 1.875 | 3.000 |',table)
        self.assertIn('| — | — |',table)

    def test_report_only_uses_saved_jobs_without_reading_data_or_loading_models(self):
        saved_jobs=[dict(dataset='ctx32768_r1',output='/old/summary')]
        cli.write_json(self.root/'manifest.json',dict(jobs=saved_jobs))
        with patch.object(cli,'report') as render, \
             patch.object(cli,'verify_dataset',side_effect=AssertionError('must not read dataset')), \
             patch.object(cli,'resolve_models',side_effect=AssertionError('must not load checkpoints')), \
             patch.object(cli,'execute',side_effect=AssertionError('must not rerun evaluation')):
            self.assertEqual(cli.main(['--report-only',str(self.root)]),0)
        render.assert_called_once_with(self.root,saved_jobs)


class ProfileTests(unittest.TestCase):
    def module(self):
        path=HERE.parent/'litespec/vlm/inference_profile.py'
        spec=importlib.util.spec_from_file_location('profile_under_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_first_and_later_draft_timings_keep_original_totals(self):
        module=self.module()
        timeline=module.DecodeTimeline('cpu',torch_module=object())
        with patch.object(module.time,'perf_counter',side_effect=[0.,.01,.03,.05]):
            timeline.mark('draft');timeline.mark('verify');timeline.mark('draft');timeline.mark(None)
        result=timeline.resolve(.05)
        self.assertAlmostEqual(result['stage_seconds']['draft'],.03)
        self.assertAlmostEqual(result['first_stage_seconds']['draft'],.01)
        self.assertAlmostEqual(result['later_stage_seconds']['draft'],.02)
        self.assertEqual(result['later_stage_calls']['draft'],1)

    def test_cache_snapshot_reports_storage_capacity_without_double_counting(self):
        class Storage:
            def data_ptr(self):return 1
            def nbytes(self):return 1000
        tensor=SimpleNamespace(untyped_storage=lambda:Storage(),shape=(1,1,320))
        cache=SimpleNamespace(layers=[SimpleNamespace(keys=tensor,values=tensor)],get_seq_length=lambda:64)
        state=self.module().draft_cache_snapshot(SimpleNamespace(cache=cache,hidden=tensor),SimpleNamespace(context_slots=5))
        self.assertEqual(state['projected_kv_storage_bytes'],1000)
        self.assertEqual(state['raw_hidden_storage_bytes'],1000)
        self.assertEqual(state['raw_hidden_positions'],1)
        self.assertEqual(state['context_slots'],5)


if __name__=='__main__':unittest.main()
