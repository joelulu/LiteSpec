"""Common-data provenance, mixed blocks, four-GPU launch order and failures."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import dflash_30w as cli
import evaluate_dflash_all as all_eval
import evaluate_dflash_longctx as long_eval


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.target=self.root/'target';self.target.mkdir()
        cli.write(self.target/'config.json',dict(text_config=dict(max_position_embeddings=131072)))
        (self.target/'model.safetensors').write_bytes(b'target')
        self.source=self.root/'generated.jsonl'
        self.records=[dict(id=f'q{i}',image='/image.png',question='Question',answer='Answer') for i in range(5)]
        self.source.write_text(''.join(json.dumps(r)+'\n' for r in self.records))
        self.common=self.root/'common'

    def common_fixture(self):
        packets={8:dict(valid=[0,1,2,3,4],skipped=[]),
                 16:dict(valid=[1,3,4],skipped=[dict(index=0,id='q0',reason='short'),dict(index=2,id='q2',reason='short')])}
        cfgs={b:dict(cli.RECIPE,block_size=b,target_model=str(self.target),data_sha256=cli.sha(self.source)) for b in (8,16)}
        def reuse(rows,cfg,directory):
            self.assertEqual(cfg['eligible_samples'],len(rows))
            self.assertEqual((directory/'skipped_training.jsonl').read_text(),'')
            self.assertEqual(cfg['data_sha256'],cli.sha(directory.parent/'train.jsonl'))
        return cli.publish_common(self.common,self.source,self.records,packets,cfgs,
            dict(target_model=str(self.target),target=all_eval.fingerprint(self.target),
                 source_files={str(self.source):cli.sha(self.source)},candidates=300000),reuse)

    def run_fixture(self,blocks=(8,16),contexts=('full','pivot1')):
        if not self.common.exists(): self.common_fixture()
        group=self.root/('run' if tuple(blocks)==(8,16) else 'run_b'+'_'.join(map(str,blocks)))
        if tuple(contexts)!=('full','pivot1'):group=group.with_name(group.name+'_'+'_'.join(contexts))
        with patch.object(cli.subprocess,'check_output',return_value='test-head\n'),redirect_stdout(io.StringIO()):
            cli.prepare_run(argparse.Namespace(common=self.common,target_model=self.target,run_group=group,blocks=blocks,contexts=contexts))
        for label,(mode,block) in cli.selected_variants(blocks,contexts).items():
            directory=group/label
            cfg=dict(cli.RECIPE,block_size=block,draft_context=mode,world_size=4,
                     target_model=str(self.target),train_data=str(self.common/'train.jsonl'),
                     data_sha256=cli.sha(self.common/'train.jsonl'),eligible_samples=3)
            cli.write(directory/'effective_config.json',cfg)
            for name in [*[f'epoch_{i}' for i in range(1,6)],'last']:
                export=directory/name;export.mkdir()
                cli.write(export/'config.json',dict(architectures=['DFlashDraftModel'],block_size=block,
                    max_position_embeddings=131072,vlm_config=dict(draft_context=mode)))
                (export/'model.safetensors').write_bytes(b'draft')
        return group

    def test_intersection_and_two_correct_block_caches(self):
        saved=self.common_fixture()
        rows=[json.loads(l) for l in (self.common/'train.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in rows],['q1','q3','q4'])
        self.assertEqual(saved['eligible_samples'],3)
        self.assertEqual(saved['individually_eligible'],{'8':5,'16':3})
        self.assertEqual(cli.verify_common(self.common,self.target)['files'],saved['files'])
        for b in (8,16):
            cfg=cli.read(self.common/f'validation_b{b}/effective_config.json')
            self.assertEqual(cfg['block_size'],b)
            self.assertEqual(cfg['train_data'],str(self.common/'train.jsonl'))

    def test_generation_is_complete_disjoint_and_target_generated(self):
        generated=self.root/'generated';(generated/'manifest').mkdir(parents=True)
        prompts=generated/'manifest/prompts.jsonl'
        prompts.write_text(''.join(json.dumps(dict(id=r['id']))+'\n' for r in self.records))
        for path in ('manifest/manifest_config.json','generation_config.json'):
            cli.write(generated/path,dict(manifest_sha256=cli.sha(prompts),target_model=str(self.target)))
        for name,indices in [('train',[0,1,2]),('val',[3]),('skipped',[4])]:
            (generated/f'{name}.jsonl').write_text(''.join(json.dumps(dict(self.records[i],
                split=name,answer_target=str(self.target)))+'\n' for i in indices))
        self.assertEqual(cli.verify_generation(generated,5),dict(train=3,val=1,skipped=1))
        path=generated/'skipped.jsonl';original=path.read_text();path.write_text('')
        with self.assertRaisesRegex(ValueError,'Incomplete'):cli.verify_generation(generated,5)
        path.write_text(original+original)
        with self.assertRaisesRegex(ValueError,'duplicate'):cli.verify_generation(generated,5)

    def test_position_counts_real_tiny_image_target_for_both_blocks(self):
        import torch
        from types import SimpleNamespace
        from test_vlm import tiny_qwen35,tiny_image_inputs
        from litespec.vlm.baselines import make_draft
        from litespec.vlm.benchmark import RequestPositionState
        from litespec.vlm.context import visual_positions
        from litespec.vlm.runtime import PreparedExample
        from litespec.vlm.static_batch import generate_static_batch
        from benchmark_vlm_static_batch import summarize_batches
        torch.set_num_threads(1);torch.manual_seed(7)
        # Hub CPU Transformers lacks the server's hybrid transaction extension.
        # Exercise the real VLM verifier/counters with full-attention Target layers
        # and a real native cache; production hybrid/FSDP remains a server check.
        from transformers import DynamicCache,Qwen3_5ForConditionalGeneration
        cfg=tiny_qwen35().config;cfg.text_config.layer_types=['full_attention']*4
        target=Qwen3_5ForConditionalGeneration(cfg).eval().requires_grad_(False)
        class RelativeCache(DynamicCache):
            is_croppable=True
            def crop(self,remove):super().crop(self.get_seq_length()+remove)
        factory=lambda t,c:RelativeCache(config=t.config)
        processor=SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(127)}))
        inputs=tiny_image_inputs()
        example=PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),'image')
        for label,(mode,block) in cli.VARIANTS.items():
            draft=make_draft(target,processor,dict(algorithm='dflash',draft_context=mode,
                num_draft_layers=1,block_size=block,target_layer_ids=[0,3],target_model='tiny',
                context_policy=dict(prefill_text='keep',prefill_vision='keep',decode_mode='all'))).eval()
            with torch.no_grad(),RequestPositionState(target).activate(), \
                    patch('litespec.vlm.static_batch.make_hybrid_cache',side_effect=factory):
                result=generate_static_batch(target,draft,example,max_new_tokens=20,state_mode='full_target_replay')
            self.assertEqual(len(result['draft_position_opportunities']),block-1)
            self.assertTrue(any(result['draft_position_opportunities']))
            for h,p,n in zip(result['draft_position_agreement_hits'],result['draft_position_prefix_hits'],result['draft_position_opportunities']):
                self.assertTrue(0<=p<=h<=n)
            summary=summarize_batches([result],1)
            self.assertEqual(len(summary['draft_position_verifier_agreement']),block-1)
            self.assertTrue(all(x is None or 0<=x<=1 for x in summary['draft_position_prefix_acceptance']))

    def test_changed_source_cache_or_target_is_rejected(self):
        self.common_fixture()
        for path in [self.source,self.common/'train.jsonl',self.common/'validation_b16/effective_config.json',self.target/'config.json']:
            old=path.read_bytes();path.write_bytes(old+b' ')
            with self.assertRaisesRegex(ValueError,'changed'):cli.verify_common(self.common,self.target)
            path.write_bytes(old)

    def test_all_recipes_are_fresh_and_batch_256(self):
        group=self.run_fixture()
        import yaml
        for label,(mode,block) in cli.VARIANTS.items():
            cfg=yaml.safe_load((group/label/'config.yaml').read_text())
            self.assertEqual(cfg['block_size'],block);self.assertEqual(cfg['draft_context'],mode)
            self.assertEqual(cfg['num_epochs'],5);self.assertEqual(4*cfg['batch_size']*cfg['accumulation_steps'],256)
            self.assertNotIn('init_checkpoint',cfg)
        self.assertEqual(set(cli.finalize(group)['checkpoints']),set(cli.VARIANTS))
        with self.assertRaises(FileExistsError),patch.object(cli.subprocess,'check_output',return_value='head'):
            cli.prepare_run(argparse.Namespace(common=self.common,target_model=self.target,run_group=group))

    def test_wrong_block_or_warm_start_fails_before_evaluation(self):
        group=self.run_fixture();path=group/'pivot1_b16/epoch_5/config.json'
        original=cli.read(path);cli.write(path,dict(original,block_size=8))
        with self.assertRaisesRegex(ValueError,'context/block'):cli.finalize(group)
        cli.write(path,original)
        path=group/'full_b8/effective_config.json';cfg=cli.read(path);cfg['init_checkpoint']='/old'
        cli.write(path,cfg)
        with self.assertRaisesRegex(ValueError,'Warm start'):cli.finalize(group)

    def test_long_context_resolves_all_four_actual_blocks(self):
        group=self.run_fixture()
        args=argparse.Namespace(suite_run_group=group,lengths=[0,8192,32768,65536],max_new_tokens=256)
        source,models,target=long_eval.resolve_models(args)
        self.assertEqual(source,group);self.assertEqual(list(models),list(cli.VARIANTS))
        self.assertEqual(target['path'],str(self.target))
        for label,item in models.items():
            self.assertEqual(cli.read(Path(item['path'])/'config.json')['block_size'],cli.VARIANTS[label][1])

    def test_ratios_use_matching_full_block(self):
        rows=[dict(model=m,tok=i+1) for i,m in enumerate(['ar',*cli.VARIANTS])]
        self.assertEqual(all_eval.matching_full(rows,rows[-1])['model'],'full_b16')
        self.assertEqual(all_eval.matching_full(rows,rows[2])['model'],'full_b8')

    def test_independent_b8_and_b16_groups_finalize_and_long_eval(self):
        for block in (8,16):
            group=self.run_fixture([block]);wanted={f'full_b{block}',f'pivot1_b{block}'}
            saved=cli.finalize(group)
            self.assertEqual(set(saved['checkpoints']),wanted)
            self.assertEqual(set(saved['variants']),wanted)
            self.assertFalse((group/f'full_b{24-block}').exists())
            args=argparse.Namespace(suite_run_group=group,lengths=[0,65536],max_new_tokens=256)
            _,models,_=long_eval.resolve_models(args)
            self.assertEqual(set(models),wanted)
            rows=[dict(model='ar'),dict(model=f'full_b{block}'),dict(model=f'pivot1_b{block}')]
            self.assertEqual(all_eval.matching_full(rows,rows[0])['model'],f'full_b{block}')

    def test_inputs_accept_existing_100_record_long_context_pool(self):
        import prepare_longctx_data
        args=argparse.Namespace(workspace=self.root,longctx_data_dir=self.root/'longctx100',
                                result_file=self.root/'eval.env')
        with patch.dict(os.environ,SOURCE_EVAL_DIR='/previous'), \
             patch.object(prepare_longctx_data,'verify_dataset',return_value=([{}]*100,[],{})) as verify, \
             patch.object(all_eval,'frozen_selection',return_value=({},dict(manifest='/previous/manifest.json'))), \
             redirect_stdout(io.StringIO()):
            cli.check_inputs(args)
        verify.assert_called_once_with(args.longctx_data_dir)
        self.assertIn('/previous',args.result_file.read_text())

    def exercise_shell(self,fail=None,blocks='8,16',gpus='0,1,2,3',contexts='full,pivot1'):
        repo=self.root/'shell';(repo/'tests').mkdir(parents=True)
        (repo/'scripts/data').mkdir(parents=True);(repo/'scripts/dlite').mkdir(parents=True)
        shutil.copy(HERE/'run_dflash_30w.sh',repo/'tests')
        log=self.root/'calls.jsonl';group=self.root/'shell-run';data=self.root/'data'
        python=repo/'fake-python'
        python.write_text('#!'+sys.executable+'\n'+'''import json,os,sys
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['CALLS'],'a') as h:h.write(json.dumps(dict(args=a,env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','DATA_GPUS','NPROC_PER_NODE','MICRO_BATCH_SIZE','ACCUMULATION_STEPS','MAX_SAMPLES','MAX_NEW_TOKENS','CONFIG','REUSE_VALIDATION_FROM')}))+'\\n')
if 'dflash_30w.py' in ' '.join(a):
 if a[1]=='inputs':Path(a[-1]).write_text('export EVAL_REUSE_SELECTION_FROM=\\n')
 if a[1]=='data':Path(os.environ['COMMON_DATA_DIR']).mkdir(parents=True)
 if a[1]=='run':
  root=Path(os.environ['RUN_GROUP']);root.mkdir()
  labels=[mode+'_b'+b for b in os.environ['BLOCK_SIZES'].split(',') for mode in os.environ['TRAIN_CONTEXTS'].split(',')]
  (root/'launch.env').write_text('export TRAIN_DATA='+os.environ['COMMON_DATA_DIR']+'/train.jsonl\\n'+'export RUN_VARIANTS="'+' '.join(labels)+'"\\n')
  for label in labels:(root/label).mkdir()
if os.environ.get('FAIL_STAGE') and os.environ['FAIL_STAGE'] in ' '.join(a):sys.exit(1)
''');python.chmod(0o755)
        for path in ['scripts/data/generate_finevision_8gpu.sh','scripts/dlite/run_training_vlm.sh',
                     'tests/evaluate_dflash_all.sh','tests/evaluate_dflash_longctx.sh']:
            dest=repo/path
            dest.write_text('#!/bin/bash\n"$PYTHON_BIN" "'+path+'" "$@"\n')
        raw=self.root/'raw';raw.write_text('source')
        env=dict(os.environ,PYTHON_BIN=str(python),TARGET_MODEL=str(self.target),SOURCE_DATA=str(raw),
            DATA_ROOT=str(data),SUITE_RUN_GROUP=str(group),CALLS=str(log),FAIL_STAGE=fail or '',
            MICRO_BATCH_SIZE='99',ACCUMULATION_STEPS='99',MAX_SAMPLES='7',INIT_CHECKPOINT='/old',
            DATA_GPUS='4,5,6,7',CUDA_VISIBLE_DEVICES='4,5,6,7',BLOCK_SIZES=blocks,TRAIN_GPUS=gpus,TRAIN_CONTEXTS=contexts)
        result=subprocess.run(['bash',str(repo/'tests/run_dflash_30w.sh'),'all'],env=env,capture_output=True,text=True)
        return result,[json.loads(l) for l in log.read_text().splitlines()]

    def test_shell_generation_precheck_four_training_then_evaluation(self):
        result,calls=self.exercise_shell();self.assertEqual(result.returncode,0,result.stderr)
        train=[r for r in calls if r['args'][0]=='scripts/dlite/run_training_vlm.sh']
        self.assertEqual(len(train),4)
        for row,label in zip(train,cli.VARIANTS):
            env=row['env'];self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1,2,3')
            self.assertEqual(env['NPROC_PER_NODE'],'4');self.assertEqual(env['MAX_SAMPLES'],'0')
            self.assertEqual((env['MICRO_BATCH_SIZE'],env['ACCUMULATION_STEPS']),('2','32'))
            self.assertTrue(env['CONFIG'].endswith(label+'/config.yaml'))
            self.assertTrue(env['REUSE_VALIDATION_FROM'].endswith('validation_b'+label.split('_b')[1]))
        regen=next(r for r in calls if r['args'][0]=='scripts/data/generate_finevision_8gpu.sh')
        self.assertEqual(regen['env']['MAX_SAMPLES'],'300000');self.assertEqual(regen['env']['MAX_NEW_TOKENS'],'256')
        self.assertEqual(regen['env']['DATA_GPUS'],'0,1,2,3')
        long=next(r for r in calls if r['args'][0]=='tests/evaluate_dflash_longctx.sh')
        self.assertIn('--suite-run-group',long['args']);self.assertIn('--repeats',long['args'])

    def test_failed_precheck_stops_all_training(self):
        result,calls=self.exercise_shell('check_dflash_blocks.py')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(r['args'][0]=='scripts/dlite/run_training_vlm.sh' for r in calls))

    def test_b8_only_shell_trains_and_evaluates_two_models(self):
        result,calls=self.exercise_shell(blocks='8')
        self.assertEqual(result.returncode,0,result.stderr)
        train=[r for r in calls if r['args'][0]=='scripts/dlite/run_training_vlm.sh']
        self.assertEqual([Path(r['env']['CONFIG']).parent.name for r in train],['full_b8','pivot1_b8'])
        job=next(r for r in calls if r['args'][0]=='tests/evaluate_dflash_all.sh')
        self.assertEqual(job['args'].count('--checkpoint'),2)
        self.assertNotIn('full_b16',job['args'])

    def test_b16_only_shell_uses_back_four_gpus(self):
        result,calls=self.exercise_shell(blocks='16',gpus='4,5,6,7')
        self.assertEqual(result.returncode,0,result.stderr)
        train=[r for r in calls if r['args'][0]=='scripts/dlite/run_training_vlm.sh']
        self.assertEqual([Path(r['env']['CONFIG']).parent.name for r in train],['full_b16','pivot1_b16'])
        self.assertTrue(all(r['env']['CUDA_VISIBLE_DEVICES']=='4,5,6,7' for r in calls))

    def test_missing_evaluation_inputs_stop_before_training(self):
        result,calls=self.exercise_shell('inputs')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(any(r['args'][0]=='scripts/dlite/run_training_vlm.sh' for r in calls))

    def test_context_groups_finalize_independently(self):
        for context in ('full','pivot1'):
            group=self.run_fixture(contexts=[context])
            self.assertEqual(set(cli.finalize(group)['checkpoints']),{context+'_b8',context+'_b16'})

    def test_full_only_front_gpu_launcher(self):
        result,calls=self.exercise_shell(contexts='full')
        self.assertEqual(result.returncode,0,result.stderr)
        rows=[r for r in calls if r['args'][0]=='scripts/dlite/run_training_vlm.sh']
        self.assertEqual([Path(r['env']['CONFIG']).parent.name for r in rows],['full_b8','full_b16'])

    def test_pivot_only_back_gpu_launcher(self):
        result,calls=self.exercise_shell(contexts='pivot1',gpus='4,5,6,7')
        self.assertEqual(result.returncode,0,result.stderr)
        rows=[r for r in calls if r['args'][0]=='scripts/dlite/run_training_vlm.sh']
        self.assertEqual([Path(r['env']['CONFIG']).parent.name for r in rows],['pivot1_b8','pivot1_b16'])
        self.assertTrue(all(r['env']['CUDA_VISIBLE_DEVICES']=='4,5,6,7' for r in rows))


if __name__=='__main__':unittest.main()
