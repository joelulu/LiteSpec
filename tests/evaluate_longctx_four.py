"""Four independent GPU queues: 16 long multimodal samples, AR/Full/Pivot1."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from prepare_multimodal_longctx import SETS, prepare_video, prepare_documents
from prepare_longctx_data import prepare as prepare_math, verify_dataset, sha256, jsonl
from evaluate_dflash_all import fingerprint, write_json


def local_shard(args, name, explicit):
    if explicit is not None:return explicit.expanduser().resolve(strict=True)
    folders=[args.data_dir,args.data_dir/'mathvista',args.data_dir/'govreport',
        args.workspace/'data/longctx_sources/mathvista',args.workspace/'data/longctx_sources/govreport']
    paths=[folder/name for folder in folders if (folder/name).is_file()]
    if not paths:raise FileNotFoundError(f'Local shard missing: {name}; pass its --mathvista-parquet/--govreport-parquet path')
    return paths[0].resolve()


def models(args):
    target=fingerprint(args.target_model)
    drafts={}
    for name in ('full','pivot1'):
        drafts[name]=fingerprint(args.run_group/name/args.checkpoint_name)
        cfg=json.loads((Path(drafts[name]['path'])/'config.json').read_text())
        if cfg.get('architectures')!=['DFlashDraftModel'] or cfg.get('vlm_config',{}).get('draft_context','full')!=name:
            raise ValueError(f'Wrong architecture/context for {name}')
    configs=[json.loads((Path(v['path'])/'config.json').read_text()) for v in drafts.values()]
    if len({c['block_size'] for c in configs})!=1:raise ValueError('Full/Pivot1 block sizes differ')
    return target,drafts


def select_inputs(args, name):
    """Separate outputs from downloaded sources and any existing 100-sample pool."""
    prep=args.output_dir/'prepared'
    prep.mkdir(exist_ok=True)
    from types import SimpleNamespace
    if name=='mathvista-govreport':
        source=args.mathvista_source or args.workspace/'data/longctx_mathvista100'
        if (source/'manifest.json').is_file():
            rows,reports,_=verify_dataset(source)
        elif (args.data_dir/name/'manifest.json').is_file():
            source=args.data_dir/name
            rows,reports,_=verify_dataset(source)
        else:
            source=prep/name
            mv=local_shard(args,'testmini-00000-of-00001-725687bf7a18d64b.parquet',args.mathvista_parquet)
            gov=local_shard(args,'test-00000-of-00001.parquet',args.govreport_parquet)
            prepare_math(SimpleNamespace(output_dir=source,samples=args.samples,cache_dir=args.data_dir/'hf_cache',
                mathvista_parquet=mv,govreport_parquet=gov))
            rows,reports,_=verify_dataset(source)
        if len(rows)<args.samples:raise ValueError('Not enough MathVista samples')
        from evaluate_dflash_longctx import processor_encoder,build_inputs
        tokenizer,encode,_,_=processor_encoder(str(args.target_model),args.image_max_tokens)
        cfg=json.loads((args.target_model/'config.json').read_text())
        limit=cfg.get('text_config',cfg)['max_position_embeddings']
        dest=prep/'mathvista_encoded';dest.mkdir()
        paths,ids=build_inputs(SimpleNamespace(samples=args.samples,lengths=args.lengths,seed=args.seed,
            context_limit=limit,max_new_tokens=args.max_new_tokens),dest,rows,reports,tokenizer,encode)
        return [(('short' if length==0 else f'ctx{length}'),path) for length,path in paths.items()]
    if name=='MMLongBench-Doc':
        raw=args.data_dir/'mmlongbench_doc'
        doc_parquet=args.doc_parquet or raw/'train-00000-of-00001.parquet'
        doc_root=args.doc_root or raw/'documents'
        if not doc_parquet.is_file() or not doc_root.is_dir():raise FileNotFoundError(f'Require {doc_parquet} and {doc_root}')
        settings=SimpleNamespace(output_dir=prep,samples=args.samples,seed=args.seed,doc_parquet=doc_parquet,
            doc_root=doc_root,doc_revision='local-files',cache_dir=args.data_dir/'hf_cache',doc_page_edge=args.doc_page_edge)
        prepare_documents(settings)
        return [('all_pages',prep/name/'base.jsonl')]
    settings=SimpleNamespace(output_dir=prep,video_root=args.video_root,samples=args.samples,seed=args.seed,
        frames=args.frames,video_max_pixels=args.video_max_pixels)
    prepare_video(settings,name)
    return [(f'frames{f}',prep/name/f'frames_{f}.jsonl') for f in args.frames]


def case_command(args, name, variant, path, count, output):
    video=name in ('Video-MME','VideoDetailCaption')
    command=[sys.executable,str(HERE/('benchmark_video.py' if video else 'benchmark_vlm_static_batch.py')),
        '--target-model',str(args.target_model),'--output-dir',str(output),'--batch-sizes','1',
        '--max-new-tokens',str(args.max_new_tokens),'--seed',str(args.seed),'--warmup','1','--prefill-hidden-mode',getattr(args,'prefill_hidden_mode','selected'),'--sample-memory']
    for label in ('full','pivot1'):
        command+=['--checkpoint',label,str((args.run_group/label/args.checkpoint_name).resolve())]
    if video:
        rows=jsonl(path)[:count]
        if len(rows)!=count:raise ValueError(f'{name}: need {count} samples, got {len(rows)}')
        snapshot=output.parent/(output.name+'_selection.json')
        write_json(snapshot,{name:rows})
        command+=['--workspace',str(args.workspace),'--data-root',str(args.video_root),'--datasets',name,
            '--selection-file',str(snapshot),'--samples',str(count),'--frames',variant.removeprefix('frames'),
            '--max-pixels-per-frame',str(args.video_max_pixels)]
    else:
        cfg=json.loads((args.target_model/'config.json').read_text())['vision_config']
        pixels=args.image_max_tokens*(cfg['patch_size']*cfg['spatial_merge_size'])**2
        command+=['--data',str(path),'--dataset-name',name,'--max-samples',str(count),
            '--image-max-pixels',str(pixels),'--cpu-inputs','--attention-backend','sdpa',
            '--state-mode','per_token']
        if name=='MMLongBench-Doc':command+=['--document-input']
    return command


def expected_variants(args, name):
    if name == 'mathvista-govreport':
        return ['short' if n == 0 else f'ctx{n}' for n in args.lengths]
    if name == 'MMLongBench-Doc':return ['all_pages']
    return [f'frames{n}' for n in args.frames]


def reused_inputs(args, name):
    source = args.reuse_inputs_from
    variants = []
    for variant in expected_variants(args, name):
        path = source/'inputs'/f'{name}_{variant}.jsonl'
        if not path.is_file():
            # Prior workers materialize every variant before starting GPU tests.
            if name == 'mathvista-govreport':
                path = source/'prepared/mathvista_encoded/inputs'/f'{variant}.jsonl'
            elif name in ('Video-MME','VideoDetailCaption'):
                path = source/'prepared'/name/f'frames_{variant.removeprefix("frames")}.jsonl'
            else:path = source/'prepared'/name/'base.jsonl'
        if not path.is_file():raise FileNotFoundError(f'Missing frozen {name}/{variant}: {path}')
        variants.append((variant,path))
    return variants


def freeze_inputs(args, name):
    variants = reused_inputs(args,name) if getattr(args,'reuse_inputs_from',None) else select_inputs(args,name)
    if [v for v,_ in variants] != expected_variants(args,name):raise ValueError('Variant plan differs')
    source_rows=jsonl(variants[0][1])
    ids=[r['id'] for r in source_rows[:args.samples]]
    if len(ids)!=args.samples or len(set(ids))!=len(ids):raise ValueError('Need requested unique samples')
    write_json(args.output_dir/'inputs'/f'{name}_ids.json',ids)
    plan=[]
    for variant,path in variants:
        rows=jsonl(path)[:args.samples]
        if [r['id'] for r in rows] != ids:raise ValueError('Length/frame variants have different sample IDs')
        dest=args.output_dir/'inputs'/f'{name}_{variant}.jsonl'
        dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        smoke_indices = (sorted(range(len(rows)), key=lambda i: (-len(rows[i].get('images',[])),i))
                         if name == 'MMLongBench-Doc' else list(range(len(rows))))[:args.smoke_samples]
        smoke_rows = [rows[i] for i in smoke_indices]
        smoke_path = args.output_dir/'inputs'/f'{name}_{variant}_smoke.jsonl'
        smoke_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke_rows))
        plan.append(dict(variant=variant,path=str(dest),sha256=sha256(dest),
            smoke_path=str(smoke_path),smoke_sha256=sha256(smoke_path),smoke_ids=[r['id'] for r in smoke_rows],
            smoke_policy='most_pages' if name=='MMLongBench-Doc' else 'first_selected',
            source_path=str(path),source_sha256=sha256(path)))
        print(f'{name}/{variant}: {args.samples} fixed samples; sha256={sha256(dest)}',flush=True)
    write_json(args.output_dir/'inputs'/f'{name}_plan.json',dict(samples=args.samples,seed=args.seed,ids=ids,cases=plan))
    return plan


def checked_plan(args,name):
    plan=json.loads((args.output_dir/'inputs'/f'{name}_plan.json').read_text())
    if plan['samples']!=args.samples or plan['seed']!=args.seed:raise ValueError('Frozen sample settings changed')
    if [r['variant'] for r in plan['cases']]!=expected_variants(args,name):raise ValueError('Frozen variants changed')
    for item in plan['cases']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError(f'Frozen input changed: {path}')
        if [r['id'] for r in jsonl(path)]!=plan['ids']:raise ValueError('Frozen IDs changed')
        smoke_path=Path(item['smoke_path'])
        if sha256(smoke_path)!=item['smoke_sha256']:raise ValueError('Smoke input changed')
        if [r['id'] for r in jsonl(smoke_path)]!=item['smoke_ids']:raise ValueError('Smoke IDs changed')
        if len(item['smoke_ids'])!=args.smoke_samples or not set(item['smoke_ids'])<=set(plan['ids']):
            raise ValueError('Smoke samples must come from the formal sample set')
    return plan['cases']


def validate_cell(output,count):
    rows=json.loads((output/'summary.json').read_text())
    if len(rows)!=3 or {r['model'] for r in rows}!={'ar','full','pivot1'}:
        raise ValueError(f'Incomplete AR/Full/Pivot1 results: {output}')
    if any(r['batch_size']!=1 or r['requests']!=count or r.get('state_mode')!='per_token'
           or r.get('prefill_hidden_mode')!='selected' for r in rows):
        raise ValueError(f'Unexpected sample/batch/state/hidden settings: {output}')
    # AR token matches are recorded but are not a smoke gate for raw speed diagnostics.


def worker(args):
    name=args.worker
    phase=args.worker_phase
    if phase=='smoke':
        freeze_inputs(args,name)
    elif not (args.output_dir/'smoke_passed.json').is_file():
        raise RuntimeError('Formal evaluation requires all-dataset smoke success')
    cases=checked_plan(args,name)
    count=args.smoke_samples if phase=='smoke' else args.samples
    for item in cases:
        variant=item['variant']
        path=Path(item['smoke_path'] if phase=='smoke' else item['path'])
        out=args.output_dir/('smoke' if phase=='smoke' else 'results')/f'{name}_{variant}'
        subprocess.run(case_command(args,name,variant,path,count,out),check=True,cwd=HERE.parent)
        validate_cell(out,count)


def run_gated(args, run_phase):
    write_json(args.output_dir/'phase.json',dict(phase='smoke',evaluation_started=False))
    smoke=run_phase('smoke')
    if set(smoke)!=set(args.tests) or any(s['status']!='success' for s in smoke.values()):
        write_json(args.output_dir/'phase.json',dict(phase='smoke_failed',evaluation_started=False))
        raise RuntimeError('Smoke failed; NO 16-sample evaluation started. Inspect logs/smoke_*.log')
    write_json(args.output_dir/'smoke_passed.json',dict(samples=args.smoke_samples,tests=args.tests,
        models=['ar','full','pivot1'],cases={name:checked_plan(args,name) for name in args.tests}))
    write_json(args.output_dir/'phase.json',dict(phase='smoke_passed',evaluation_started=False))
    print('ALL SMOKE CASES PASSED. Starting full fixed-sample evaluation.',flush=True)
    write_json(args.output_dir/'phase.json',dict(phase='evaluation',evaluation_started=True))
    evaluation=run_phase('evaluation')
    success=set(evaluation)==set(args.tests) and all(s['status']=='success' for s in evaluation.values())
    write_json(args.output_dir/'phase.json',dict(phase='complete' if success else 'evaluation_failed',evaluation_started=True))
    if not success:raise RuntimeError('Some evaluations failed; completed cells remain in comparison.md')



def report(root, directory="results", prefix="comparison"):
    rows=[]
    for path in sorted((root/directory).glob('*/summary.json')):
        case=path.parent.name
        try:
            summaries=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):
            continue
        # Include only complete, internally comparable cells; failures remain in status.json.
        if {r['model'] for r in summaries}!={'ar','full','pivot1'} or len(summaries)!=3:continue
        for r in summaries:
            video='decode_tok_s' in r
            tokens=r['output_tokens']
            total=r.get('total_seconds',r.get('wall_seconds'))
            rows.append(dict(case=case,model=r['model'],samples=r['requests'],
                input_tokens=r['mean_input_tokens'],committed_length=r['mean_committed_length'],
                accepted_draft=r['mean_accepted_draft_tokens'],
                decode_tok_s=r['decode_tok_s'] if video else r['output_tokens_per_second'],
                decode_speedup=r['raw_decode_ratio_vs_ar'] if video else r['raw_throughput_speedup_vs_ar'],
                end_to_end_speedup=r['raw_end_to_end_ratio_vs_ar'] if video else r['raw_end_to_end_speedup_vs_ar'],
                prefill_seconds=r['prefill_seconds'],end_to_end_tok_s=tokens/total,
                peak_gib=r['peak_allocated_gib'],matched=r['matched']))
    write_json(root/f'{prefix}.json',rows)
    if rows:
        with (root/f'{prefix}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Long multimodal context: AR / Full / Pivot1','',
        'B=1, greedy, per_token. Acceptance below is committed length INCLUDING the guaranteed Target token.',
        'Raw observed speedups; matches are recorded. End-to-end includes model prefill + decode, excludes CPU preprocessing and model loading.',
        'Video-MME uses the existing description protocol, not official QA accuracy.','',
        '| Case | Model | Input tokens (mean) | Accept | Decode tok/s | Decode speedup | E2E speedup | Prefill s (sum) | Peak GiB | Match |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        accept='—' if r['committed_length'] is None else f"{r['committed_length']:.3f}"
        lines.append(f"| {r['case']} | {r['model']} | {r['input_tokens']:.0f} | {accept} | {r['decode_tok_s']:.2f} | {r['decode_speedup']:.3f}× | {r['end_to_end_speedup']:.3f}× | {r['prefill_seconds']:.2f} | {r['peak_gib']:.2f} | {r['matched']}/{r['samples']} |")
    (root/f'{prefix}.md').write_text('\n'.join(lines)+'\n')
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--target-model',type=Path,required=True)
    p.add_argument('--run-group',type=Path,required=True)
    p.add_argument('--prefill-hidden-mode',choices=['selected','all'],default='selected')
    p.add_argument('--checkpoint-name',default='epoch_5')
    p.add_argument('--gpus',nargs='+',default=['0','1','2','3'])
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--smoke-samples',type=int,default=1)
    p.add_argument('--lengths',nargs='+',type=int,default=[0,8192,32768,65536])
    p.add_argument('--frames',nargs='+',type=int,default=[32,64,128])
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--video-max-pixels',type=int,default=448*448)
    p.add_argument('--doc-page-edge',type=int,default=896)
    p.add_argument('--video-root',type=Path,default=Path.home()/'LMM/liyt/datasets/eval_bench')
    for name in ('mathvista-source','mathvista-parquet','govreport-parquet','doc-parquet','doc-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--tests',nargs='+',choices=SETS,default=list(SETS))
    p.add_argument('--reuse-inputs-from',type=Path,help='Reuse fixed JSONL/prepared inputs from a previous four-dataset run')
    p.add_argument('--worker',choices=SETS,help=argparse.SUPPRESS)
    p.add_argument('--worker-phase',choices=['smoke','evaluation'],default='smoke',help=argparse.SUPPRESS)
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args(argv)
    if not 1<=args.samples<=100 or args.max_new_tokens<1 or not 1<=args.smoke_samples<=args.samples:p.error('invalid sample/output budget')
    if len(set(args.tests))!=len(args.tests) or len(args.gpus)!=len(set(args.gpus)):p.error('duplicate tests/GPUs')
    if not args.gpus or any(not g.isdecimal() for g in args.gpus):p.error('physical GPU IDs required')
    if any(f<2 or f%2 for f in args.frames):p.error('even video frame counts required')
    if len(set(args.frames))!=len(args.frames) or len(set(args.lengths))!=len(args.lengths) or any(n<0 for n in args.lengths):p.error('unique frame/length variants required')
    if min(args.image_max_tokens,args.video_max_pixels,args.doc_page_edge)<1:p.error('invalid image budgets')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.prefill_hidden_mode!='selected':p.error('This gated runner requires selected hidden capture; use standalone benchmark for all-mode controls')
    if args.report_only:report(args.output_dir);return
    if args.worker:worker(args);return
    target,drafts=models(args)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name in ('logs','inputs','results','smoke'):(args.output_dir/name).mkdir()
    write_json(args.output_dir/'manifest.json',dict(settings=vars(args),target=target,checkpoints=drafts,
        gpu_assignments={name:args.gpus[i%len(args.gpus)] for i,name in enumerate(args.tests)}))
    raw=list(sys.argv[1:] if argv is None else argv)
    processes=set();lock=threading.Lock();stop=threading.Event()
    queues={gpu:[] for gpu in args.gpus}
    for i,name in enumerate(args.tests):queues[args.gpus[i%len(args.gpus)]].append(name)
    def launch(name,gpu,phase):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(HERE.parent))
        logfile=args.output_dir/'logs'/f'{phase}_{name}.log'
        print(f'{phase}: {name}, physical GPU {gpu}: {logfile}',flush=True)
        with logfile.open('w') as log:
            with lock:
                if stop.is_set():return dict(status='cancelled',returncode=130,gpu=gpu)
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*raw,
                    '--worker',name,'--worker-phase',phase],cwd=HERE.parent,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes.add(child)
            try:code=child.wait()
            finally:
                with lock:processes.discard(child)
        return dict(status='success' if code==0 else 'failed',returncode=code,gpu=gpu,phase=phase,log=str(logfile))
    def run_phase(phase):
        statuses={}
        def queue(gpu,names):
            for name in names:
                status=launch(name,gpu,phase)
                with lock:
                    statuses[name]=status
                    write_json(args.output_dir/f'{phase}_status.json',statuses)
                    write_json(args.output_dir/'status.json',dict(phase=phase,datasets=statuses))
                    report(args.output_dir, 'smoke' if phase=='smoke' else 'results',
                           'smoke_comparison' if phase=='smoke' else 'comparison')
                print(f'{phase}: {name}: {status["status"]}; GPU={gpu}',flush=True)
        pool=ThreadPoolExecutor(max_workers=len(args.gpus))
        try:
            futures=[pool.submit(queue,gpu,names) for gpu,names in queues.items() if names]
            for future in as_completed(futures):future.result()
        except BaseException:
            stop.set()
            with lock:
                for child in list(processes):
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            raise
        finally:pool.shutdown(wait=True)
        return statuses
    run_gated(args,run_phase)
    report(args.output_dir)
    print(f'Results: {args.output_dir}/comparison.md',flush=True)


if __name__=='__main__':main()
