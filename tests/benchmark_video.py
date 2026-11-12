"""Offline replicated-video batches: AR and three existing draft checkpoints."""
import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_benchmark_data import DATASETS, select_data


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)


def report(root):
    rows = [json.loads(p.read_text())['summary'] for p in sorted((root/'cells').glob('*.json'))]
    write_json(root/'summary.json', rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# Video benchmark', '',
        'Static replicated inputs; B is batch size, not online concurrency. Decode-only raw ratios.',
        'Video-MME is repurposed as video description here; no official QA accuracy is reported.',
        'Preprocessing/replication/host transfers are outside model timing. Token match does not gate ratios.', '',
        '| Dataset | Model | B | Committed length | Accepted draft | tok/s | Raw ratio / AR | Match | Peak GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda x: '—' if x is None else f'{x:.3f}'
        lines.append(f"| {r['dataset']} | {r['model']} | {r['batch_size']} | "
            f"{fmt(r['mean_committed_length'])} | {fmt(r['mean_accepted_draft_tokens'])} | "
            f"{r['decode_tok_s']:.2f} | {r['raw_decode_ratio_vs_ar']:.3f} | "
            f"{r['matched']}/{r['requests']} | {r['peak_allocated_gib']:.3f} |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path.home()/'LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/dllm_vllm/dlite-vlm'
    parser.add_argument('--prefill-hidden-mode',choices=['all','selected'],default='all')
    parser.add_argument('--sample-memory',action='store_true')
    parser.add_argument('--context-window-size',type=int,help='inference override; omit to use trained window')
    parser.add_argument('--checkpoint',nargs=2,action='append',default=[],metavar=('LABEL','PATH'))
    parser.add_argument('--selection-file',type=Path,help='frozen select_data JSON snapshot')
    parser.add_argument('--workspace', type=Path, default=workspace)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'LMM/liyt/datasets/eval_bench')
    parser.add_argument('--target-model', type=Path, default=Path.home()/'LMM/lmm_model/Qwen3.5-4B')
    parser.add_argument('--original-dlite', type=Path, default=Path.home()/'LMM/lmm_model/Qwen3.5-4B-DLite')
    parser.add_argument('--dlite-vlm', type=Path)
    parser.add_argument('--dflash-vlm', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=['dlite_original','dlite_vlm_100k','dflash_vlm_100k'],
                        default=['dlite_original','dlite_vlm_100k','dflash_vlm_100k'])
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--max-pixels-per-frame', type=int, default=448*448)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1,16])
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    if args.context_window_size is not None and args.context_window_size < 1:
        parser.error('positive context window required')
    if args.checkpoint:
        labels=[label for label,_ in args.checkpoint]
        if len(labels)!=len(set(labels)) or any(label=='ar' or not label or any(not(c.isalnum() or c in '-_') for c in label) for label in labels):
            parser.error('unique safe checkpoint labels, excluding ar, required')
        args.models=labels
    if min(args.samples,args.frames,args.max_pixels_per_frame,args.max_new_tokens,*args.batch_sizes) < 1 or args.warmup < 1:
        parser.error('positive budgets and at least one warmup required')
    if args.frames % 2 or args.frames < 2: parser.error('even frame count >= 2 required')
    for values in (args.datasets,args.models,args.batch_sizes):
        if len(set(values)) != len(values): parser.error('duplicate selections')
    root = args.output_dir.expanduser().resolve()
    old = args.workspace.expanduser()/'runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049'
    paths = dict(dlite_original=args.original_dlite,
                 dlite_vlm_100k=args.dlite_vlm or old/'dlite/last',
                 dflash_vlm_100k=args.dflash_vlm or old/'dflash/last')
    if args.checkpoint: paths={label:Path(path) for label,path in args.checkpoint}
    paths = {k: v.expanduser().resolve() for k,v in paths.items() if k in args.models}
    selected = (json.loads(args.selection_file.read_text()) if args.selection_file else
                select_data(args.data_root,args.datasets,args.samples,args.seed))
    if set(selected)!=set(args.datasets) or any(len(rows)!=args.samples for rows in selected.values()):
        parser.error('selection snapshot dataset/sample counts differ')
    for rows in selected.values():
        for row in rows:
            video=Path(row['video']);stat=video.stat()
            if stat.st_size!=row['video_bytes'] or stat.st_mtime_ns!=row['video_mtime_ns']:
                raise ValueError(f'video changed after selection: {video}')
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
                if k not in {'resume','inspect_only','output_dir'}}
    settings.update(state_mode='per_token', attention='sdpa', greedy=True,
                    benchmark_mode='static_replicated_batch', selected=selected,
                    checkpoint_paths={k:str(v) for k,v in paths.items()})
    if not args.inspect_only:
        fingerprints = {}
        for key, folder in {'target':args.target_model.expanduser().resolve(),**paths}.items():
            cfg = folder/'config.json'
            if not cfg.is_file(): raise FileNotFoundError(cfg)
            weights = sorted([*folder.glob('*.safetensors'),*folder.glob('*.bin')])
            if not weights: raise FileNotFoundError(f'no weights in {folder}')
            fingerprints[key] = dict(config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
                weights=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in weights])
        settings['model_fingerprints'] = fingerprints
        import torch
        import transformers
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required for performance evaluation')
        settings.update(torch_version=torch.__version__,transformers_version=transformers.__version__,
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        gpu=str(torch.cuda.get_device_properties(0)))
    settings['code_sha256'] = {p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                              for p in ['benchmark_video.py','video_runtime.py','video_benchmark_data.py']}
    if root.exists():
        if not args.resume: raise FileExistsError(f'{root}: use a fresh directory or --resume')
        if json.loads((root/'settings.json').read_text()) != json.loads(json.dumps(settings,default=str)):
            raise ValueError('resume settings, files, code or sample selection changed')
    else:
        root.mkdir(parents=True)
        write_json(root/'settings.json',settings)
    write_json(root/'samples.json',selected)
    if args.inspect_only:
        print(f'Inspection: {root / "samples.json"}'); return
    (root/'cells').mkdir(exist_ok=True)
    (root/'requests').mkdir(exist_ok=True)
    from litespec.vlm.runtime import load_target
    from litespec.vlm.baselines import load_draft
    from litespec.vlm.benchmark import RequestPositionState
    from litespec.vlm.static_batch import generate_static_batch
    from litespec.vlm.gdn_state import warmup_state_kernels
    from video_runtime import prepare_video
    target,processor=load_target(str(args.target_model.expanduser()),device='cuda',dtype=torch.bfloat16,attention='sdpa')
    for dataset, records in selected.items():
        wanted = [root/'cells'/f'{dataset}_{label}_b{b}.json'
                  for label in ['ar',*args.models] for b in args.batch_sizes]
        if all(p.is_file() for p in wanted):
            print(f'Skip completed dataset: {dataset}',flush=True); continue
        examples, metadata = [], []
        for i,record in enumerate(records):
            print(f'Preparing {dataset} {i+1}/{len(records)}: {record["id"]}',flush=True)
            example,meta=prepare_video(processor,target.config,record,args.frames,args.max_pixels_per_frame)
            examples.append(example); metadata.append(meta)
        text_config=getattr(target.config,'text_config',target.config)
        context_limit=getattr(text_config,'max_position_embeddings',None)
        if context_limit is not None:
            for example in examples:
                if example.prefill_length+args.max_new_tokens>context_limit:
                    raise ValueError(f'{example.sample_id}: input + output exceeds context {context_limit}; not truncating')
        metadata_path=root/f'{dataset}_inputs.json'
        if args.resume and metadata_path.exists():
            prior=json.loads(metadata_path.read_text())
            stable=lambda rows:[{k:v for k,v in r.items() if k!='preprocess_seconds'} for r in rows]
            if stable(prior)!=stable(metadata): raise ValueError('processed video metadata changed on resume')
        else: write_json(metadata_path,metadata)
        for label in ['ar',*args.models]:
            if all((root/'cells'/f'{dataset}_{label}_b{b}.json').is_file() for b in args.batch_sizes): continue
            draft=None if label=='ar' else load_draft(str(paths[label]),attn_implementation='sdpa').to('cuda',torch.bfloat16).eval()
            if draft is not None and args.context_window_size is not None:
                from litespec.vlm.sw_token import set_window
                set_window(draft,args.context_window_size)
            for batch in args.batch_sizes:
                cell=root/'cells'/f'{dataset}_{label}_b{batch}.json'
                if cell.exists(): print(f'Skip completed: {cell.name}',flush=True); continue
                references=None
                if label!='ar':
                    references=json.loads((root/'cells'/f'{dataset}_ar_b{batch}.json').read_text())['results']
                def run(example, cap):
                    gpu=replace(example, inputs={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in example.inputs.items()},
                                vision_mask=example.vision_mask.to('cuda'))
                    with RequestPositionState(target).activate(),torch.inference_mode():
                        return generate_static_batch(target,draft,gpu,batch_size=batch,max_new_tokens=cap,state_mode='per_token',prefill_hidden_mode=args.prefill_hidden_mode)
                if draft is not None:
                    warmup_state_kernels(target,batch_size=batch,width=draft.block_size,mode='per_token')
                print(f'Warmup {dataset} {label} B={batch}',flush=True)
                for _ in range(args.warmup): run(examples[0],min(args.max_new_tokens,16))
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                results=[]
                request_path=root/'requests'/f'{dataset}_{label}_b{batch}.jsonl'
                with request_path.open('w') as f:
                    for i,example in enumerate(examples):
                        print(f'Running {dataset} {label} B={batch} {i+1}/{len(examples)}',flush=True)
                        if args.sample_memory:
                            from sample_memory import measured_request
                            result=measured_request(run,example,args.max_new_tokens,path=root/'sample_memory.jsonl',
                                label=label,index=i+1,count=len(examples),batch_size=batch)
                        else:
                            result=run(example,args.max_new_tokens)
                        match=True if references is None else result['token_ids']==references[i]['token_ids']
                        results.append(result)
                        f.write(json.dumps(dict(id=example.sample_id,token_match=match,result=result,
                            text=processor.tokenizer.decode(result['token_ids'],skip_special_tokens=True)))+'\n'); f.flush()
                tokens=sum(r['output_tokens'] for r in results)
                prefill=sum(r['prefill_seconds'] for r in results)
                decode=sum(r['decode_seconds'] for r in results)
                total=sum(r['total_seconds'] for r in results)
                lengths=[n for r in results for n in r['accept_lengths']]
                opportunities=sum(r.get('first_candidate_opportunities',0) for r in results)
                matched=batch*sum(True if references is None else r['token_ids']==ref['token_ids']
                                  for r,ref in zip(results,references or results))
                summary=dict(dataset=dataset,model=label,batch_size=batch,unique_videos=len(results),
                    requests=len(results)*batch,output_tokens=tokens,prefill_seconds=prefill,decode_seconds=decode,
                    total_seconds=total,decode_tok_s=tokens/decode,end_to_end_tok_s=tokens/total,
                    mean_committed_length=sum(lengths)/len(lengths) if lengths else None,
                    first_candidate_hit_rate=(sum(r.get('first_candidate_hits',0) for r in results)/opportunities if opportunities else None),
                    mean_accepted_draft_tokens=sum(n-1 for n in lengths)/len(lengths) if lengths else None,
                    speculative_rounds=len(lengths),matched=matched,valid_token_match=matched==len(results)*batch,
                    peak_allocated_gib=(max(r['sample_memory']['peak_allocated_gib'] for r in results) if args.sample_memory else torch.cuda.max_memory_allocated()/1024**3),
                    prefill_hidden_mode=args.prefill_hidden_mode,
                    mean_input_tokens=sum(m['input_tokens'] for m in metadata)/len(metadata),
                    mean_video_tokens=sum(m['video_tokens'] for m in metadata)/len(metadata),
                    context_window_size=getattr(draft,'context_window_size',None),
                    context_slots=getattr(draft,'condition_slot_count',None),
                    state_mode='per_token',benchmark_mode='static_replicated_batch',
                    gdn_verify_backend=results[0]['gdn_verify_backend'])
                base=summary if label=='ar' else json.loads((root/'cells'/f'{dataset}_ar_b{batch}.json').read_text())['summary']
                summary.update(raw_decode_ratio_vs_ar=summary['decode_tok_s']/base['decode_tok_s'],
                    raw_end_to_end_ratio_vs_ar=summary['end_to_end_tok_s']/base['end_to_end_tok_s'])
                write_json(cell,dict(summary=summary,results=results)); report(root)
                print(json.dumps(summary),flush=True)
            del draft; gc.collect(); torch.cuda.empty_cache()
        del examples; gc.collect()
    report(root)
    print(f'Results: {root / "comparison.md"}',flush=True)


if __name__=='__main__': main()

