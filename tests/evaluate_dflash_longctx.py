"""Fixed 16-image length sweep: short/8k/32k/64k, AR/Full/Pivot5."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

from evaluate_dflash_all import execute, fingerprint, write_json
from evaluate_full_visual import source_directory
from prepare_longctx_data import verify_dataset

HERE=Path(__file__).resolve().parent
SEPARATOR='\n\nThe preceding reports are unrelated context. Answer the following question about the image.\n'


def fit_prefix(ids, decode, measure, question, target, tolerance=8):
    """Only shorten filler; never truncate image blocks or the original question."""
    def candidate(n):
        text=decode(ids[:n])+SEPARATOR+question
        return text,measure(text)
    empty,base=candidate(0)
    if base>target: raise ValueError(f'Original task/image already uses {base} tokens > {target}')
    n=min(len(ids),target-base)
    seen=set()
    for _ in range(8):
        if n in seen: break
        seen.add(n)
        text,length=candidate(n)
        if 0<=target-length<=tolerance:return text,length,n
        n=max(0,min(len(ids),n+target-length))
    lo,hi,best=0,len(ids),(empty,base,0)
    while lo<=hi:
        mid=(lo+hi)//2; text,length=candidate(mid)
        if length<=target:
            if length>best[1]:best=(text,length,mid)
            lo=mid+1
        else:hi=mid-1
    if target-best[1]>tolerance: raise ValueError(f'Could only fit {best[1]} of {target} tokens')
    return best


def filler_for(row,reports,tokenizer,needed,seed):
    key=int(hashlib.sha256(f'{seed}:{row["id"]}'.encode()).hexdigest()[:16],16)
    pool=list(reports);random.Random(key).shuffle(pool)
    pieces,used=[],[]
    for i in range(max(100,len(pool)*4)):
        report=pool[i%len(pool)]
        pieces.append(report['text']);used.append(report['id'])
        text='\n\n'.join(pieces)
        ids=tokenizer.encode(text,add_special_tokens=False)
        if len(ids)>=needed:return ids,used,hashlib.sha256(text.encode()).hexdigest()
    raise ValueError('Report pool could not supply enough filler')


def resolve_models(args):
    source=source_directory(args.workspace,args.source_eval_dir)
    prior=json.loads((source/'manifest.json').read_text())
    group=Path(prior['run_group'])
    models={}
    for label in ('full','pivot5'):
        item=fingerprint(group/label/'epoch_5')
        if item!=prior['checkpoints'][label]:raise ValueError(f'{label}: source evaluation must use the epoch-5 export')
        cfg=json.loads((Path(item['path'])/'config.json').read_text())
        if cfg.get('architectures')!=['DFlashDraftModel'] or cfg.get('vlm_config',{}).get('draft_context','full')!=label:
            raise ValueError(f'Wrong checkpoint architecture/context: {label}')
        if label=='pivot5' and (cfg.get('dflash_config',{}).get('context_projection','shared')!='shared' or cfg.get('dflash_config',{}).get('position_layer_bias',False)):
            raise ValueError('Use baseline Pivot5, not B/C')
        models[label]=item
    target=fingerprint(prior['target']['path'])
    if target!=prior['target']:raise ValueError('Target files changed since experiment 1')
    for label,item in [('target',target),*models.items()]:
        config=json.loads((Path(item['path'])/'config.json').read_text())
        text=config.get('text_config',config)
        limit=text.get('max_position_embeddings')
        if limit is not None and max(args.lengths)+args.max_new_tokens>limit:
            raise ValueError(f'{label} configured context limit {limit} is below requested length + output')
    return source,models,target


def processor_encoder(target,image_tokens):
    from transformers import AutoConfig,AutoProcessor
    from litespec.vlm.runtime import prepare_example
    from reference_vlm_image_budget import configure_image_budget
    config=AutoConfig.from_pretrained(target)
    processor=AutoProcessor.from_pretrained(target)
    vision=config.vision_config
    pixels=image_tokens*(vision.patch_size*vision.spatial_merge_size)**2
    size=configure_image_budget(processor,max_pixels=pixels)
    def encode(row):
        example=prepare_example(processor,config,row,device='cpu',training=False)
        return example.prefill_length,int(example.vision_mask.sum())
    return processor.tokenizer,encode,pixels,size


def build_inputs(args,root,rows,reports,tokenizer,encode):
    selected=rows[:args.samples]
    if len(selected)!=args.samples:raise ValueError('Not enough prepared records')
    variants={length:[] for length in args.lengths}
    metadata=[]
    for index,row in enumerate(selected):
        base_length,visual=encode(row)
        ids,report_ids,digest=filler_for(row,reports,tokenizer,max(args.lengths)+1024,args.seed)
        previous_cut=0
        for length in args.lengths:
            if length==0:
                record=dict(row);physical=base_length;cut=0
            else:
                def measure(question):return encode(dict(row,question=question))[0]
                question,physical,cut=fit_prefix(ids,lambda x:tokenizer.decode(x,skip_special_tokens=False,
                    clean_up_tokenization_spaces=False),measure,row['question'],length)
                record=dict(row,question=question)
                measured,measured_visual=encode(record)
                if measured!=physical or measured_visual!=visual:raise ValueError('Processor/image tokenization changed while fitting')
                if cut<previous_cut:raise ValueError('Filler prefixes are not nested')
                previous_cut=cut
            if physical+args.max_new_tokens>args.context_limit:raise ValueError('Processed input exceeds configured context limit')
            variants[length].append(record)
            metadata.append(dict(id=row['id'],length_bin=length,input_tokens=physical,
                visual_tokens_with_delimiters=visual,filler_token_count=cut,filler_source_sha256=digest,report_ids=report_ids))
        print(f'Prepared length variants {index+1}/{args.samples}: {row["id"]}',flush=True)
    paths={}
    (root/'inputs').mkdir()
    for length,records in variants.items():
        name='short' if length==0 else f'ctx{length}'
        path=root/'inputs'/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
        paths[length]=path
    write_json(root/'input_lengths.json',metadata)
    return paths,[row['id'] for row in selected]


def make_jobs(args,root,paths,models,target,pixels,smoke=False):
    jobs=[]
    lengths=[args.lengths[0],args.lengths[-1]] if smoke else args.lengths
    for repeat in range(1 if smoke else args.repeats):
        for length in lengths:
            name='short' if length==0 else f'ctx{length}'
            case=f'{name}_r{repeat+1}'
            gpu=args.gpus[len(jobs)%len(args.gpus)]
            labels=['full','pivot5'] if repeat%2==0 else ['pivot5','full']
            count=min(2,args.samples) if smoke else args.samples
            output=root/'results'/case
            cmd=[sys.executable,str(HERE/'benchmark_vlm_static_batch.py'),
                '--target-model',target['path'],'--data',str(paths[length]),'--dataset-name',name,
                '--output-dir',str(output),'--max-samples',str(count),'--batch-sizes','1',
                '--max-new-tokens',str(args.max_new_tokens),'--warmup','1','--seed',str(args.seed),
                '--image-max-pixels',str(pixels),'--cpu-inputs','--attention-backend','sdpa',
                '--state-mode','per_token','--visual-condition','correct']
            for label in labels:cmd+=['--checkpoint',label,models[label]['path']]
            if not smoke and args.profile_samples:
                cmd+=['--profile-decode','--profile-samples',str(args.profile_samples)]
            jobs.append(dict(dataset=case,kind='image',gpu=gpu,samples=count,models=['ar',*labels],
                output=str(output),log=str(root/'logs'/f'{case}.log'),command=cmd,
                length_bin=length,repeat=repeat+1,snapshot=str(paths[length])))
    return jobs


def report(root,jobs):
    statuses=json.loads((root/'status.json').read_text())
    rows=[]
    for job in jobs:
        if statuses[job['dataset']]['status']!='success':continue
        output=Path(job['output']);summaries=json.loads((output/'summary.json').read_text())
        full=next(s for s in summaries if s['model']=='full')
        for s in summaries:
            first,later,calls=[],0.0,0
            cache_bytes,raw_bytes=[],[]
            path=output/f"{s['model']}_b1_profile.jsonl"
            if path.is_file():
                for line in path.read_text().splitlines():
                    profile=json.loads(line)['result']['profile']
                    if 'draft' in profile.get('first_stage_seconds',{}):
                        first.append(profile['first_stage_seconds']['draft']*1000)
                    later+=profile.get('later_stage_seconds',{}).get('draft',0)*1000
                    calls+=profile.get('later_stage_calls',{}).get('draft',0)
                    cache=profile.get('draft_cache',{})
                    if cache.get('projected_kv_storage_bytes') is not None:cache_bytes.append(cache['projected_kv_storage_bytes'])
                    if cache.get('raw_hidden_storage_bytes') is not None:raw_bytes.append(cache['raw_hidden_storage_bytes'])
            rows.append(dict(length_bin=job['length_bin'],repeat=job['repeat'],gpu=job['gpu'],model=s['model'],
                input_tokens=s['mean_input_tokens'],output_tokens_per_prompt=s['output_tokens']/s['requests'],
                committed=s['mean_committed_length'],first_hit=s.get('first_candidate_hit_rate'),
                tok_s=s['output_tokens_per_second'],ratio_vs_ar=s['raw_throughput_speedup_vs_ar'],
                ratio_vs_full=s['output_tokens_per_second']/full['output_tokens_per_second'],
                prefill_s=s['prefill_seconds'],decode_s=s['decode_seconds'],total_s=s['wall_seconds'],
                first_draft_ms=statistics.mean(first) if first else None,later_draft_ms=later/calls if calls else None,
                final_draft_kv_mib=statistics.mean(cache_bytes)/1024**2 if cache_bytes else None,
                final_raw_hidden_mib=statistics.mean(raw_bytes)/1024**2 if raw_bytes else None,
                total_latency_ratio_vs_full=full['wall_seconds']/s['wall_seconds'],
                peak_gib=s['peak_allocated_gib'],matched=s['matched'],requests=s['requests']))
    write_json(root/'longctx_comparison.json',rows)
    if rows:
        with (root/'longctx_comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Full vs Pivot5 length sweep','',
        'B=1; same prompts at every length, each ratio uses a same-GPU baseline. All samples contribute.',
        'Input tokens include visual placeholders and chat overhead. Profile timings are from a separate pass.',
        'AR mismatches and natural EOS remain visible; no task-accuracy or lossless-speedup claim.','',
        '| Bin | Repeat | GPU | Model | Input tokens | Output/prompt | Committed | tok/s | /AR | /Full | Prefill s | Decode s | Total s | Peak GiB | Match |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        committed='—' if r['committed'] is None else f"{r['committed']:.3f}"
        lines.append(f"| {r['length_bin'] or 'short'} | {r['repeat']} | {r['gpu']} | {r['model']} | {r['input_tokens']:.0f} | "
            f"{r['output_tokens_per_prompt']:.1f} | {committed} | {r['tok_s']:.2f} | {r['ratio_vs_ar']:.3f} | {r['ratio_vs_full']:.3f} | "
            f"{r['prefill_s']:.2f} | {r['decode_s']:.2f} | {r['total_s']:.2f} | {r['peak_gib']:.2f} | {r['matched']}/{r['requests']} |")
    if any(r['model']=='ar' and r['output_tokens_per_prompt']<64 for r in rows):
        lines+=['','AR outputs average fewer than 64 tokens in at least one cell; inspect output lengths before interpreting steady decode gains.']
    for case,status in statuses.items():
        if status['status']!='success':lines+=['',f"{case}: {status['status']}; {status.get('log','')}"]
    path=root/'longctx_comparison.md';path.write_text('\n'.join(lines)+'\n')
    print(path.read_text(),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--source-eval-dir',type=Path)
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--lengths',type=int,nargs='+',default=[0,8192,32768,65536])
    parser.add_argument('--gpus',nargs='+',default=list(map(str,range(8))))
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--image-max-tokens',type=int,default=1024)
    parser.add_argument('--max-new-tokens',type=int,default=256)
    parser.add_argument('--profile-samples',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=len(args.gpus)<=8 or len(set(args.gpus))!=len(args.gpus):parser.error('one to eight distinct GPUs required')
    if min(args.samples,args.repeats,args.image_max_tokens,args.max_new_tokens)<1 or args.profile_samples<0:parser.error('invalid budgets')
    if args.lengths!=sorted(set(args.lengths)) or args.lengths[0]!=0 or len(args.lengths)<2:parser.error('lengths must start at 0 and increase')
    rows,reports,data_manifest=verify_dataset(args.data_dir)
    source,models,target=resolve_models(args)
    limits=[]
    for item in [target,*models.values()]:
        config=json.loads((Path(item['path'])/'config.json').read_text());text=config.get('text_config',config)
        if text.get('max_position_embeddings'):limits.append(text['max_position_embeddings'])
    args.context_limit=min(limits) if limits else max(args.lengths)+args.max_new_tokens
    root=args.output_dir.expanduser().resolve()
    if root.exists():raise FileExistsError(f'Choose a fresh output directory: {root}')
    root.mkdir(parents=True)
    tokenizer,encode,pixels,size=processor_encoder(target['path'],args.image_max_tokens)
    paths,ids=build_inputs(args,root,rows,reports,tokenizer,encode)
    for name in ('logs','results'):(root/name).mkdir()
    jobs=make_jobs(args,root,paths,models,target,pixels)
    write_json(root/'manifest.json',dict(source_evaluation=str(source),checkpoints=models,target=target,
        data_manifest_sha256=hashlib.sha256((args.data_dir/'manifest.json').read_bytes()).hexdigest(),
        settings=vars(args),sample_ids=ids,image_max_pixels=pixels,image_processor_size=size,
        input_sha256={str(k):hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},jobs=jobs))
    print(f'Checkpoints: {models}\nOutput: {root}',flush=True)
    if args.prepare_only:return 0
    smoke=root/'smoke'
    for name in ('logs','results'):(smoke/name).mkdir(parents=True)
    smoke_jobs=make_jobs(args,smoke,paths,models,target,pixels,smoke=True)
    if execute(smoke,smoke_jobs,1):
        print(f'Smoke failed; formal 16-sample evaluation was not started. See {smoke}/logs',flush=True)
        return 1
    result=execute(root,jobs,1);report(root,jobs)
    print(f'Results: {root}/longctx_comparison.md',flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
