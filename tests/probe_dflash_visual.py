"""Correct/wrong/masked Draft visual slots at identical AR prefixes; untimed."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm.baselines import load_draft, DraftCache
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.runtime import load_target,prepare_example
from litespec.vlm.sw_token import configure_image_tokens
from litespec.vlm.dflash_context import propose_features
from run_vlm_matrix import select_data


def select_packets(args):
    from video_benchmark_data import DATASETS,select_data as select_videos
    videos=[name for name in args.tests if name in DATASETS]
    images=[name for name in args.tests if name not in DATASETS]
    packets=select_data(SimpleNamespace(tests=images,samples=args.samples,seed=args.seed)) if images else {}
    if videos:
        selected=select_videos(args.video_data_root,videos,args.samples,args.seed)
        for dataset,records in selected.items():
            serialized=''.join(json.dumps(record,ensure_ascii=False)+'\n' for record in records)
            packets[dataset]=dict(source=str(args.video_data_root/dataset),samples=len(records),
                sha256=hashlib.sha256(serialized.encode()).hexdigest(),serialized=serialized)
    return packets


def prepare_probe_input(processor,config,record,args,source):
    if 'video' in record:
        from video_runtime import prepare_video
        from litespec.vlm.context import visual_positions
        example,metadata=prepare_video(processor,config,record,args.frames,args.max_pixels_per_frame)
        # Match the image probe's intervention scope: include visual delimiters.
        example=replace(example,vision_mask=visual_positions(example.inputs['input_ids'][0],config))
        metadata.update(modality='video',intervention_slots=int(example.vision_mask.sum()))
        pixels=example.inputs['pixel_values_videos']
    else:
        example=prepare_example(processor,config,record,device=args.device,base_dir=Path(source).parent,training=False)
        metadata=dict(modality='image',sample_id=example.sample_id,prefill_tokens=example.prefill_length,
                      intervention_slots=int(example.vision_mask.sum()))
        pixels=example.inputs.get('pixel_values')
    if pixels is None or not example.vision_mask.any():raise ValueError('visual inputs required')
    fingerprint=hashlib.sha256(pixels.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    example=replace(example,inputs={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in example.inputs.items()},
                    vision_mask=example.vision_mask.to(args.device))
    return example,metadata,fingerprint


@torch.no_grad()
def collect(target,draft,example,cap):
    chunks=[]; tokens=[]
    config=getattr(target.config,'text_config',target.config)
    eos=config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    with RequestPositionState(target).activate():
        output=target(**example.inputs,use_cache=True,output_hidden_states=True,logits_to_keep=1)
        chunks.append(DraftCache.features(draft,output).cpu())
        for i in range(cap):
            pending=output.logits[:,-1:].argmax(-1)
            token=int(pending.item());tokens.append(token)
            if token in eos or i+1==cap: break
            output=target(input_ids=pending,past_key_values=output.past_key_values,
                          use_cache=True,output_hidden_states=True,logits_to_keep=1)
            chunks.append(DraftCache.features(draft,output).cpu())
    return dict(sample_id=example.sample_id,prefill=example.prefill_length,
                features=torch.cat(chunks,1),tokens=tokens,vision=example.vision_mask.cpu())


def report(root,rows):
    lines=['# DFlash visual-slot fixed-prefix probes','',
        'Untimed; Target always receives the correct image/video. Same AR prefix, anchor and reference within each checkpoint.',
        'Wrong swaps visual-position Target hidden only; own text/pivot and absolute positions are unchanged.',
        'Visual scope includes image/video placeholders and vision delimiters. Mask is an attention mask, not zero-valued V.',
        'Donors are distinct images/videos within the same dataset; uniform deterministic index alignment keeps recipient slot count.',
        'Repair/break use Mask as the paired reference; candidate change uses Correct. Equal probe weights; no task accuracy claim.','',
        '| Dataset | Model | Condition | N | Accept | First hit | Repair vs mask | Break vs mask | Candidate change |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['condition']} | {r['probes']} | {r['accept']:.3f} | {r['first_hit']:.1%} | {r['repair']:.1%} | {r['break']:.1%} | {r['candidate_change']:.1%} |")
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')
    (root/'summary.json').write_text(json.dumps(rows,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-model',required=True)
    p.add_argument('--checkpoint',nargs=2,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--tests',nargs='+',default=['charxiv_val','textvqa'])
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--probes-per-sample',type=int,default=8)
    p.add_argument('--stride',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--video-data-root',type=Path,default=Path.home()/'LMM/liyt/datasets/eval_bench')
    p.add_argument('--frames',type=int,default=16)
    p.add_argument('--max-pixels-per-frame',type=int,default=448*448)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.samples<2 or min(a.max_new_tokens,a.probes_per_sample,a.stride,a.image_max_tokens)<1: p.error('positive budgets, >=2 images required')
    if a.frames<2 or a.frames%2 or a.max_pixels_per_frame<1:p.error('even frames >=2 and positive per-frame pixel budget required')
    if len(a.tests)!=len(set(a.tests)):p.error('duplicate datasets')
    if len({n for n,_ in a.checkpoint})!=len(a.checkpoint):p.error('duplicate labels')
    root=a.output_dir.expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed)
    dtype=torch.float32 if a.device=='cpu' else torch.bfloat16
    target,processor=load_target(a.target_model,device=a.device,dtype=dtype,attention='sdpa')
    target.eval().requires_grad_(False)
    budget=configure_image_tokens(processor,target.config,a.image_max_tokens)
    packets=select_packets(a)
    settings=dict(arguments=vars(a),image_budget=budget,checkpoints=[],selection={},donors={},inputs={})
    summaries=[];ar_references={}
    with (root/'probes.jsonl').open('w') as sink:
        for label,path in a.checkpoint:
            path=Path(path).expanduser().resolve(strict=True)
            draft=load_draft(str(path),attn_implementation='sdpa').to(a.device,dtype).eval().requires_grad_(False)
            if type(draft).__name__!='DFlashDraftModel' or draft.context_window is not None:
                raise ValueError('visual probes require full-context DFlash')
            settings['checkpoints'].append(dict(label=label,path=str(path),config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
                weights=[dict(name=x.name,bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in sorted(path.glob('*.safetensors'))]))
            for dataset,packet in packets.items():
                settings['selection'][dataset]={k:v for k,v in packet.items() if k!='serialized'}
                (root/f'{dataset}_inputs.jsonl').write_text(packet['serialized'])
                pool=[]; fingerprints=[]
                for record in map(json.loads,packet['serialized'].splitlines()):
                    ex,metadata,fingerprint=prepare_probe_input(processor,target.config,record,a,packet['source'])
                    settings['inputs'][f'{dataset}/{ex.sample_id}']=metadata
                    fingerprints.append(fingerprint)
                    item=collect(target,draft,ex,a.max_new_tokens)
                    key=(dataset,ex.sample_id)
                    if key in ar_references and ar_references[key]!=item['tokens']: raise ValueError('AR reference differs across checkpoints')
                    ar_references[key]=item['tokens']
                    with (root/'ar_sequences.jsonl').open('a') as ar_sink:
                        ar_sink.write(json.dumps(dict(model=label,dataset=dataset,sample_id=ex.sample_id,tokens=item['tokens']))+'\n')
                    pool.append(item)
                    print(f'{label}/{dataset}: prepared {len(pool)}/{a.samples}',flush=True)
                totals=defaultdict(list)
                for i,item in enumerate(pool):
                    donor=next((pool[(i+d)%len(pool)] for d in range(1,len(pool)) if fingerprints[(i+d)%len(pool)]!=fingerprints[i]),None)
                    if donor is None:raise ValueError('no distinct visual-content donor')
                    settings['donors'][f'{dataset}/{item["sample_id"]}']=donor['sample_id']
                    donor_features=donor['features'][:,:donor['prefill']][:,donor['vision']].to(a.device)
                    offsets=list(range(0,len(item['tokens'])-draft.block_size+1,a.stride))[:a.probes_per_sample]
                    for offset in offsets:
                        anchor=item['prefill']+offset
                        features=item['features'][:,:anchor].to(a.device)
                        vision=torch.cat([item['vision'],torch.zeros(offset,dtype=torch.bool)]).to(a.device)
                        pending=torch.tensor([[item['tokens'][offset]]],device=a.device)
                        reference=torch.tensor(item['tokens'][offset+1:offset+draft.block_size],device=a.device)
                        outputs={}
                        for condition in ('correct','wrong','mask'):
                            proposal,meta=propose_features(draft,target,features,pending,vision_mask=vision,condition=condition,donor_features=donor_features)
                            hits=proposal[0]==reference
                            outputs[condition]=dict(proposal=proposal[0],accept=1+int(hits.long().cumprod(0).sum()),first_hit=bool(hits[0]),metadata=meta)
                        base=outputs['correct']; mask_base=outputs['mask']
                        for condition,out in outputs.items():
                            row=dict(dataset=dataset,model=label,sample_id=item['sample_id'],donor_id=donor['sample_id'],offset=offset,anchor=anchor,condition=condition,
                                accept=out['accept'],first_hit=out['first_hit'],repair=out['first_hit'] and not mask_base['first_hit'],break_=mask_base['first_hit'] and not out['first_hit'],
                                candidate_change=float((out['proposal']!=base['proposal']).float().mean()),metadata=out['metadata'],reference=reference.tolist(),proposal=out['proposal'].tolist())
                            sink.write(json.dumps(row)+'\n');totals[condition].append(row)
                for condition,rows in totals.items():
                    summaries.append(dict(dataset=dataset,model=label,condition=condition,probes=len(rows),**{key:sum(r[source] for r in rows)/len(rows) for key,source in [('accept','accept'),('first_hit','first_hit'),('repair','repair'),('break','break_'),('candidate_change','candidate_change')]}))
                if not totals:raise ValueError('no complete-block probes; increase output budget')
                del pool
                report(root,summaries)
            del draft
    (root/'settings.json').write_text(json.dumps(settings,default=str,indent=2))
    print((root/'diagnostics.md').read_text())


if __name__=='__main__':main()
