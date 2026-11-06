"""D0/D1/D2 fixed-prefix feedback on/off/cross-sample hidden-shuffle diagnosis."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from litespec.vlm.baselines import load_draft
from litespec.vlm.runtime import load_target,prepare_example
from litespec.vlm.sw_token import configure_image_tokens
from run_vlm_matrix import select_data
from pivot_feedback_probe import collect_feedback,evaluate_feedback,paired_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-model',required=True)
    p.add_argument('--checkpoint',nargs=2,action='append',required=True,metavar=('LABEL','PATH'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--tests',nargs='+',default=['charxiv_val','textvqa','text-only'])
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--probes-per-sample',type=int,default=8)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if os.environ.get('PIVOT_AUGMENTATION_OFF')=='1': p.error('unset PIVOT_AUGMENTATION_OFF for paired probes')
    if args.samples<2 or min(args.max_new_tokens,args.probes_per_sample,args.stride,args.image_max_tokens)<1:
        p.error('at least two samples and positive budgets required')
    if len({name for name,_ in args.checkpoint})!=len(args.checkpoint): p.error('duplicate checkpoint labels')
    root=args.output_dir.expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    dtype=torch.float32 if args.device=='cpu' else torch.bfloat16
    target,processor=load_target(args.target_model,device=args.device,dtype=dtype,attention='sdpa')
    target.eval().requires_grad_(False)
    budget=configure_image_tokens(processor,target.config,args.image_max_tokens)
    settings=dict(arguments=vars(args),scope='fixed_AR_prefix_no_training_no_timing',
        effective_image_processor_size=budget,checkpoint_identities=[],selection={})
    packets=select_data(SimpleNamespace(tests=args.tests,samples=args.samples,seed=args.seed))
    summaries=[]
    ar_references={}
    with (root/'probes.jsonl').open('w') as sink:
        for label,path in args.checkpoint:
            path=Path(path).expanduser().resolve(strict=True)
            draft=load_draft(str(path),attn_implementation='sdpa').to(args.device,dtype).eval().requires_grad_(False)
            aug=draft.pivot_augmentation
            if aug is None or aug.mode not in ('D0','D1','D2'): raise ValueError('D0/D1/D2 checkpoints required')
            settings['checkpoint_identities'].append(dict(label=label,path=str(path),
                config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
                weights=[dict(name=x.name,bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns)
                    for x in sorted(path.glob('*.safetensors'))]))
            for dataset,packet in packets.items():
                records=[json.loads(x) for x in packet['serialized'].splitlines()]
                (root/f'{dataset}_inputs.jsonl').write_text(packet['serialized'])
                settings['selection'][dataset]={k:v for k,v in packet.items() if k!='serialized'}
                pool=[]
                for record in records:
                    example=prepare_example(processor,target.config,record,device=args.device,base_dir=Path(packet['source']).parent,training=False)
                    tokens,probes=collect_feedback(target,draft,example,cap=args.max_new_tokens,
                        stride=args.stride,max_probes=args.probes_per_sample)
                    key=(dataset,example.sample_id)
                    if key in ar_references and ar_references[key]!=tokens:
                        raise ValueError('Target AR trajectory differs between checkpoints; refusing unpaired report')
                    ar_references[key]=tokens
                    for probe in probes: pool.append(probe)
                    with (root/'ar_sequences.jsonl').open('a') as f:
                        f.write(json.dumps(dict(model=label,dataset=dataset,sample_id=example.sample_id,tokens=tokens))+'\n')
                    print(f'{label}/{dataset}/{example.sample_id}: {len(probes)} eligible feedback probes',flush=True)
                groups=defaultdict(list)
                donors=defaultdict(list)
                for probe in pool: donors[probe['sample_id']].append(probe)
                sample_ids=list(donors)
                for index,probe in enumerate(pool):
                    if len(sample_ids)<2: continue
                    next_id=sample_ids[(sample_ids.index(probe['sample_id'])+1)%len(sample_ids)]
                    choices=donors[next_id];donor=choices[index%len(choices)]
                    outputs={c:evaluate_feedback(draft,target,probe,c,donor) for c in ('on','off','shuffled')}
                    for row in paired_rows(outputs):
                        row.update(model=label,dataset=dataset,sample_id=probe['sample_id'],anchor=probe['anchor'],
                            verifier_bonus_disagrees=probe['verifier_bonus_disagrees'],donor=donor['sample_id'])
                        sink.write(json.dumps(row)+'\n');groups[row['condition']].append(row)
                for condition,rows in groups.items():
                    fields=('first_hit','accepted','ce','first_repair','first_break','candidate_change',
                        'mean_abs_logit_delta','applied_residual_ratio','applied_element_changed','gate_mean','verifier_bonus_disagrees')
                    summaries.append(dict(model=label,dataset=dataset,condition=condition,probes=len(rows),
                        **{k:sum(r[k] for r in rows)/len(rows) for k in fields}))
                if not groups: raise ValueError(f'{label}/{dataset}: no cross-sample feedback probes; increase cap/samples')
            del draft
    (root/'settings.json').write_text(json.dumps(settings,indent=2,default=str))
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    lines=['# Paired verifier-feedback diagnosis','','Untimed; no training. All interventions share the same AR prefix, pivot, anchor and reference.',
        'Boundary follows the fixed Target trajectory. Shuffled swaps hidden only across samples; boundary embeddings remain correct.',
        'First hit excludes anchor; accept includes anchor. Averages weight probes equally, not tokens.','',
        '| Dataset | Model | Condition | N | Accept | First hit | Repair | Break | Candidate change | Residual ratio | Changed elements | Gate |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['condition']} | {r['probes']} | {r['accepted']:.3f} | {r['first_hit']:.1%} | {r['first_repair']:.1%} | {r['first_break']:.1%} | {r['candidate_change']:.1%} | {r['applied_residual_ratio']:.3e} | {r['applied_element_changed']:.1%} | {r['gate_mean']:.4f} |")
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')
    print(root/'diagnostics.md')

if __name__=='__main__': main()
