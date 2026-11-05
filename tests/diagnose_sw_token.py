"""TextVQA + early 20k DLite: fixed-prefix K=1/K=2 alignment and attention probes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from litespec.vlm.baselines import load_draft
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.runtime import load_target, prepare_example
from litespec.vlm.sw_token import configure_image_tokens
from run_vlm_matrix import select_data
from sw_token_probe import CONDITIONS, collect_ar_prefixes, run_condition, compare_conditions


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')


def checkpoint_identity(path):
    return dict(path=str(path), config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
        weights=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                 for p in sorted(path.iterdir()) if p.suffix in ('.safetensors','.bin')])


def summarize(rows, attention, *, expected_probes, tolerances):
    by_condition=defaultdict(list)
    for row in rows:
        by_condition[row['condition']].append(row)
    alignment={}
    for name in ('k2_pivot_only', 'k2_pivot_only_origin0'):
        subset=by_condition[name]
        alignment[name]=dict(probes=len(subset),
            hidden_pass=all(r['hidden_close'] for r in subset),
            logits_pass=all(r['logits_close'] for r in subset),
            first_proposal_pass=all(r['first_proposal_equal'] for r in subset),
            max_hidden_delta=max((r['hidden_max_abs'] for r in subset),default=0.),
            max_logit_delta=max((r['logits_max_abs'] for r in subset),default=0.))
    # Both controls must restore the original prediction. Never silently train
    # past an unresolved alignment failure or an empty probe set.
    passed=expected_probes>0 and all(r['probes']==expected_probes and r['hidden_pass'] and r['logits_pass'] and r['first_proposal_pass'] for r in alignment.values())
    conditions=[]
    for name in CONDITIONS:
        subset=by_condition[name]
        if not subset: continue
        n=len(subset)
        conditions.append(dict(condition=name,probes=n,
            first_hit=sum(r['first_hit'] for r in subset)/n,
            accepted=sum(r['accepted'] for r in subset)/n,
            ce=sum(r['ce']*r['candidate_count'] for r in subset)/sum(r['candidate_count'] for r in subset),
            first_repair=sum(r['first_repair'] for r in subset)/n,
            first_break=sum(r['first_break'] for r in subset)/n,
            proposal_agreement=sum(r['proposal_agreement'] for r in subset)/n,
            max_hidden_delta=max(r['hidden_max_abs'] for r in subset),
            max_logit_delta=max(r['logits_max_abs'] for r in subset)))
    groups=defaultdict(list)
    for row in attention:
        groups[row['condition'],row['layer'],row['query']].append(row)
    masses=[]
    for (condition,layer,query), group in sorted(groups.items()):
        masses.append(dict(condition=condition,layer=layer,query=query,probes=len(group),
            **{key:sum(r[key] for r in group)/len(group) for key in
               ('old_mass','pivot_mass','block_mass','old_mass_per_slot','pivot_mass_per_slot','mass_sum')}))
    return dict(schema='sw_textvqa_probe_v1',alignment_passed=passed,
        tolerances=tolerances,expected_probes=expected_probes,alignment=alignment,
        conditions=conditions,attention=masses)


def write_report(root, summary):
    lines=['# TextVQA / 20k DLite fixed-prefix window diagnosis','',
        'No parameter updates; no timed speedup. Each condition shares the same Target AR prefix.',
        f"Target dtype: {summary.get('target_dtype', 'unknown')}; Draft compute dtype: {summary.get('draft_compute_dtype', 'unknown')}; prior AR prefixes verified: {summary.get('prefixes_verified_against_previous', False)}.",
        'First hit excludes the guaranteed anchor; accepted length includes it and is capped by remaining AR tokens.',
        'Attention uses actual post-norm/RoPE Q/K, FP32 softmax, all context and block keys in denominator.',
        'Rows average heads, then probes; attention mass alone does not establish causal usefulness.','',
        f"Alignment gate: **{'PASS' if summary['alignment_passed'] else 'FAIL — do not train yet'}**",'',
        '| Control vs K=1 | N | Hidden close | Logits close | First proposal equal | Max hidden delta | Max logit delta |',
        '|---|---:|---|---|---|---:|---:|']
    for name,r in summary['alignment'].items():
        lines.append(f"| {name} | {r['probes']} | {r['hidden_pass']} | {r['logits_pass']} | {r['first_proposal_pass']} | {r['max_hidden_delta']:.3e} | {r['max_logit_delta']:.3e} |")
    if summary.get('previous_alignment'):
        lines += ['', '## Previous run vs current precision check', '',
            '| Control | Previous max hidden delta | Current max hidden delta | Previous max logit delta | Current max logit delta |',
            '|---|---:|---:|---:|---:|']
        for name,r in summary['alignment'].items():
            old=summary['previous_alignment'][name]
            lines.append(f"| {name} | {old['max_hidden_delta']:.3e} | {r['max_hidden_delta']:.3e} | {old['max_logit_delta']:.3e} | {r['max_logit_delta']:.3e} |")
    lines += ['', '| Condition | N | First hit | Fixed-prefix accept | CE | First repair | First break | Proposal agreement vs K=1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['conditions']:
        lines.append(f"| {r['condition']} | {r['probes']} | {r['first_hit']:.2%} | {r['accepted']:.3f} | {r['ce']:.4f} | {r['first_repair']:.2%} | {r['first_break']:.2%} | {r['proposal_agreement']:.2%} |")
    lines += ['', '## Attention mass by Draft layer and raw query index','',
        '| Condition | Layer | Query | Old | Pivot | Block | Old / valid slot | Pivot / valid slot |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['attention']:
        lines.append(f"| {r['condition']} | {r['layer']} | {r['query']} | {r['old_mass']:.2%} | {r['pivot_mass']:.2%} | {r['block_mass']:.2%} | {r['old_mass_per_slot']:.5f} | {r['pivot_mass_per_slot']:.5f} |")
    lines += ['', 'See probes.jsonl for anchor positions, individual deltas, repair/break and first/later scopes.',
        'Old/pivot are Target-token positions, not necessarily visual tokens. No new image KV bank is added.',
        'Native K=2 masked control retains its absolute RoPE shift. origin0 shifts ALL rotary coordinates by -1.',
        'All logits deltas and CE use the same teacher-forced Target previous tokens; proposal agreement uses greedy RNN proposals.']
    (root/'diagnostics.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--target-model',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--probes-per-sample',type=int,default=16)
    p.add_argument('--image-max-tokens',type=int,default=1024)
    p.add_argument('--device',default='cuda')
    p.add_argument('--draft-dtype',choices=['target','float32'],default='target',
        help='float32 promotes BF16-rounded Draft weights only; Target dtype stays unchanged')
    p.add_argument('--reuse-prefixes-from',type=Path,
        help='reuse prior selection/protocol and require identical Target AR token sequences')
    p.add_argument('--alignment-atol',type=float,default=0.05)
    p.add_argument('--alignment-rtol',type=float,default=0.01)
    args=p.parse_args(argv)
    if min(args.samples,args.max_new_tokens,args.stride,args.probes_per_sample,args.image_max_tokens)<1 or min(args.alignment_atol,args.alignment_rtol)<0:
        p.error('positive budgets and nonnegative tolerances required')
    root=args.output_dir.expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    path=(args.checkpoint or args.workspace/'runs/vlm_three_3ep_20261021_045040_2983971/dlite/last').expanduser().resolve(strict=True)
    cfg=json.loads((path/'config.json').read_text())
    if cfg.get('architectures')!=['DLiteDraftModel'] or cfg.get('dlite_config',{}).get('model_role')!='student':
        p.error('original DLite student required')
    prior_settings = prior_sequences = None
    if args.reuse_prefixes_from:
        previous=args.reuse_prefixes_from.expanduser().resolve(strict=True)
        prior_settings=json.loads((previous/'settings.json').read_text())
        prior_args=prior_settings['arguments']
        for key in ('samples','seed','max_new_tokens','stride','probes_per_sample','image_max_tokens'):
            if prior_args[key] != getattr(args,key):
                raise ValueError(f'prefix reuse differs in {key}')
        if Path(prior_args['target_model']).expanduser().resolve()!=Path(args.target_model).expanduser().resolve():
            raise ValueError('prefix reuse requires identical Target model path')
        serialized=(previous/'inputs.jsonl').read_text()
        if hashlib.sha256(serialized.encode()).hexdigest()!=prior_settings['selection_sha256']:
            raise ValueError('prior diagnostic selection changed')
        prior_rows=[json.loads(line) for line in (previous/'ar_sequences.jsonl').read_text().splitlines() if line.strip()]
        ids=[json.loads(line)['id'] for line in serialized.splitlines() if line.strip()]
        if [row['sample_id'] for row in prior_rows]!=ids or len(ids)!=args.samples:
            raise ValueError('prior AR sequences must cover the full selection in order')
        prior_sequences={row['sample_id']:row for row in prior_rows}
        packet=dict(source=prior_settings['data_source'],serialized=serialized)
    else:
        packet=select_data(SimpleNamespace(tests=['textvqa'],samples=args.samples,seed=args.seed))['textvqa']
    manifest=root/'inputs.jsonl';manifest.write_text(packet['serialized'])
    identity=checkpoint_identity(path)
    if not identity['weights']: p.error('missing checkpoint weights')
    if prior_settings and identity!=prior_settings['checkpoint']:
        raise ValueError('prefix reuse requires the same unchanged Draft checkpoint')
    settings=dict(arguments=vars(args),checkpoint=identity,
        data_source=packet['source'],selection_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        conditions=CONDITIONS,scope='fixed_target_ar_prefix_no_training_no_timing')
    write_json(root/'settings.json',settings)
    dtype=torch.float32 if torch.device(args.device).type=='cpu' else torch.bfloat16
    if prior_settings:
        old_dtype=prior_settings.get('target_dtype') or ('torch.float32' if torch.device(prior_settings['arguments']['device']).type=='cpu' else 'torch.bfloat16')
        if str(dtype)!=old_dtype:
            raise ValueError('prefix reuse must keep Target precision unchanged')
    target,processor=load_target(args.target_model,device=args.device,dtype=dtype,attention='sdpa')
    settings['effective_image_processor_size']=configure_image_tokens(processor,target.config,args.image_max_tokens)
    draft=load_draft(str(path),attn_implementation='sdpa').to(args.device,dtype).eval().requires_grad_(False)
    # Round weights exactly as in the original run, then change compute precision.
    # Do not introduce higher-precision checkpoint values as another variable.
    if args.draft_dtype=='float32':
        draft=draft.to(dtype=torch.float32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
    settings.update(target_dtype=str(target.dtype),draft_weight_rounding_dtype=str(dtype),
        draft_compute_dtype=str(draft.dtype),float32_matmul_precision=torch.get_float32_matmul_precision(),
        prefix_reference=str(args.reuse_prefixes_from) if args.reuse_prefixes_from else None)
    if prior_settings and settings['effective_image_processor_size']!=prior_settings['effective_image_processor_size']:
        raise ValueError('prefix reuse requires the same effective image processor size')
    write_json(root/'settings.json',settings)
    print(f"Target={target.dtype}; Draft={draft.dtype}; weight rounding={dtype}; reuse prefixes={prior_sequences is not None}",flush=True)
    rows,attentions,sequence_rows=[],[],[]
    count=0
    for index,raw in enumerate(manifest.read_text().splitlines()):
        record=json.loads(raw)
        example=prepare_example(processor,target.config,record,device=args.device,training=False,base_dir=manifest.parent)
        print(f"TextVQA {index+1}/{args.samples}: {example.sample_id}",flush=True)
        with RequestPositionState(target).activate():
            tokens,probes=collect_ar_prefixes(target,draft,example,cap=args.max_new_tokens,
                stride=args.stride,max_probes=args.probes_per_sample)
        if prior_sequences is not None:
            reference=prior_sequences[example.sample_id]
            if tokens!=reference['tokens'] or example.prefill_length!=reference['prefill_tokens'] or len(probes)!=reference['probes']:
                write_json(root/'prefix_mismatch.json',dict(sample_id=example.sample_id,
                    previous=reference,current=dict(tokens=tokens,prefill_tokens=example.prefill_length,probes=len(probes))))
                raise ValueError('Target AR prefix changed; inspect prefix_mismatch.json; no valid precision comparison')
        sequence_rows.append(dict(sample_id=example.sample_id,tokens=tokens,probes=len(probes),prefill_tokens=example.prefill_length))
        for probe in probes:
            count+=1
            base=None
            for condition in CONDITIONS:
                current=run_condition(draft,target.get_input_embeddings(),probe,condition)
                if base is None: base=current
                delta=compare_conditions(base,current,atol=args.alignment_atol,rtol=args.alignment_rtol)
                meta=dict(sample_id=example.sample_id,offset=probe['offset'],anchor=probe['anchor'],
                    scope='first_round' if probe['offset']==0 else 'later_ar_prefix',condition=condition)
                rows.append(dict(**meta,**delta,first_hit=current['first_hit'],accepted=current['accepted'],
                    ce=current['ce'],candidate_count=current['candidate_count'],
                    first_repair=not base['first_hit'] and current['first_hit'],
                    first_break=base['first_hit'] and not current['first_hit'],
                    proposals=current['proposals'][0].tolist(),reference=probe['reference']))
                attentions.extend(dict(**meta,**a) for a in current['attention'])
            del base,current
        for name,values in [('probes.jsonl',rows),('attention.jsonl',attentions),('ar_sequences.jsonl',sequence_rows)]:
            (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    summary=summarize(rows,attentions,expected_probes=count,
        tolerances=dict(atol=args.alignment_atol,rtol=args.alignment_rtol))
    summary['checkpoint']=identity
    summary['selection_sha256']=settings['selection_sha256']
    summary.update(target_dtype=settings['target_dtype'],draft_compute_dtype=settings['draft_compute_dtype'],
        draft_weight_rounding_dtype=settings['draft_weight_rounding_dtype'],
        prefixes_verified_against_previous=prior_sequences is not None)
    if prior_sequences is not None:
        summary['previous_alignment']=json.loads((previous/'summary.json').read_text())['alignment']
    write_json(root/'summary.json',summary)
    write_report(root,summary)
    fields=list(summary['attention'][0]) if summary['attention'] else []
    with (root/'attention.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(summary['attention'])
    print(f"Report: {root}/diagnostics.md; alignment_passed={summary['alignment_passed']}",flush=True)
    if not summary['alignment_passed']:
        raise SystemExit(2)


if __name__=='__main__': main()

