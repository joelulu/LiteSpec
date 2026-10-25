"""Read-only GDN checkpoint/gradient/causal-state diagnostics on fixed anchors.

No retraining, optimizer step or speed claims. Native target greedy continuations
are generated once and shared by every mode and checkpoint. Diagnostic gradients
are NEW CE probes, not reconstructed historical training gradients.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from types import MethodType

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from safetensors import safe_open
from litespec.vlm.baselines import is_baseline, load_draft
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.context import ContextPolicy
from litespec.vlm.gdn_reader import cache_reader_states
from litespec.vlm.runtime import configure_image_token_budget, draft_block, load_target, prepare_example, selected_layers
from litespec.vlm.static_batch import generate_static_batch


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def tensor_stats(tensor):
    value=tensor.detach().float()
    finite=bool(torch.isfinite(value).all())
    return dict(shape=list(value.shape),dtype=str(tensor.dtype),finite=finite,
        norm=float(value.norm()) if finite else None,
        max_abs=float(value.abs().max()) if finite and value.numel() else None,
        nonzero=int(torch.count_nonzero(value)),numel=value.numel())


def reader_of(draft):
    cfg=getattr(draft.config,'gdn_reader_config',None)
    if not cfg:
        raise ValueError('checkpoint has no gdn_reader_config')
    return draft.layers[cfg['injection_layer']].gdn_reader


def checkpoint_audit(path,draft):
    path=Path(path)
    files=sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError('diagnostics require the exported safetensors checkpoint')
    expected=draft.state_dict()
    keys=set(); reports={}
    for file in files:
        with safe_open(file,framework='pt',device='cpu') as weights:
            for name in weights.keys():
                if name in keys: raise ValueError('duplicate checkpoint key: '+name)
                keys.add(name)
                if '.gdn_reader.' not in name: continue
                saved=weights.get_tensor(name)
                row=tensor_stats(saved)
                if name in expected:
                    loaded=expected[name].detach().cpu()
                    wanted=saved.to(loaded.dtype)
                    row['exact_after_load_cast']=tuple(wanted.shape)==tuple(loaded.shape) and torch.equal(wanted,loaded)
                    row['max_load_error']=float((wanted.float()-loaded.float()).abs().max()) if wanted.shape==loaded.shape else None
                else:
                    row['exact_after_load_cast']=False
                if name.endswith('.up.weight'):
                    row['zero_initialized_output_projection']=True
                    row['changed_from_zero_initialization']=bool(saved.count_nonzero())
                reports[name]=row
    missing=sorted(set(expected)-keys);unexpected=sorted(keys-set(expected))
    reader_keys=[k for k in expected if '.gdn_reader.' in k]
    passed=bool(reader_keys) and not missing and not unexpected and set(reader_keys)==set(reports) and all(
        r['finite'] and r['exact_after_load_cast'] for r in reports.values())
    result=dict(checkpoint=str(path.resolve()),config=draft.config.gdn_reader_config,
        full_missing_keys=missing,full_unexpected_keys=unexpected,parameters=reports,passed=passed,
        historical_reader_gradient_status='not_recorded_by_original_trainer',
        warning='Nonzero up.weight shows a change from zero initialization; fresh gradient probes do not establish historical per-step gradients.')
    for name in ('checkpoint.json','vlm_run.json'):
        if (path/name).exists():result[name]=json.loads((path/name).read_text())
    log=path.resolve().parent/'train.log'
    global_gradients=[]
    if log.exists():
        for line in log.open(errors='replace'):
            try:row=json.loads(line)
            except (ValueError,TypeError):continue
            if 'optimizer_step' in row and isinstance(row.get('grad_norm'),(float,int)):
                global_gradients.append(dict(step=row['optimizer_step'],global_grad_norm=row['grad_norm']))
    result['historical_global_gradients']=dict(count=len(global_gradients),last=global_gradients[-1] if global_gradients else None,
        scope='all_draft_parameters_not_reader_specific')
    return result


@contextmanager
def reader_mode(reader,mode):
    if mode not in {'normal','off','wrong'}:raise ValueError(mode)
    had_local='forward' in reader.__dict__;local=reader.__dict__.get('forward')
    if mode=='off':reader.forward=MethodType(lambda self,hidden,states:hidden,reader)
    try:yield
    finally:
        if mode=='off':
            if had_local:reader.forward=local
            else:del reader.forward


def extended_inputs(example,continuation,offset):
    values=dict(example.inputs)
    prefix=values['input_ids']
    extra=prefix.new_tensor(continuation[:offset]).reshape(1,-1)
    values['input_ids']=torch.cat((prefix,extra),1)
    for key,fill in (('attention_mask',1),('mm_token_type_ids',0)):
        if key in values:
            values[key]=torch.cat((values[key],values[key].new_full((1,offset),fill)),1)
    if 'position_ids' in values:raise ValueError('explicit position_ids unsupported; native Qwen MRoPE required')
    return values


def block_predictions(target,draft,example,output,states,continuation,offset):
    """Return teacher-forced logits AND actual free-running draft candidates."""
    anchor=example.prefill_length+offset
    pending=example.inputs['input_ids'].new_tensor([[continuation[offset]]])
    if is_baseline(draft):
        query=pending.new_full((1,draft.block_size),draft.mask_token_id);query[:,:1]=pending
        context=selected_layers(output.hidden_states,draft.target_layer_ids,draft.config.num_target_layers).flatten(-2).detach()
        hidden=draft(noise_embedding=target.get_input_embeddings()(query),target_hidden=context,
            position_ids=torch.arange(anchor+draft.block_size,device=pending.device)[None],
            use_cache=False,is_causal=False,gdn_states=states)
        logits=target.get_output_embeddings()(hidden[:,-draft.block_size+1:]).float()
        candidates=draft._propose_draft_tokens(target,hidden,query,temperature=0.)[0]
    else:
        pivot=selected_layers(output.hidden_states,draft.target_layer_ids,draft.config.num_target_layers)[:,-1:].detach()
        hidden,_=draft_block(draft,target.get_input_embeddings(),anchor_ids=pending,pivot=pivot,
            history=None,anchor=anchor,prefill_length=example.prefill_length,
            vision_mask=example.vision_mask,policy=ContextPolicy.no_history(),gdn_states=states)
        previous=pending.new_zeros(1,draft.proposal_length)
        available=continuation[offset:offset+draft.proposal_length]
        previous[:,:len(available)]=pending.new_tensor(available)
        head=draft.sequential_head
        latent=head.forward_teacher_forcing(hidden_states=hidden,prev_token_ids=previous)
        logits=head.project_logits(latent).float()
        candidates=draft.sample_draft_tokens(draft_hidden=hidden,first_prev_token_ids=pending[:,0],temperature=0.)[0]
    width=min(len(continuation)-offset-1,logits.shape[1])
    return logits[:,:width],candidates[:,:width]


def run_mode(target,draft,example,output,states,reference,offset,mode,teacher_logits):
    reader=reader_of(draft)
    residuals=[]
    def observe(module,args,value):
        # Exclude known anchor slots; measure prediction positions only.
        width=draft.block_size-1
        before=args[0][:,-width:].detach().float()
        delta=value[:,-width:].detach().float()-before
        residuals.append(dict(delta_norm=float(delta.norm()),hidden_norm=float(before.norm()),
            relative_norm=float(delta.norm()/before.norm().clamp_min(1e-12))))
    hook=reader.register_forward_hook(observe)
    try:
        with torch.no_grad(),reader_mode(reader,mode):
            logits,candidates=block_predictions(target,draft,example,output,states,reference,offset)
    finally:hook.remove()
    if len(residuals)!=1:raise RuntimeError(f'reader was called {len(residuals)} times; expected one injection')
    p=logits.softmax(-1);teacher=teacher_logits.float().softmax(-1)
    truth=candidates.new_tensor(reference[offset+1:offset+1+candidates.shape[1]])[None]
    row=dict(mode=mode,positions=logits.shape[1],residual=residuals[0],
        ce_target_greedy=float(F.cross_entropy(logits.flatten(0,1),teacher_logits.argmax(-1).flatten())),
        ce_reference_tokens=float(F.cross_entropy(logits.flatten(0,1),truth.flatten())),
        l1_to_target=float((p-teacher).abs().sum(-1).mean()),
        fixed_reference_committed_length=1+int((candidates==truth).cumprod(1).sum()),
        candidate_token_ids=candidates[0].tolist(),teacher_forced_top1=logits.argmax(-1)[0].tolist())
    return row,logits,candidates


def compare(normal_logits,logits,normal_candidates,candidates):
    p=normal_logits.softmax(-1)
    return dict(logits_mean_abs_change=float((normal_logits-logits).abs().mean()),
        logits_max_abs_change=float((normal_logits-logits).abs().max()),
        kl_normal_to_mode=float((p*(normal_logits.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean()),
        teacher_forced_top1_change_fraction=float((normal_logits.argmax(-1)!=logits.argmax(-1)).float().mean()),
        free_running_candidate_change_fraction=float((normal_candidates!=candidates).float().mean()))


def gradient_probe(target,draft,example,output,states,reference,offset,teacher_logits):
    reader=reader_of(draft)
    params=list(reader.named_parameters())
    with torch.enable_grad():
        logits,_=block_predictions(target,draft,example,output,states,reference,offset)
        loss=F.cross_entropy(logits.flatten(0,1),teacher_logits.argmax(-1).flatten())
        gradients=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    return dict(loss=float(loss.detach()),objective='fresh_teacher_greedy_CE_probe_not_historical_training_gradient',
        optimizer_step_performed=False,parameters={name:dict(connected=g is not None,
            requires_grad=p.requires_grad,gradient=tensor_stats(g) if g is not None else None)
            for (name,p),g in zip(params,gradients)})


def donor_index(records,index):
    row=records[index]
    def key(r):return str(Path(r['image']).expanduser().resolve()) if r.get('image') else r['question']
    for step in range(1,len(records)):
        other=(index+step)%len(records)
        if key(records[other])!=key(row):return other
    raise ValueError('wrong-state test requires another image (or another text-only question)')


def summarize(rows):
    summaries=[]
    for model,case,mode in dict.fromkeys((r['model'],r['dataset'],r['mode']) for r in rows):
        group=[r for r in rows if (r['model'],r['dataset'],r['mode'])==(model,case,mode)]
        fields=('ce_target_greedy','ce_reference_tokens','l1_to_target','fixed_reference_committed_length',
                'logits_mean_abs_change','logits_max_abs_change','kl_normal_to_mode',
                'teacher_forced_top1_change_fraction','free_running_candidate_change_fraction')
        summary=dict(model=model,dataset=case,mode=mode,anchors=len(group),
            examples=len({r['sample_id'] for r in group}),positions=sum(r['positions'] for r in group),
            mean_residual_ratio=statistics.mean(r['residual']['relative_norm'] for r in group),
            aggregation='equal_anchor_weight_except_position_weighted_CE_L1_and_flip_rates')
        weighted={'ce_target_greedy','ce_reference_tokens','l1_to_target','logits_mean_abs_change',
                  'kl_normal_to_mode','teacher_forced_top1_change_fraction','free_running_candidate_change_fraction'}
        for field in fields:
            summary[field]=(sum(r[field]*r['positions'] for r in group)/summary['positions'] if field in weighted
                            else statistics.mean(r[field] for r in group))
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-group',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--samples-per-dataset',type=int,default=8)
    parser.add_argument('--reference-tokens',type=int,default=64)
    parser.add_argument('--offsets',nargs='+',type=int,default=[0,8,16,32])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--attention',default='sdpa',choices=['sdpa','eager','flash_attention_2'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(argv)
    if args.samples_per_dataset<2 or args.reference_tokens<2 or not args.offsets or min(args.offsets)<0:
        parser.error('need >=2 different samples, >=2 reference tokens and nonnegative offsets')
    run=args.run_group.expanduser().resolve();root=args.output_dir.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    cfg=json.loads((run/'dlite_gdn/last/vlm_run.json').read_text())
    target_path=cfg['target_model'];image_cap=cfg.get('image_max_tokens',1024)
    target,processor=load_target(target_path,device=args.device,
        dtype=torch.float32 if args.device=='cpu' else torch.bfloat16,attention=args.attention)
    configure_image_token_budget(processor,target.config,image_cap)
    drafts={};audits={};checkpoints={}
    for label in ('dflash_gdn','dlite_gdn'):
        path=run/label/'last'
        draft=load_draft(path,attn_implementation='sdpa',dtype=target.dtype).to(args.device).eval()
        audits[label]=checkpoint_audit(path,draft)
        write_json(root/'checkpoint_audit.json',audits)
        if not audits[label]['passed']:raise ValueError(f'{label}: checkpoint/loading audit failed')
        trained=audits[label].get('vlm_run.json',{})
        if trained.get('target_model')!=target_path or trained.get('image_max_tokens',1024)!=image_cap:
            raise ValueError('target/image budget differs between checkpoints')
        drafts[label]=draft
        checkpoints[label]=str(path.resolve())
    files=sorted((run/'evaluation/inputs').glob('*.jsonl'))
    if not files:raise ValueError('missing saved evaluation inputs under run-group/evaluation/inputs')
    settings=dict(run_group=str(run),checkpoints=checkpoints,target_model=target_path,image_max_tokens=image_cap,
        samples_per_dataset=args.samples_per_dataset,reference_tokens=args.reference_tokens,offsets=args.offsets,
        seed=args.seed,attention=args.attention,device=args.device,
        historical_gradients='reader-specific gradients not recorded; this program only probes current gradients',
        scope='fixed native-AR reference anchors; not speed benchmark or lossless verifier test',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'settings.json',settings)
    rows=[];probes={};skipped=[]
    with (root/'references.jsonl').open('w') as refs,(root/'anchors.jsonl').open('w') as detail:
        for file in files:
            case=file.stem
            records=[json.loads(line) for line in file.read_text().splitlines() if line.strip()][:args.samples_per_dataset]
            if len(records)<2:raise ValueError(f'{case}: need at least two distinct reference samples')
            if len({str(r['id']) for r in records})!=len(records):raise ValueError('duplicate sample IDs in diagnostic inputs')
            for record in records:
                if record.get('image') and not Path(record['image']).is_absolute():
                    record['image']=str((file.parent/record['image']).resolve())
            examples=[];continuations=[]
            for record in records:
                example=prepare_example(processor,target.config,record,device=args.device,base_dir=file.parent)
                with RequestPositionState(target).activate():
                    reference=generate_static_batch(target,None,example,max_new_tokens=args.reference_tokens)['token_ids']
                examples.append(example);continuations.append(reference)
                refs.write(json.dumps(dict(dataset=case,sample_id=record['id'],reference_token_ids=reference))+'\n');refs.flush()
            for index,(example,reference) in enumerate(zip(examples,continuations)):
                other=donor_index(records,index)
                for offset in args.offsets:
                    if offset+1>=len(reference):
                        skipped.append(dict(dataset=case,sample_id=example.sample_id,offset=offset,reason='reference_ended'))
                        continue
                    donor_offset=min(offset,len(continuations[other])-1)
                    if donor_offset<0:raise ValueError('empty donor reference')
                    with torch.no_grad(),RequestPositionState(target).activate():
                        donor=target(**extended_inputs(examples[other],continuations[other],donor_offset),
                                     use_cache=True,output_hidden_states=False,logits_to_keep=1)
                    wrong={label:tuple(s.clone() for s in cache_reader_states(draft,donor.past_key_values)) for label,draft in drafts.items()}
                    del donor
                    with RequestPositionState(target).activate():
                        with torch.no_grad():
                            output=target(**extended_inputs(example,reference,offset),use_cache=True,output_hidden_states=True,logits_to_keep=1)
                            normal={label:tuple(s.clone() for s in cache_reader_states(draft,output.past_key_values)) for label,draft in drafts.items()}
                            width=min(max(d.block_size for d in drafts.values()),len(reference)-offset)
                            ids=example.inputs['input_ids'].new_tensor(reference[offset:offset+width])[None]
                            verified=target(input_ids=ids,past_key_values=output.past_key_values,use_cache=True)
                            teacher=verified.logits[:,:width-1].detach()
                            del verified
                        for label,draft in drafts.items():
                            normal_logits=normal_candidates=None
                            for mode in ('normal','off','wrong'):
                                states=wrong[label] if mode=='wrong' else normal[label]
                                row,logits,candidates=run_mode(target,draft,example,output,states,reference,offset,mode,teacher[:,:min(draft.block_size-1,teacher.shape[1])])
                                if mode=='normal':normal_logits,normal_candidates=logits,candidates
                                row.update(compare(normal_logits,logits,normal_candidates,candidates))
                                row.update(model=label,dataset=case,sample_id=example.sample_id,offset=offset,
                                    anchor=example.prefill_length+offset,donor_sample_id=examples[other].sample_id,
                                    donor_offset=donor_offset,donor_prompt_tokens=examples[other].prefill_length,
                                    recipient_prompt_tokens=example.prefill_length,
                                    state_rms=[float(s.float().square().mean().sqrt()) for s in states])
                                rows.append(row);detail.write(json.dumps(row,allow_nan=False)+'\n');detail.flush()
                            if label not in probes:
                                probes[label]=gradient_probe(target,draft,example,output,normal[label],reference,offset,teacher[:,:min(draft.block_size-1,teacher.shape[1])])
                                write_json(root/'gradient_probe.json',probes)
                        del output,normal,wrong
                    print(f'{case} sample={index+1}/{len(records)} offset={offset} completed',flush=True)
    summaries=summarize(rows)
    if not summaries:raise ValueError('no usable reference anchors')
    write_json(root/'summary.json',summaries);write_json(root/'skipped.json',skipped)
    with (root/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    lines=['# GDN diagnostic results','','Fixed target sequences and anchors; same checkpoint for all modes.',
        'Gradient probes measure current connectivity, not historical training gradients.',
        'Wrong-state donors are different images/questions. Their prompt lengths may differ; see anchors.jsonl.',
        'CE/L1/flip rates are weighted by supervised positions; reference committed length is an equal-anchor mean.',
        'This is not a throughput benchmark or an AR equivalence test.','',
        '| Model | Dataset | Mode | Anchors | CE | L1 | Reference length | Residual ratio | Candidate change |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['model']} | {s['dataset']} | {s['mode']} | {s['anchors']} | {s['ce_target_greedy']:.5f} | {s['l1_to_target']:.5f} | {s['fixed_reference_committed_length']:.3f} | {s['mean_residual_ratio']:.6g} | {s['free_running_candidate_change_fraction']:.2%} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Diagnostics: {root}/report.md',flush=True)


if __name__=='__main__':main()
