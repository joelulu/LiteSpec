"""Untimed paired interventions; proposals never modify the fixed AR prefix."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from litespec.vlm.context import ContextPolicy
from litespec.vlm.runtime import draft_block
from litespec.vlm.sw_token import source_layers
from litespec.vlm.pivot_augmentation import gather_positions
from sw_token_probe import collect_ar_prefixes


@torch.no_grad()
def proposal(draft, target, pivot, pending, anchor, data):
    hidden, _ = draft_block(draft,target.get_input_embeddings(),anchor_ids=pending,
        pivot=pivot,history=None,anchor=anchor,prefill_length=anchor,
        vision_mask=None,policy=ContextPolicy.no_history(),augmentation_data=data)
    ids, logits = draft.sample_draft_tokens(draft_hidden=hidden,first_prev_token_ids=pending[:,0],temperature=0.)
    return hidden, ids, logits


@torch.no_grad()
def collect_feedback(target, draft, example, *, cap=128, stride=8, max_probes=8):
    # Keep all AR prefix snapshots so each second-round pivot is Target-exact.
    tokens, snapshots = collect_ar_prefixes(target,draft,example,cap=cap,stride=1,max_probes=cap)
    lookup = {p['offset']:p for p in snapshots}
    rows = []
    for p in snapshots[::stride]:
        if len(rows) >= max_probes: break
        offset, anchor = p['offset'], p['anchor']
        if offset+draft.block_size >= len(tokens): continue
        pivot = p['context'][:,:,-draft.chs_num_layers:].clone()
        _, ids, _ = proposal(draft,target,pivot,p['pending'],anchor,{})
        reference = ids.new_tensor(tokens[offset+1:offset+draft.block_size])[None]
        kept = 1+int((ids == reference).long().cumprod(-1).sum())
        if kept >= draft.block_size: continue  # no rejected position to intervene on
        next_offset = offset+kept
        if next_offset not in lookup: continue
        nxt = lookup[next_offset]
        continuation = p['pending'].new_tensor(tokens[:offset])[None]
        candidates = torch.cat([p['pending'],ids],1)
        inputs = dict(example.inputs)
        extra = torch.cat([continuation,candidates],1)
        inputs['input_ids'] = torch.cat([inputs['input_ids'],extra],1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        if 'mm_token_type_ids' in inputs:
            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(extra)],1)
        inputs.pop('position_ids',None)
        positions = torch.arange(anchor,anchor+draft.block_size,device=ids.device)
        verified = target(**inputs,use_cache=False,output_hidden_states=True,logits_to_keep=positions)
        predictions = verified.logits.argmax(-1)
        raw = gather_positions(source_layers(verified.hidden_states,draft),ids.new_tensor([[anchor+kept]]),draft.target_layer_ids).clone()
        wrong = candidates[:,kept]
        correct = ids.new_tensor([tokens[next_offset]])
        embedding = target.get_input_embeddings()
        data = dict(feedback=raw,feedback_valid=torch.ones(1,device=ids.device,dtype=torch.bool),
            wrong_embedding=embedding(wrong).clone(),correct_embedding=embedding(correct).clone(),distance=torch.ones_like(correct))
        rows.append(dict(sample_id=example.sample_id,anchor=nxt['anchor'],offset=next_offset,
            pivot=nxt['context'][:,:,-draft.chs_num_layers:].clone(),pending=nxt['pending'].clone(),
            reference=nxt['reference'],data=data,
            verifier_bonus_disagrees=bool(predictions[0,kept-1] != correct[0])))
        del verified
    return tokens, rows


@contextmanager
def observe_residual(augmentation):
    saved = augmentation.inputs
    stats = {}
    def wrapped(pivot, query, data):
        p, q = saved(pivot,query,data)
        base, out = (pivot,p) if augmentation.mode=='D1' else (query,q)
        delta = out.float()-base.float()
        stats.update(applied_residual_ratio=float(delta.norm()/base.float().norm().clamp_min(1e-8)),
            applied_element_changed=float(delta.ne(0).float().mean()),
            gate_mean=float(augmentation.gate.weight.sigmoid().mean()) if hasattr(augmentation,'gate') else 0.)
        return p,q
    augmentation.inputs = wrapped
    try: yield stats
    finally: augmentation.inputs = saved


@torch.no_grad()
def evaluate_feedback(draft, target, probe, condition, donor=None):
    if condition not in ('on','off','shuffled'): raise ValueError('unknown feedback condition')
    data = dict(probe['data'])
    if condition=='off': data['feedback_valid'] = torch.zeros_like(data['feedback_valid'])
    if condition=='shuffled':
        if donor is None or donor['sample_id']==probe['sample_id']:
            raise ValueError('shuffled feedback requires a different sample donor')
        # Isolate verifier hidden: retain this probe's boundary token embeddings.
        data['feedback'] = donor['data']['feedback']
    with observe_residual(draft.pivot_augmentation) as residual:
        hidden, ids, _ = proposal(draft,target,probe['pivot'],probe['pending'],probe['anchor'],data)
    ref = ids.new_tensor(probe['reference'])[None]
    n = ref.size(1)
    prev = probe['pending'].expand(1,draft.proposal_length).clone()
    if n>1: prev[:,1:n] = ref[:,:-1]
    latent = draft.sequential_head.forward_teacher_forcing(hidden_states=hidden,prev_token_ids=prev)
    logits = draft.sequential_head.project_logits(latent)[:,:n].float()
    hit = ids[:,:n].eq(ref)
    return dict(condition=condition,proposals=ids[0,:n].tolist(),first_hit=bool(hit[0,0]),
        accepted=1+int(hit.long().cumprod(-1).sum()),candidate_count=n,
        ce=float(F.cross_entropy(logits.flatten(0,1),ref.flatten())),logits=logits,**residual)


def paired_rows(results):
    base = results['off']
    rows = []
    for name, value in results.items():
        row = {k:v for k,v in value.items() if k!='logits'}
        row.update(first_repair=int(value['first_hit'] and not base['first_hit']),
            first_break=int(base['first_hit'] and not value['first_hit']),
            accepted_delta=value['accepted']-base['accepted'],
            candidate_change=sum(a!=b for a,b in zip(value['proposals'],base['proposals']))/value['candidate_count'],
            mean_abs_logit_delta=float((value['logits']-base['logits']).abs().mean()))
        rows.append(row)
    return rows
