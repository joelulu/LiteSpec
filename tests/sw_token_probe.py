"""Untimed fixed-prefix DLite window interventions and actual Q/K probes."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

from litespec.vlm.context import ContextPolicy
from litespec.vlm.runtime import draft_block
from litespec.vlm.sw_token import RecentTargetWindow, set_window
from litespec.vlm.static_batch import make_hybrid_cache

CONDITIONS = ('k1', 'k2', 'k2_pivot_only', 'k2_pivot_only_origin0')


class DraftPrecisionEmbedding:
    """Cast lookup outputs only; never convert shared Target embedding weights."""
    def __init__(self, embedding, dtype):
        self.embedding = embedding
        self.dtype = dtype

    @property
    def weight(self):
        # Keep OOV MASK fallback detection/mean in the original Target dtype.
        return self.embedding.weight

    def __call__(self, ids):
        return self.embedding(ids).to(dtype=self.dtype)


@contextmanager
def preserve_window(draft):
    old = draft.context_window_size
    present = 'context_window_size' in draft.config.dlite_config
    saved = draft.config.dlite_config.get('context_window_size')
    try:
        yield
    finally:
        draft.context_window_size = old
        if present:
            draft.config.dlite_config['context_window_size'] = saved
        else:
            draft.config.dlite_config.pop('context_window_size', None)


class AttentionMass:
    """Observe post-QK-norm/RoPE attention, including block-query keys in denominator.

    Compute in FP32 without changing the production SDPA call. Values are means
    across heads; each row is one actual layer/query. Not a causal attribution.
    """
    def __init__(self, slots_per_position, keep):
        self.slots = slots_per_position
        self.keep = keep
        self.rows = []

    @torch.no_grad()
    def __call__(self, module, q, k, attention_mask, context_length):
        if q.size(0) != 1 or context_length != self.keep.size(-1):
            raise ValueError('probe requires B=1 and explicit context slots')
        if module.sliding_window is not None:
            raise ValueError('attention probe currently requires full-attention Draft layers')
        k = k.repeat_interleave(module.num_key_value_groups, dim=1)
        scores = (q.float() @ k.float().transpose(-1, -2)) * module.scaling
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores.masked_fill_(~attention_mask, float('-inf'))
            else:
                scores = scores + attention_mask.float()
        weights = scores.softmax(-1)[0].mean(0)
        old_end = context_length - self.slots
        old_count = int(self.keep[0, :old_end].sum())
        pivot_count = int(self.keep[0, old_end:context_length].sum())
        for query, row in enumerate(weights):
            old = float(row[:old_end].sum())
            pivot = float(row[old_end:context_length].sum())
            block = float(row[context_length:].sum())
            self.rows.append(dict(layer=module.layer_idx, query=query,
                old_mass=old, pivot_mass=pivot, block_mass=block,
                old_valid_slots=old_count, pivot_valid_slots=pivot_count,
                old_mass_per_slot=old / old_count if old_count else 0.,
                pivot_mass_per_slot=pivot / pivot_count if pivot_count else 0.,
                mass_sum=old+pivot+block))


@contextmanager
def observe_attention(draft, observer):
    modules = [layer.self_attn for layer in draft.layers]
    prior = [(hasattr(m, '_sw_attention_observer'), getattr(m, '_sw_attention_observer', None)) for m in modules]
    try:
        for module in modules:
            module._sw_attention_observer = observer
        yield
    finally:
        for module, (exists, value) in zip(modules, prior):
            if exists:
                module._sw_attention_observer = value
            else:
                del module._sw_attention_observer


@torch.no_grad()
def collect_ar_prefixes(target, draft, example, *, cap=256, stride=8, max_probes=16):
    """Only Target AR changes its cache. Snapshot K=2 BEFORE processing known anchor.

    Candidates from all interventions are evaluated later and never enter Target.
    Last prompt positions are sliced before stacking selected layers.
    """
    cache = make_hybrid_cache(target, example.prefill_length + cap + 1)
    output = target(**example.inputs, past_key_values=cache, use_cache=True,
                    output_hidden_states=True, logits_to_keep=1)
    cache = output.past_key_values
    with preserve_window(draft):
        set_window(draft, 2)
        recent = RecentTargetWindow(draft, output.hidden_states, example.prefill_length)
        pending = output.logits[:, -1:].argmax(-1)
        del output
        eos = getattr(target.config, 'text_config', target.config).eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        tokens, probes = [], []
        for offset in range(cap):
            token = int(pending.item())
            tokens.append(token)
            if token in eos:
                break
            if offset % stride == 0 and len(probes) < max_probes:
                context, keep = recent.context()
                probes.append(dict(offset=offset, anchor=example.prefill_length+offset,
                    context=context.clone(), keep=keep.clone(), pending=pending.clone()))
            if offset + 1 == cap:
                break
            output = target(input_ids=pending, past_key_values=cache, use_cache=True,
                            output_hidden_states=True, logits_to_keep=1)
            cache = output.past_key_values
            recent.commit(output.hidden_states, 1, example.prefill_length+offset)
            pending = output.logits[:, -1:].argmax(-1)
            del output
    eligible = []
    for probe in probes:
        reference = tokens[probe['offset']+1:probe['offset']+1+draft.proposal_length]
        if reference:
            probe['reference'] = reference
            eligible.append(probe)
    return tokens, eligible


@torch.no_grad()
def run_condition(draft, embedding, probe, condition, *, capture=True):
    if condition not in CONDITIONS:
        raise ValueError('unknown condition')
    with preserve_window(draft):
        k = 1 if condition == 'k1' else 2
        set_window(draft, k)
        slots = draft.chs_num_layers
        context = probe['context'].to(dtype=draft.dtype)
        keep = probe['keep'].clone()
        if k == 1:
            context, keep = context[:, :, -slots:], keep[:, -slots:]
        elif condition.startswith('k2_pivot_only'):
            keep[:, :-slots] = False
        observer = AttentionMass(slots, keep)
        shift = -1 if condition == 'k2_pivot_only_origin0' else 0
        with observe_attention(draft, observer if capture else None):
            hidden, _ = draft_block(draft, DraftPrecisionEmbedding(embedding, draft.dtype), anchor_ids=probe['pending'],
                pivot=context, pivot_keep=keep, history=None, anchor=probe['anchor'],
                prefill_length=probe['anchor']-probe['offset'], vision_mask=None,
                policy=ContextPolicy.no_history(), rotary_shift=shift)
        proposals, _ = draft.sample_draft_tokens(draft_hidden=hidden,
            first_prev_token_ids=probe['pending'][:, 0], temperature=0.)
        reference = torch.tensor(probe['reference'], device=hidden.device, dtype=torch.long)
        n = reference.numel()
        # Teacher-forced logits share the same previous Target tokens in ALL conditions.
        previous = probe['pending'].expand(1, draft.proposal_length).clone()
        if n > 1:
            previous[:, 1:n] = reference[:-1]
        latent = draft.sequential_head.forward_teacher_forcing(hidden_states=hidden, prev_token_ids=previous)
        logits = draft.sequential_head.project_logits(latent)[:, :n].float()
        matched = proposals[0, :n].eq(reference)
        return dict(hidden=hidden, logits=logits, proposals=proposals,
            first_hit=bool(matched[0]), accepted=1+int(matched.long().cumprod(0).sum()),
            candidate_count=n, ce=float(F.cross_entropy(logits.flatten(0,1), reference)),
            attention=observer.rows)


def compare_conditions(base, current, *, atol, rtol):
    delta_h = (base['hidden'].float()-current['hidden'].float()).abs()
    delta_l = (base['logits']-current['logits']).abs()
    n = current['candidate_count']
    return dict(hidden_max_abs=float(delta_h.max()), hidden_mean_abs=float(delta_h.mean()),
        logits_max_abs=float(delta_l.max()), logits_mean_abs=float(delta_l.mean()),
        hidden_close=torch.allclose(base['hidden'].float(),current['hidden'].float(),atol=atol,rtol=rtol),
        logits_close=torch.allclose(base['logits'],current['logits'],atol=atol,rtol=rtol),
        first_proposal_equal=bool(base['proposals'][0,0]==current['proposals'][0,0]),
        proposal_agreement=float(base['proposals'][0,:n].eq(current['proposals'][0,:n]).float().mean()))

