"""Untimed, common-prefix memory diagnostics. No optimizer or weight updates."""
from contextlib import AbstractContextManager

import torch
import torch.nn.functional as F


class ReaderCapture(AbstractContextManager):
    def __init__(self, reader):
        self.reader = reader
        self.handles = []
        self.gate = self.residual = self.ungated = None

    def __enter__(self):
        def gate(module, inputs, output):
            self.gate = output.detach().float().sigmoid()
        def ungated(module, inputs, output):
            self.ungated = output.detach()
        def residual(module, inputs, output):
            self.residual = output.detach()
        self.handles = [self.reader.gate.register_forward_hook(gate),
                        self.reader.output.register_forward_hook(ungated),
                        self.reader.register_forward_hook(residual)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        return False

    def summarize(self, query):
        if self.residual is None:
            raise RuntimeError('reader was not called')
        gate = self.gate.flatten()
        residual = self.residual[:, 0:1]
        # Addition happens in the actual model dtype; BF16 can round a tiny
        # nonzero residual away. Report both intended and actually applied delta.
        updated = query + residual
        denom = query.float().norm(dim=-1).clamp_min(1e-12)
        ratio = residual.float().norm(dim=-1) / denom
        applied = (updated.float() - query.float()).norm(dim=-1) / denom
        return dict(gate_mean=float(gate.mean()), gate_min=float(gate.min()),
            gate_max=float(gate.max()), gate_below_001_fraction=float((gate < .01).float().mean()),
            ungated_norm=float(self.ungated.float().norm(dim=-1).mean()),
            residual_norm=float(residual.float().norm(dim=-1).mean()),
            query_norm=float(denom.mean()), residual_query_norm_ratio=float(ratio.mean()),
            applied_query_norm_ratio=float(applied.mean()),
            query_changed_fraction=float((updated != query).float().mean()))


def compare_logits(reference, other):
    a, b = reference.float(), other.float()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('finite, shape-matched logits required')
    delta = a-b
    logp, logq = a.log_softmax(-1), b.log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    agreement = (a.argmax(-1) == b.argmax(-1)).float()
    return dict(logits_mean_abs_delta=float(delta.abs().mean()),
        logits_max_abs_delta=float(delta.abs().max()),
        logits_relative_rms_delta=float(delta.square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)),
        probability_l1=float((p-q).abs().sum(-1).mean()),
        kl_correct_to_condition=float((p*(logp-logq)).sum(-1).mean().clamp_min(0)),
        proposal_token_agreement=float(agreement.mean()),
        proposal_block_exact=bool(agreement.bool().all()),
        per_position_token_agreement=agreement.flatten().tolist())


@torch.no_grad()
def compare_conditions(draft, target, example, pivot, pending, anchor, memories):
    from litespec.vlm.context import ContextPolicy
    from litespec.vlm.runtime import draft_block
    ids = pending.new_full((1, draft.block_size), draft.mask_token_id)
    ids[:, 0] = pending[:, 0]
    query = draft.build_inference_query_embeddings(target.get_input_embeddings(), ids)
    logits, rows, residuals = {}, [], {}
    for condition in ('correct', 'wrong_image', 'disabled'):
        with ReaderCapture(draft.memory_reader) as capture:
            hidden, _ = draft_block(draft, target.get_input_embeddings(), anchor_ids=pending,
                pivot=pivot, history=None, anchor=anchor, prefill_length=example.prefill_length,
                vision_mask=example.vision_mask, policy=ContextPolicy.no_history(),
                prompt_memory=memories.get(condition), disable_memory=condition == 'disabled')
            proposals, scores = draft.sample_draft_tokens(draft_hidden=hidden,
                first_prev_token_ids=pending[:, 0], temperature=0.)
        stats = capture.summarize(query) if condition != 'disabled' else dict(
            gate_mean=None, gate_min=None, gate_max=None, gate_below_001_fraction=None,
            ungated_norm=0., residual_norm=0., query_norm=float(query.float().norm(dim=-1).mean()),
            residual_query_norm_ratio=0., applied_query_norm_ratio=0., query_changed_fraction=0.)
        logits[condition] = scores.detach()
        residuals[condition] = capture.residual
        rows.append(dict(condition=condition, **stats, proposals=proposals[0].tolist()))
    for row in rows:
        condition = row['condition']
        row.update(compare_logits(logits['correct'], logits[condition]))
        if condition != 'disabled':
            a, b = residuals['correct'].float().flatten(), residuals[condition].float().flatten()
            row['residual_cosine_vs_correct'] = float(F.cosine_similarity(a[None], b[None]))
            row['residual_difference_ratio_vs_correct'] = float((a-b).norm()/a.norm().clamp_min(1e-12))
    return rows


def gradient_probe(draft, target, example, pivot, continuation, raw_memory):
    """Current held-out-prefix CE gradient; NOT historical SFT gradient logs."""
    from litespec.vlm.context import ContextPolicy
    from litespec.vlm.runtime import draft_block
    span = min(draft.proposal_length, len(continuation)-1)
    if span < 1:
        return dict(skipped='need at least two AR continuation tokens')
    draft.zero_grad(set_to_none=True)
    with torch.enable_grad():
        tokens = torch.tensor([continuation[:span+1]], device=pivot.device)
        hidden, _ = draft_block(draft, target.get_input_embeddings(), anchor_ids=tokens[:, :1],
            pivot=pivot.detach(), history=None, anchor=example.prefill_length,
            prefill_length=example.prefill_length, vision_mask=example.vision_mask,
            policy=ContextPolicy.no_history(), prompt_memory=raw_memory)
        latent = draft.sequential_head.forward_teacher_forcing(
            hidden_states=hidden[:, :span], prev_token_ids=tokens[:, :span])
        logits = draft.sequential_head.project_logits(latent).float()
        loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
    result = dict(objective='teacher_forced_CE_on_fixed_AR_continuation', historical_training_gradient=False,
        optimizer_step=False, loss=float(loss.detach()), tokens=span, parameters={})
    for name, parameter in draft.memory_reader.named_parameters():
        grad = parameter.grad
        norm = float(parameter.detach().float().norm())
        result['parameters'][name] = dict(parameter_norm=norm, gradient_present=grad is not None,
            gradient_norm=float(grad.detach().float().norm()) if grad is not None else None,
            gradient_finite=bool(torch.isfinite(grad).all()) if grad is not None else None)
    if any(p.grad is not None for p in target.parameters()):
        raise RuntimeError('target must remain frozen in gradient diagnosis')
    draft.zero_grad(set_to_none=True)
    return result


def probe_reference(target, draft, example, donor, reference, *, stride=8, gradients=True):
    """Walk one native HF AR trace; all memory conditions share each exact prefix.

    No condition's proposals enter the target cache. Hence differences in reader
    outputs/logits cannot be attributed to divergent speculative trajectories.
    """
    from litespec.vlm.benchmark import RequestPositionState
    from litespec.vlm.memory import build_prompt_memory, prompt_mask_from_example
    from litespec.vlm.runtime import selected_layers
    if stride < 1 or not reference:
        raise ValueError('positive stride and nonempty reference required')
    raw, memories = {}, {}
    with torch.no_grad(), RequestPositionState(target).activate():
        other = target(**donor.inputs, use_cache=False, output_hidden_states=True, logits_to_keep=1)
        raw['wrong_image'] = build_prompt_memory(other.hidden_states, draft, donor.inputs['input_ids'],
            prompt_mask_from_example(donor), target.config)
        memories['wrong_image'] = draft.memory_reader.project(raw['wrong_image'])
        del other
    rows, grad = [], None
    with RequestPositionState(target).activate():
        with torch.no_grad():
            output = target(**example.inputs, use_cache=True, output_hidden_states=True, logits_to_keep=1)
            cache = output.past_key_values
            raw['correct'] = build_prompt_memory(output.hidden_states, draft, example.inputs['input_ids'],
                prompt_mask_from_example(example), target.config)
            memories['correct'] = draft.memory_reader.project(raw['correct'])
        for position, token in enumerate(reference):
            with torch.no_grad():
                pivot = selected_layers(output.hidden_states, draft.target_layer_ids,
                    draft.config.num_target_layers)[:, -1:].detach().clone()
                next_matches_reference = int(output.logits[:, -1].argmax(-1)) == token
            del output
            if position % stride == 0:
                pending = torch.tensor([[token]], device=pivot.device)
                conditions = compare_conditions(draft, target, example, pivot, pending,
                    example.prefill_length+position, memories)
                for row in conditions:
                    row.update(position=position, target_next_matches_reference=next_matches_reference,
                        memory_tokens=0 if row['condition'] == 'disabled' else int(memories[row['condition']].valid.sum()))
                rows.extend(conditions)
                if gradients and grad is None:
                    grad = gradient_probe(draft, target, example, pivot, reference[position:], raw['correct'])
            if position+1 < len(reference):
                with torch.no_grad():
                    output = target(input_ids=torch.tensor([[token]], device=pivot.device),
                        past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=1)
                    cache = output.past_key_values
    return rows, grad
