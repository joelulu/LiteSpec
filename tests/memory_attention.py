"""Untimed FP32 reconstruction of the reader's scaled QK attention.

No changes to the production SDPA call. BF16 SDPA may differ numerically; the
actual captured readout and its reconstruction error are reported separately.
"""
import torch


ATTENTION_FIELDS = ('attention_valid_tokens', 'attention_logit_std', 'attention_q_norm',
    'attention_k_norm', 'attention_max_weight', 'attention_peak_vs_uniform',
    'attention_top4_mass', 'attention_top16_mass', 'attention_top64_mass',
    'attention_entropy_normalized', 'attention_effective_tokens', 'attention_effective_fraction',
    'attention_readout_norm', 'attention_value_rms_norm', 'attention_readout_value_ratio',
    'attention_readout_uniform_delta_ratio', 'attention_value_relative_spread',
    'attention_reconstruction_relative_error')


@torch.no_grad()
def attention_summary(query, keys, values, valid, actual_readout):
    q, k, v = query.float(), keys.float(), values.float()
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or valid.shape != k.shape[:2]:
        raise ValueError('expected [B,Q,R], [B,M,R], [B,M,R], [B,M]')
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('attention shape mismatch')
    valid = valid.bool()
    available = valid.any(-1)
    if not available.any():
        return {name:None for name in ATTENTION_FIELDS}, []
    q, k, v, valid = q[available], k[available], v[available], valid[available]
    actual = actual_readout.detach().float()[available]
    scores = (q @ k.transpose(-1, -2)) / q.shape[-1]**.5
    count = valid.sum(-1).float()[:, None].expand(q.shape[:2])
    mask = valid[:, None, :].expand_as(scores)
    score_mean = scores.masked_fill(~mask, 0).sum(-1)/count
    score_std = ((scores-score_mean[..., None]).square().masked_fill(~mask, 0).sum(-1)/count).sqrt()
    weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    entropy = -(weights*weights.clamp_min(1e-30).log()).sum(-1)
    # A single token cannot distinguish selective attention from uniform attention.
    entropy_normalized = torch.where(count > 1, entropy/count.clamp_min(2).log(), torch.zeros_like(entropy))
    effective = entropy.exp()
    v2 = v.square().sum(-1)
    value_rms = (v2.masked_fill(~valid, 0).sum(-1)/valid.sum(-1)).sqrt()[:, None]
    mean_value = v.masked_fill(~valid[..., None], 0).sum(1)/valid.sum(-1)[:, None]
    spread = (((v-mean_value[:, None]).square().sum(-1).masked_fill(~valid, 0).sum(-1)
               / valid.sum(-1)).sqrt() / value_rms[:, 0].clamp_min(1e-12))
    weighted_v_norm = (weights*v2[:, None]).sum(-1).sqrt()
    actual_norm = actual.norm(dim=-1)
    reconstructed = weights @ v
    stats = dict(attention_valid_tokens=float(count.mean()), attention_logit_std=float(score_std.mean()),
        attention_q_norm=float(q.norm(dim=-1).mean()),
        attention_k_norm=float(k.norm(dim=-1)[valid].mean()),
        attention_max_weight=float(weights.max(-1).values.mean()),
        attention_peak_vs_uniform=float((weights.max(-1).values*count).mean()),
        attention_entropy_normalized=float(entropy_normalized.mean()),
        attention_effective_tokens=float(effective.mean()), attention_effective_fraction=float((effective/count).mean()),
        attention_readout_norm=float(actual_norm.mean()), attention_value_rms_norm=float(value_rms.mean()),
        attention_readout_value_ratio=float((actual_norm/weighted_v_norm.clamp_min(1e-12)).mean()),
        attention_readout_uniform_delta_ratio=float(((actual-mean_value[:, None]).norm(dim=-1)/value_rms.clamp_min(1e-12)).mean()),
        attention_value_relative_spread=float(spread.mean()),
        attention_reconstruction_relative_error=float(((actual-reconstructed).norm(dim=-1)/actual_norm.clamp_min(1e-12)).mean()))
    for top in (4, 16, 64):
        stats[f'attention_top{top}_mass'] = float(weights.topk(min(top, weights.shape[-1]), -1).values.sum(-1).mean())
    # Evidence indices refer to selected memory slots, not original image pixels.
    top_weights, slots = weights.topk(min(8, weights.shape[-1]), -1)
    details = []
    for b in range(q.shape[0]):
        for j in range(q.shape[1]):
            n = min(8, int(count[b, j]))
            details.append(dict(query_index=j, valid_tokens=int(count[b,j]),
                slots=slots[b,j,:n].tolist(), weights=top_weights[b,j,:n].tolist()))
    return stats, details
