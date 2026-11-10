"""Layer-specific context projection and block-position layer priors."""
import torch
from torch import nn


class PerLayerContextProjection(nn.Module):
    """Keep parameters as ordinary matrices for FSDP/optimizer compatibility."""

    def __init__(self, slots, hidden_size):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(slots)
        )

    def forward(self, hidden):
        return torch.stack([projection(hidden[:, :, i])
                            for i, projection in enumerate(self.projections)], dim=2)


def layer_prior_score_mod(bias, block_size, slots, context_length):
    """Anchor is offset zero; learned rows apply to predicted offsets 1..b-1.

    Context keys are flattened in time-major, source-layer-minor order. The
    returned scalar rule works for both FlexAttention and dense SDPA indices.
    It never changes visibility, absolute positions, or draft-block key scores.
    """
    def score_mod(score, batch, head, query, key):
        offset = query % block_size
        layer = key % slots
        prior = bias[(offset - 1).clamp(min=0), layer]
        enabled = (offset > 0) & (key < context_length)
        return score + torch.where(enabled, prior, 0.0)
    return score_mod


def dense_layer_prior_mask(mask, bias, block_size, slots, context_length,
                           query_length, key_length, dtype, device):
    """Dense inference/reference path; CUDA training uses the scalar Flex rule."""
    query = torch.arange(query_length, device=device)[:, None]
    key = torch.arange(key_length, device=device)[None, :]
    prior = layer_prior_score_mod(bias, block_size, slots, context_length)(0.0, 0, 0, query, key).to(dtype)
    if mask is None:
        return prior
    if mask.dtype == torch.bool:
        return prior.masked_fill(~mask, torch.finfo(dtype).min)
    return prior + mask
