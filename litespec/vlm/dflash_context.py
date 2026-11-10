"""DFlash-only context controls; never modify the Target's image or cache."""
import torch

CONTEXTS = ("full", "recent128", "pivot1", "pivot5", "pivot12")
CONDITIONS = ("correct", "wrong", "mask")


def intervene_visual(features, vision_mask, condition="correct", donor_features=None):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown visual intervention: {condition}")
    keep = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
    mask = torch.zeros_like(keep) if vision_mask is None else vision_mask.to(features.device, torch.bool)
    if mask.ndim == 1:
        mask = mask[None].expand(features.size(0), -1)
    if mask.shape != keep.shape:
        raise ValueError("visual mask must cover exactly the prefill positions")
    metadata = dict(condition=condition, visual_slots=int(mask[0].sum()), donor_slots=0)
    if condition != "correct" and not mask.any():
        raise ValueError("visual intervention requested but no visual slots were found")
    if condition == "mask":
        keep &= ~mask
    elif condition == "wrong":
        if donor_features is None or donor_features.ndim != 3 or donor_features.size(1) == 0:
            raise ValueError("wrong-image intervention needs another sample's visual features")
        if donor_features.size(-1) != features.size(-1):
            raise ValueError("donor Target feature width differs")
        features = features.clone()
        metadata["donor_slots"] = donor_features.size(1)
        for b in range(features.size(0)):
            n = int(mask[b].sum())
            indices = torch.linspace(0, donor_features.size(1)-1, n, device=features.device).round().long()
            features[b, mask[b]] = donor_features[min(b, donor_features.size(0)-1), indices].to(features)
    return features, keep, metadata


@torch.no_grad()
def propose_features(draft, target, features, pending, *, vision_mask=None,
                     condition="correct", donor_features=None):
    """Uncached fixed-prefix oracle with original absolute text positions."""
    anchor = features.size(1)
    features, keep, metadata = intervene_visual(features, vision_mask, condition, donor_features)
    window = getattr(draft, "context_window", None)
    if window is not None:
        features, keep = features[:, -window:], keep[:, -window:]
    ids = pending.new_full((pending.size(0), draft.block_size), draft.mask_token_id)
    ids[:, :1] = pending
    positions = torch.cat([
        torch.arange(anchor-features.size(1), anchor, device=pending.device).repeat_interleave(draft.context_slots),
        torch.arange(anchor, anchor+draft.block_size, device=pending.device)])[None].expand(pending.size(0), -1)
    keep = keep.repeat_interleave(draft.context_slots, -1)
    mask = torch.cat([keep, torch.ones_like(ids, dtype=torch.bool)], -1)[:, None, None, :]
    hidden = draft(position_ids=positions, noise_embedding=target.get_input_embeddings()(ids),
                   target_hidden=features, attention_mask=mask, is_causal=False, use_cache=False)
    tokens = draft._propose_draft_tokens(target, hidden, ids, temperature=0.)[0]
    return tokens, metadata
