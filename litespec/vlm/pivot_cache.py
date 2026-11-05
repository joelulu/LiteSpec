"""One visual encoding and one prefix prefill for independent rollout anchors.

Branches restore the cached prefix, not a rejected candidate's recurrent state.
The accepted short block may be replayed for round-two teacher forcing; the
image encoder and long prefix are never replayed. No variable-row GDN commit
or padded-tail recurrent update is assumed.
"""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from litespec.modeling.draft.dlite import (
    _begin_recurrent_cache_transaction, _crop_target_cache,
    _restore_recurrent_cache_transaction,
)
from .pivot_augmentation import gather_positions
from .sw_token import source_layers
from .static_batch import make_hybrid_cache


def left_padded_prefix(inputs, anchors, pad_id):
    """Each independent row ends at its real pivot, with padding only on left."""
    ids = inputs["input_ids"]
    lengths = anchors[:, 0]
    width = int(lengths.max())
    pos = torch.arange(width, device=ids.device)[None] - (width-lengths)[:, None]
    keep = pos >= 0
    safe = pos.clamp(min=0)
    result = dict(inputs)
    result["input_ids"] = ids.gather(1,safe).masked_fill(~keep,pad_id)
    result["attention_mask"] = keep.long()
    if "mm_token_type_ids" in inputs:
        result["mm_token_type_ids"] = inputs["mm_token_type_ids"].gather(1,safe).masked_fill(~keep,0)
    result.pop("position_ids",None)  # recompute MRoPE on this exact padded prefix
    original_mask = inputs.get("attention_mask",torch.ones_like(ids)).bool()
    if not original_mask.gather(1,safe)[keep].all():
        raise ValueError("rollout prefix crosses right padding")
    return result


@torch.no_grad()
def shared_visual_features(target, inputs, copies):
    """Encode the ORIGINAL microbatch once, then repeat small pooled features."""
    if "pixel_values_videos" in inputs or "video_grid_thw" in inputs:
        raise ValueError("rollout training currently supports image/text data only")
    if inputs.get("pixel_values") is None:
        return None
    image = target.get_image_features(inputs["pixel_values"],inputs["image_grid_thw"],return_dict=True)
    values = image.pooler_output
    if not isinstance(values,(tuple,list)):
        raise RuntimeError("expected per-image Qwen3.5 pooler outputs; unsupported Transformers ABI")
    return {"image":BaseModelOutputWithPooling(pooler_output=tuple(values)*copies)}


class CachedRolloutPrefix:
    def __init__(self, target, draft, original_inputs, repeated_inputs, anchors, copies):
        self.target = target
        text = target.config.text_config
        pad = text.pad_token_id if text.pad_token_id is not None else 0
        inputs = left_padded_prefix(repeated_inputs,anchors,pad)
        self.mask = inputs["attention_mask"]
        self.length = inputs["input_ids"].size(1)
        self.target_tokens = int(self.mask.numel())
        self.branch_tokens = 0
        self.branch_calls = 0
        self.vision_encodes = int(original_inputs.get("pixel_values") is not None)
        with torch.no_grad():
            image = shared_visual_features(target,original_inputs,copies)
            inputs.pop("pixel_values",None)
            if image is not None:
                inputs["mm_encoder_outputs"] = image
            # Make continuation coordinates independent of mutable model.rope_deltas.
            if inputs.get("image_grid_thw") is not None:
                if "mm_token_type_ids" not in inputs:
                    raise ValueError("image rollout requires explicit mm_token_type_ids")
                position, delta = target.model.get_rope_index(input_ids=inputs["input_ids"],
                    image_grid_thw=inputs["image_grid_thw"], attention_mask=self.mask,
                    mm_token_type_ids=inputs["mm_token_type_ids"])
                # HF returns delta relative to UNPADDED token count. Cache
                # coordinates include left padding, so compensate explicitly.
                delta = delta + anchors - self.length
            else:
                position = (self.mask.cumsum(-1)-1).masked_fill(self.mask==0,0)[None].expand(3,-1,-1)
                delta = self.mask.sum(-1,keepdim=True)-self.length
            self.delta = delta.detach().clone()
            inputs["position_ids"] = position
            self.cache = make_hybrid_cache(target,self.length+2*draft.block_size)
            output = target(**inputs,past_key_values=self.cache,use_cache=True,
                output_hidden_states=True,logits_to_keep=1)
            source = source_layers(output.hidden_states,draft)
            self.pivot = gather_positions(source,anchors.new_full(anchors.shape,self.length-1),
                draft.target_layer_ids).detach().clone()

    @torch.no_grad()
    def run(self, tokens, *, rollback=True, hidden=True):
        if self.cache.get_seq_length() != self.length:
            raise RuntimeError("rollout branch did not restore its original prefix")
        b, width = tokens.shape
        snapshot = _begin_recurrent_cache_transaction(self.cache) if rollback else None
        if rollback and snapshot is None:
            raise RuntimeError("cached rollout requires recordable Qwen3.5 recurrent states")
        position = (torch.arange(self.length,self.length+width,device=tokens.device)[None]
                    + self.delta).unsqueeze(0).expand(3,-1,-1)
        mask = torch.cat([self.mask,self.mask.new_ones(b,width)],-1)
        try:
            output = self.target(input_ids=tokens,attention_mask=mask,position_ids=position,
                past_key_values=self.cache,use_cache=True,output_hidden_states=hidden)
            self.branch_calls += 1
            self.branch_tokens += tokens.numel()
        except BaseException:
            # A partially updated cache is never reused after a failed Target call.
            self.cache = None
            raise
        if rollback:
            _crop_target_cache(self.cache,current_length=self.length+width,tokens_to_remove=width)
            _restore_recurrent_cache_transaction(snapshot)
        return output
