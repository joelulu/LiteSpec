"""Select causal prompt-only evidence without stacking full-sequence CHS."""
import torch

from litespec.modeling.draft.pivot_memory import PromptMemory


def build_prompt_memory(states, draft, input_ids, prompt_mask, target_config):
    cfg = draft.config.dlite_config
    mode = cfg.get("memory_mode", "none")
    if mode == "none":
        return None
    if mode not in {"vision", "text"}:
        raise ValueError("memory_mode must be none, vision or text")
    depth = draft.config.num_target_layers
    if len(states) not in {depth, depth + 1}:
        raise ValueError("incompatible target hidden-state tuple")
    offset = int(len(states) == depth + 1)
    image_id = target_config.image_token_id
    visual = input_ids == image_id
    video_id = getattr(target_config, "video_token_id", None)
    if video_id is not None:
        visual |= input_ids == video_id
    allowed = prompt_mask.bool() & (visual if mode == "vision" else ~visual)
    # Do not treat visual delimiters as text evidence.
    if mode == "text":
        for name in ("vision_start_token_id", "vision_end_token_id"):
            token = getattr(target_config, name, None)
            if token is not None:
                allowed &= input_ids != token
    budget = int(cfg.get("memory_max_tokens", 1024))
    if budget < 1:
        raise ValueError("positive memory budget required")
    layers = cfg["memory_layer_ids"]
    if not layers or len(set(layers)) != len(layers) or any(not isinstance(i, int) or not 0 <= i < depth for i in layers):
        raise ValueError("memory layers must be distinct valid target layer indices")
    indices = []
    for row in allowed:
        pos = row.nonzero().flatten()
        if pos.numel() > budget:
            # Deterministic evenly spaced source positions, not a learned compressor.
            pos = pos[torch.linspace(0, pos.numel()-1, budget, device=pos.device).round().long()]
        indices.append(pos)
    width = max(1, max(p.numel() for p in indices))
    values = states[0].new_zeros(input_ids.size(0), width, states[0].size(-1))
    valid = torch.zeros(input_ids.size(0), width, device=input_ids.device, dtype=torch.bool)
    for b, pos in enumerate(indices):
        if pos.numel():
            values[b, :pos.numel()] = sum(states[i+offset][b, pos] for i in layers) / len(layers)
            valid[b, :pos.numel()] = True
    return PromptMemory(values.detach(), valid)


def prompt_mask_from_example(example):
    ids = example.inputs["input_ids"]
    if hasattr(example, "prompt_mask"):
        return example.prompt_mask
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, :example.prefill_length] = True
    return mask & example.inputs.get("attention_mask", torch.ones_like(ids)).bool()


def configure_image_tokens(processor, config, tokens):
    if int(tokens) < 1:
        raise ValueError("positive image token budget required")
    image = processor.image_processor
    vision = config.vision_config
    unit = int(vision.patch_size) * int(vision.spatial_merge_size)
    size = dict(image.size)
    if not {"shortest_edge", "longest_edge"} <= size.keys():
        raise ValueError("unsupported Qwen image size policy")
    size["longest_edge"] = int(tokens) * unit * unit
    size["shortest_edge"] = min(size["shortest_edge"], size["longest_edge"])
    image.size = size
    return size
