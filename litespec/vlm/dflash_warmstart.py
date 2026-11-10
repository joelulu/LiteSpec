"""Weights-only Full/Pivot1 adaptation from the same original Full DFlash."""
import hashlib
import json
from pathlib import Path


def warm_start_config(cfg, path):
    path = Path(path).expanduser().resolve(strict=True)
    source = path / "config.json"
    parent = json.loads(source.read_text())
    if cfg.get("algorithm") != "dflash" or cfg.get("draft_context", "full") not in ("full", "pivot1"):
        raise ValueError("DFlash warm start supports Full/Pivot1 only")
    if cfg.get("pivot_augmentation") or cfg.get("pivot_two_rounds"):
        raise ValueError("DFlash warm start does not support pivot augmentation/two-round training")
    if parent.get("architectures") != ["DFlashDraftModel"]:
        raise ValueError("Warm start requires an original DFlashDraftModel checkpoint")
    method = parent.get("dflash_config", {})
    metadata = parent.get("vlm_config", {})
    if metadata.get("draft_context", "full") != "full":
        raise ValueError("Both adaptations must start from the original Full checkpoint")
    if method.get("pivot_augmentation") or metadata.get("pivot_augmentation"):
        raise ValueError("Warm start requires an unaugmented Full checkpoint")
    for key, parent_key in (("block_size", "block_size"), ("num_draft_layers", "num_hidden_layers")):
        if key in cfg and cfg[key] != parent[parent_key]:
            raise ValueError(f"warm-start {key} differs from checkpoint")
    layers = method.get("target_layer_ids")
    if not isinstance(layers, list) or not layers or any(type(i) is not int for i in layers):
        raise ValueError("Checkpoint must record its ordered target_layer_ids")
    if any(i < 0 or i >= parent["num_target_layers"] for i in layers):
        raise ValueError("Checkpoint target_layer_ids are outside Target depth")
    if cfg.get("target_layer_ids") and list(cfg["target_layer_ids"]) != layers:
        raise ValueError("warm-start target_layer_ids differ from checkpoint")
    if type(method.get("mask_token_id")) is not int or not 0 <= method["mask_token_id"] < parent["vocab_size"]:
        raise ValueError("Checkpoint must record an in-vocabulary MASK row")
    cfg.update(
        init_checkpoint=str(path), init_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        block_size=parent["block_size"], num_draft_layers=parent["num_hidden_layers"],
        target_layer_ids=list(layers),
    )
    return parent


def load_warm_start(path, target, cfg):
    from transformers import AutoConfig
    from litespec.modeling.draft.dflash import DFlashDraftModel

    parent = warm_start_config(cfg, path)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    text = target.config.text_config
    for key in ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
        if getattr(config, key) != getattr(text, key):
            raise ValueError(f"warm-start checkpoint and Target {key} differ")
    if config.num_target_layers != text.num_hidden_layers:
        raise ValueError("warm-start checkpoint and Target depth differ")
    recorded = parent.get("vlm_config", {}).get("target_model")
    if recorded and Path(recorded).expanduser().resolve() != Path(cfg["target_model"]).expanduser().resolve():
        raise ValueError("warm-start checkpoint Target path differs")
    # Full and Pivot1 have identical parameter shapes. Only visibility changes;
    # preserve every backbone/projector weight, RoPE setting and MASK row.
    config.vlm_config = dict(
        getattr(config, "vlm_config", {}), algorithm="dflash", target_model=cfg["target_model"],
        context_policy=cfg["context_policy"], draft_context=cfg.get("draft_context", "full"),
        warm_start=str(Path(path).resolve()), warm_start_config_sha256=cfg["init_config_sha256"],
    )
    draft, info = DFlashDraftModel.from_pretrained(
        path, config=config, attn_implementation="sdpa", output_loading_info=True,
        local_files_only=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"DFlash warm start must load every parameter exactly: {info}")
    print(f"DFlash weights-only warm start: {Path(path).resolve()}; context={draft.context_window}; fresh optimizer/schedule", flush=True)
    return draft.to(device=target.device, dtype=target.dtype)
