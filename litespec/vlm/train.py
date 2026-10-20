"""Direct SFT for image/text DLite students and history-aware controls.

Run: python -m litespec.vlm.train --config scripts/config/vlm/sft_student.yaml
This is a single-device reference trainer, not the distributed/two-stage runner.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import Qwen3Config

from litespec.modeling.draft.dlite import DLiteDraftModel, build_target_layer_ids
from .context import ContextPolicy
from .runtime import draft_block, load_target, prepare_example, read_records, selected_layers


def make_draft(target, processor, cfg):
    text = target.config.text_config
    depth = text.num_hidden_layers
    used = set(processor.tokenizer.get_vocab().values())
    mask_id = next((i for i in range(text.vocab_size - 1, -1, -1) if i not in used), None)
    if mask_id is None:
        raise ValueError("no unused in-vocabulary MASK row")
    layers = cfg.get("target_layer_ids") or build_target_layer_ids(depth, cfg.get("chs_num_layers", 4))
    config = Qwen3Config(
        vocab_size=text.vocab_size, hidden_size=text.hidden_size,
        intermediate_size=text.intermediate_size, num_hidden_layers=cfg.get("num_draft_layers", 2),
        num_attention_heads=text.num_attention_heads, num_key_value_heads=text.num_key_value_heads,
        head_dim=text.head_dim, rms_norm_eps=text.rms_norm_eps,
        max_position_embeddings=text.max_position_embeddings,
        layer_types=["full_attention"] * cfg.get("num_draft_layers", 2),
    )
    config.num_target_layers = depth
    config.block_size = int(cfg.get("block_size", 8))
    if config.block_size < 2:
        raise ValueError("block_size must be >= 2")
    role = cfg.get("model_role", "student")
    config.dlite_config = dict(
        architecture_version="dlite_v2", model_role=role, chs_num_layers=len(layers),
        target_layer_ids=layers, sequential_head="rnn", sequential_rank=cfg.get("sequential_rank", 64),
        mask_token_id=mask_id,
    )
    if role == "swa_teacher":
        # The VLM reference path explicitly selects history positions; it does
        # not invoke the legacy SWA inference/training path.
        config.dlite_config.update(swa_window_size=cfg.get("max_length", 4096),
                                   history_layer_ids=[0, depth // 2, depth - 1])
    config._attn_implementation = "sdpa"
    config.vlm_config = {"schema_version": 1, "target_model": cfg["target_model"],
                         "context_policy": cfg["context_policy"], "training": "direct_sft"}
    return DLiteDraftModel(config).to(device=target.device, dtype=target.dtype)


def train_block(target, draft, example, anchors, *, policy, l1_weight=0.1):
    ids = example.inputs["input_ids"]
    span = draft.proposal_length
    indices = torch.cat([torch.arange(a, a + span, device=ids.device).clamp(max=ids.size(1)-1)
                         for a in anchors])
    with torch.no_grad():
        outputs = target(**example.inputs, use_cache=False, output_hidden_states=True,
                         logits_to_keep=indices)
        states = outputs.hidden_states
    losses = []
    for i, anchor in enumerate(anchors):
        pivot = selected_layers(states, draft.target_layer_ids, draft.config.num_target_layers)[:, anchor-1:anchor]
        history = None if draft.is_student else selected_layers(
            states, draft.history_layer_ids, draft.config.num_target_layers
        )[:, :anchor]
        hidden, _ = draft_block(
            draft, target.get_input_embeddings(), anchor_ids=ids[:, anchor], pivot=pivot,
            history=history, anchor=anchor, prefill_length=example.prefill_length,
            vision_mask=example.vision_mask, policy=policy,
        )
        offsets = torch.arange(span, device=ids.device)
        valid = anchor + offsets + 1 < ids.size(1)
        previous = ids[:, (anchor + offsets).clamp(max=ids.size(1)-1)]
        labels = ids[:, (anchor + offsets + 1).clamp(max=ids.size(1)-1)].clone()
        labels[:, ~valid] = -100
        latent = draft.sequential_head.forward_teacher_forcing(hidden_states=hidden, prev_token_ids=previous)
        logits = draft.sequential_head.project_logits(latent).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        reference = outputs.logits[:, i * span:(i + 1) * span].float().softmax(-1)
        l1 = (logits.softmax(-1) - reference).abs().sum(-1)[:, valid].mean()
        losses.append(ce + float(l1_weight) * l1)
    return torch.stack(losses).mean()


def save_checkpoint(path, draft, optimizer, state, cfg):
    path.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "state": state,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               path / "vlm_training_state.pt")
    (path / "vlm_run.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="checkpoint directory including optimizer state")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    policy = ContextPolicy(**cfg["context_policy"])
    if cfg.get("model_role", "student") == "student" and not policy.is_empty:
        raise ValueError("student training requires an empty history policy")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device", "cuda")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    target, processor = load_target(cfg["target_model"], device=device, dtype=dtype)
    data_path = Path(cfg["train_data"]).resolve()
    records = list(read_records(data_path, cfg.get("max_samples")))
    if not records:
        raise ValueError("training manifest is empty")
    cfg["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    cfg["train_data"] = str(data_path)
    draft = (DLiteDraftModel.from_pretrained(args.resume, attn_implementation="sdpa").to(device, dtype)
             if args.resume else make_draft(target, processor, cfg))
    optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.get("learning_rate", 1e-4))
    state = {"step": 0, "epoch": 0, "record": 0}
    if args.resume:
        previous = json.loads((Path(args.resume) / "vlm_run.json").read_text())
        if previous != cfg:
            raise ValueError("resume config/data identity differs from checkpoint")
        saved = torch.load(Path(args.resume) / "vlm_training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    if state["step"] >= cfg.get("max_steps", 100):
        print("checkpoint has already reached max_steps", flush=True)
        return
    draft.train()
    output = Path(cfg["output_dir"])
    for epoch in range(state["epoch"], cfg.get("epochs", 1)):
        first = state["record"] if epoch == state["epoch"] else 0
        for index in range(first, len(records)):
            example = prepare_example(processor, target.config, records[index], device=device,
                                      training=True, max_length=cfg.get("max_length", 4096),
                                      base_dir=data_path.parent)
            end = example.inputs["input_ids"].size(1) - 2
            valid = list(range(example.prefill_length, end + 1))
            if not valid:
                raise ValueError(f"sample {example.sample_id} has no supervised token after its first answer token")
            anchors = sorted(random.sample(valid, min(cfg.get("anchors_per_sample", 2), len(valid))))
            optimizer.zero_grad(set_to_none=True)
            loss = train_block(target, draft, example, anchors, policy=policy,
                               l1_weight=cfg.get("l1_weight", 0.1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss for sample {example.sample_id}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(draft.parameters(), cfg.get("max_grad_norm", 1.0))
            optimizer.step()
            state = {"step": state["step"] + 1, "epoch": epoch, "record": index + 1}
            print(json.dumps({**state, "sample_id": example.sample_id, "loss": loss.item(), "grad_norm": float(grad)}), flush=True)
            if state["step"] % cfg.get("save_steps", 100) == 0:
                save_checkpoint(output / "last", draft, optimizer, state, cfg)
            if state["step"] >= cfg.get("max_steps", 100):
                save_checkpoint(output / "last", draft, optimizer, state, cfg)
                return
        state["epoch"], state["record"] = epoch + 1, 0
    save_checkpoint(output / "last", draft, optimizer, state, cfg)


if __name__ == "__main__":
    main()
