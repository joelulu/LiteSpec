"""Direct SFT for image/text DLite students and history-aware controls.

Run: python -m litespec.vlm.train --config scripts/config/vlm/sft_student.yaml
Single-device or torchrun data-parallel direct SFT; not the two-stage trainer.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
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
    if cfg.get("memory_mode", "none") != "none":
        config.dlite_config.update(
            memory_mode=cfg["memory_mode"], memory_max_tokens=int(cfg.get("memory_max_tokens", 1024)),
            memory_rank=int(cfg.get("memory_rank", 256)),
            memory_layer_ids=cfg.get("memory_layer_ids", list(dict.fromkeys([layers[0], layers[len(layers)//2], layers[-1]]))))
    config.vlm_config = {"schema_version": 1, "target_model": cfg["target_model"],
                         "context_policy": cfg["context_policy"], "training": "direct_sft"}
    if cfg.get("image_max_tokens") is not None:
        config.vlm_config["image_max_tokens"] = cfg["image_max_tokens"]
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


def capture_rng():
    return {"python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng_local": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(saved):
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved.get("cuda_rng_local") is not None:
        torch.cuda.set_rng_state(saved["cuda_rng_local"])
    elif saved.get("cuda_rng") is not None:  # Original single-device checkpoints.
        torch.cuda.set_rng_state_all(saved["cuda_rng"])


def epoch_indices(size, rank, world_size, *, seed, epoch, shuffle=False):
    """Equal-size shards, padding the final global batch like DistributedSampler."""
    if size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid dataset size or distributed rank")
    indices = list(range(size))
    if shuffle:
        random.Random(seed + epoch).shuffle(indices)
    total = math.ceil(size / world_size) * world_size
    indices = (indices * math.ceil(total / size))[:total]
    return indices[rank:total:world_size]


class TrainingObjective(torch.nn.Module):
    """Put the entire draft computation inside DDP's forward boundary."""
    def __init__(self, draft):
        super().__init__()
        self.draft = draft

    def forward(self, target, example, anchors, *, policy, l1_weight):
        return train_block(target, self.draft, example, anchors,
                           policy=policy, l1_weight=l1_weight)


def save_checkpoint(path, draft, optimizer, state, cfg, rng_states=None):
    path.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(path)
    torch.save({"optimizer": optimizer.state_dict(), "state": state,
                **capture_rng(), "rng_by_rank": rng_states},
               path / "vlm_training_state.pt")
    (path / "vlm_run.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="checkpoint directory including optimizer state")
    parser.add_argument("--target-model")
    parser.add_argument("--train-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-samples", type=int, help="0 uses all records")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("target_model", "train_data", "output_dir", "max_steps", "max_samples"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg.get("max_samples") == 0:
        cfg["max_samples"] = None
    if cfg.get("max_samples") is not None and cfg["max_samples"] < 1:
        raise ValueError("max_samples must be positive or 0/all")
    for key in ("epochs", "max_steps", "save_steps", "anchors_per_sample"):
        if int(cfg.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    if not args.resume and (Path(cfg["output_dir"]) / "last/vlm_training_state.pt").exists():
        raise ValueError("output already contains a checkpoint; use --resume or a new output directory")
    policy = ContextPolicy(**cfg["context_policy"])
    if cfg.get("model_role", "student") == "student" and not policy.is_empty:
        raise ValueError("student training requires an empty history policy")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device", "cuda")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        if torch.device(device).type == "cuda":
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        dist.init_process_group(backend="gloo" if device == "cpu" else "nccl",
                                timeout=timedelta(minutes=30))
        cfg["world_size"] = world_size
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    target, processor = load_target(cfg["target_model"], device=device, dtype=dtype)
    data_path = Path(cfg["train_data"]).resolve()
    records = list(read_records(data_path, cfg.get("max_samples")))
    if not records:
        raise ValueError("training manifest is empty")
    if rank == 0:
        print(json.dumps({"world_size": world_size, "samples": len(records),
                          "samples_per_rank": math.ceil(len(records) / world_size),
                          "padded_samples_per_epoch": (-len(records)) % world_size,
                          "global_batch_size": world_size, "target_model": cfg["target_model"],
                          "output_dir": cfg["output_dir"]}), flush=True)
    cfg["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    cfg["train_data"] = str(data_path)
    draft = (DLiteDraftModel.from_pretrained(args.resume, attn_implementation="sdpa").to(device, dtype)
             if args.resume else make_draft(target, processor, cfg))
    optimizer = torch.optim.AdamW(draft.parameters(), lr=cfg.get("learning_rate", 1e-4))
    objective = TrainingObjective(draft)
    if world_size > 1:
        objective = DistributedDataParallel(
            objective, device_ids=[torch.cuda.current_device()] if device != "cpu" else None,
            broadcast_buffers=False, find_unused_parameters=True,
        )
    # Identical initialization, independent per-rank anchor sampling afterwards.
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    state = {"step": 0, "epoch": 0, "record": 0}
    if args.resume:
        previous = json.loads((Path(args.resume) / "vlm_run.json").read_text())
        if previous != cfg:
            raise ValueError("resume config/data identity differs from checkpoint")
        saved = torch.load(Path(args.resume) / "vlm_training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        rng_states = saved.get("rng_by_rank")
        if rng_states is not None and len(rng_states) != world_size:
            raise ValueError("resume world size differs from checkpoint")
        restore_rng(rng_states[rank] if rng_states is not None else saved)
    if state["step"] >= cfg.get("max_steps", 100):
        if rank == 0:
            print("checkpoint has already reached max_steps", flush=True)
        return
    draft.train()
    output = Path(cfg["output_dir"])
    def checkpoint():
        rng_states = None
        if world_size > 1:
            rng_states = [None] * world_size if rank == 0 else None
            dist.gather_object(capture_rng(), rng_states, dst=0)
        if rank == 0:
            save_checkpoint(output / "last", draft, optimizer, state, cfg, rng_states)
        if world_size > 1:
            dist.barrier()

    for epoch in range(state["epoch"], cfg.get("epochs", 1)):
        indices = epoch_indices(len(records), rank, world_size, seed=seed,
                                epoch=epoch, shuffle=cfg.get("shuffle", False))
        first = state["record"] if epoch == state["epoch"] else 0
        for index in range(first, len(indices)):
            started = time.perf_counter()
            record = records[indices[index]]
            example = prepare_example(processor, target.config, record, device=device,
                                      training=True, max_length=cfg.get("max_length", 4096),
                                      base_dir=data_path.parent)
            end = example.inputs["input_ids"].size(1) - 2
            valid = list(range(example.prefill_length, end + 1))
            if not valid:
                raise ValueError(f"sample {example.sample_id} has no supervised token after its first answer token")
            anchors = sorted(random.sample(valid, min(cfg.get("anchors_per_sample", 2), len(valid))))
            optimizer.zero_grad(set_to_none=True)
            loss = objective(target, example, anchors, policy=policy,
                             l1_weight=cfg.get("l1_weight", 0.1))
            finite = torch.isfinite(loss).to(torch.int32)
            if world_size > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f"non-finite loss for sample {example.sample_id}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(draft.parameters(), cfg.get("max_grad_norm", 1.0))
            if not torch.isfinite(grad):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            state = {"step": state["step"] + 1, "epoch": epoch, "record": index + 1}
            metrics = loss.detach().float()
            if world_size > 1:
                dist.all_reduce(metrics)
                metrics /= world_size
            loss_value = metrics.item()
            if rank == 0:
                print(json.dumps({**state, "sample_id": example.sample_id,
                                  "loss": loss_value, "grad_norm": float(grad),
                                  "world_size": world_size, "seconds_per_step": time.perf_counter() - started,
                                  "samples_seen": state["step"] * world_size}), flush=True)
            if state["step"] >= cfg.get("max_steps", 100):
                checkpoint()
                return
            if state["step"] % cfg.get("save_steps", 100) == 0:
                checkpoint()
        state["epoch"], state["record"] = epoch + 1, 0
    checkpoint()


def main():
    try:
        run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
