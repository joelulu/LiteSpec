"""Qwen3.5 image inputs with the upstream DLite SFT objective/runner/optimizer.

The legacy ``vlm.train`` entry remains available for old smoke checkpoints.
Production CUDA execution uses FSDP1 SHARD_GRAD_OP and requires torch 2.9.1
for the upstream Muon layout adapter. CPU execution is a numerical test path.
"""
import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
import yaml

from litespec.core.dlite import OnlineDLiteModel
from litespec.training.algorithms import Phase, RunState, TrainingSession
from litespec.training.fsdp_layout import build_fsdp_layouts
from litespec.training.muon import muon_config_from_args
from litespec.training.optimizer import BF16Optimizer, optimizer_step_count, register_optimizer_parameters
from litespec.training.runner import configure_rank_caches, run_draft
from .runtime import load_target, prepare_example, read_records
from .train import make_draft


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def load_config(path):
    source = yaml.safe_load(Path(path).read_text())
    root = Path(__file__).resolve().parents[2]
    reference = root / source.pop("upstream_config")
    upstream = yaml.safe_load(reference.read_text())
    if upstream["mode"] != "sft":
        raise ValueError("this VLM adapter implements the upstream direct-SFT recipe")
    cfg = {**upstream["arguments"], **source}
    if cfg.get("batch_size") != 1 or cfg.get("model_role", "student") != "student":
        raise ValueError("VLM adapter requires one image per rank and the no-history student")
    if cfg["num_epochs"] < 1 or cfg["accumulation_steps"] < 1 or cfg["num_anchors"] < 1:
        raise ValueError("epochs, accumulation and anchors must be positive")
    cfg["upstream_config_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    cfg["context_policy"] = dict(prefill_text="drop", prefill_vision="drop", decode_mode="none",
                                 window_size=0, implementation="compact")
    if isinstance(cfg.get("target_layer_ids"), str):
        cfg["target_layer_ids"] = [int(x) for x in cfg["target_layer_ids"].split(",")]
    return cfg


def loss_mask(example):
    mask = torch.zeros_like(example.inputs["input_ids"], dtype=torch.float32)
    mask[:, example.prefill_length:] = 1
    return mask


def has_anchor(example, block_size):
    # Match setup._has_valid_anchor_supervision AND the upstream sampler.
    mask = loss_mask(example)
    if int(mask.sum().item()) < 2 * block_size:
        return False
    end = max(mask.size(1) - block_size, 0)
    return bool(((mask[:, :end + 1] > .5) & (mask[:, 1:end + 2] > .5)).any())


def preflight(records, processor, config, cfg, base_dir):
    """Rank zero fixes the eligible set before equal-size distributed sampling."""
    packet = None
    if rank() == 0:
        try:
            from tqdm import tqdm
            valid, skipped = [], []
            for index, record in enumerate(tqdm(records, desc="VLM input validation", unit="sample")):
                try:
                    example = prepare_example(processor, config, record, device="cpu", training=True,
                                              max_length=cfg["max_length"], base_dir=base_dir)
                except ValueError as exc:
                    skipped.append({"index": index, "id": record["id"], "reason": str(exc)})
                    continue
                if has_anchor(example, cfg["block_size"]):
                    valid.append(index)
                else:
                    skipped.append({"index": index, "id": record["id"],
                                    "reason": "upstream requires >=2*block_size answer labels and one full-block anchor"})
            packet = {"valid": valid, "skipped": skipped}
        except Exception as exc:
            packet = {"error": repr(exc)}
    if dist.is_initialized():
        payload = [packet]
        dist.broadcast_object_list(payload, src=0)
        packet = payload[0]
    if "error" in packet:
        raise RuntimeError(f"VLM input validation failed: {packet['error']}")
    return packet


class CPUReferenceDLite(OnlineDLiteModel):
    """Numerical tests only: independent student blocks as an SDPA batch.

    Production CUDA uses the unchanged upstream FlexAttention implementation.
    """
    def forward_backbone(self, batch, *, seq_len):
        bsz, blocks = batch.anchor_positions.shape
        d = self.draft_model
        context, positions = d.build_block_position_ids(
            batch.anchor_positions, batch.token_position_ids, batch.token_keep_mask)
        query = batch.query_embeddings.reshape(bsz * blocks, d.draft_query_length, -1)
        context = context.reshape(bsz * blocks, -1)
        positions = positions.reshape(bsz * blocks, -1)
        output = d(noise_embedding=query,
                   target_hidden=batch.target_hidden.reshape(bsz * blocks, 1, d.chs_num_layers, -1),
                   history_hidden=query.new_empty(bsz * blocks, 1, 0, query.size(-1)),
                   position_ids=positions, rotary_position_ids=torch.cat([context, positions], -1),
                   attention_mask=None, is_causal=False)
        return d._prediction_hidden(output).reshape(bsz, blocks, d.proposal_length, -1)


def make_online(target, draft, cfg):
    cls = CPUReferenceDLite if draft.device.type == "cpu" else OnlineDLiteModel
    return cls(
        draft, target.get_output_embeddings(), target.get_input_embeddings(),
        draft.mask_token_id, cfg["block_size"], attention_backend="flex_attention",
        flex_attention_backend=cfg["flex_attention_backend"], num_anchors=cfg["num_anchors"],
        loss_decay_gamma=cfg.get("loss_decay_gamma"), final_ce_weight=cfg["final_ce_weight"],
        l1_loss_weight=cfg["l1_loss_weight"], base_lm_ce_weight=cfg["base_lm_ce_weight"],
        base_lm_ce_decay_gamma=cfg.get("base_lm_ce_decay_gamma"), use_target_greedy_ce_labels=True,
    )


def prepare_forward(target, online, example):
    """Original anchors/labels/positions; HF target retains its image and MRoPE inputs."""
    ids = example.inputs["input_ids"]
    mask = loss_mask(example)
    anchors, keep = online.sample_anchor_positions(ids.size(1), mask)
    offsets = torch.arange(online.block_size - 1, device=ids.device)
    positions = anchors.unsqueeze(-1) + offsets
    # Project only distinct supervised text positions, then gather the block layout.
    unique, inverse = positions.reshape(-1).unique(sorted=True, return_inverse=True)
    with torch.no_grad():
        output = target(**example.inputs, use_cache=False, output_hidden_states=True,
                        logits_to_keep=unique)
        reference = output.logits[:, inverse].reshape(*positions.shape, -1)
        batch = online.prepare_batch(ids, output.hidden_states, mask,
                                     anchor_positions=anchors, block_keep_mask=keep)
    return dict(prepared_batch=batch, seq_len=ids.size(1), target_prefill_logits=reference,
                target_logits_are_gathered=True)


class VLMAdapter:
    algorithm = "dlite_vlm"
    modes = ()

    def enter_phase(self, session, phase):
        session.draft.train()

    def compute_step(self, session, example, *, microbatch_start=None, microbatch_end=None,
                     collect_detailed_metrics=True):
        inputs = prepare_forward(session.target, session.draft_online, example)
        return session.fsdp_model(**inputs, collect_detailed_metrics=collect_detailed_metrics)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda_local=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda_local"] is not None:
        torch.cuda.set_rng_state(value["cuda_local"])


def save(session, name):
    """Atomic, rank-local FP32/Muon checkpoint plus an inference-compatible export."""
    root = Path(session.args.output_dir)
    destination = root / f"step_{session.state.optimizer_step:07d}"
    exists = destination.exists() if rank() == 0 else None
    if dist.is_initialized():
        values = [exists]
        dist.broadcast_object_list(values, src=0)
        exists = values[0]
    if exists:
        return  # A boundary/final save can share the same optimizer step.
    staging = str(root / (".checkpoint-" + uuid.uuid4().hex)) if rank() == 0 else None
    if dist.is_initialized():
        values = [staging]
        dist.broadcast_object_list(values, src=0)
        staging = values[0]
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    state = dict(schema="vlm_upstream_sft_v1", config=session.vlm_config,
                 cursor=asdict(session.state), rng=rng_state(), optimizer=session.optimizer.state_dict())
    torch.save(state, staging / f"training_state_rank_{rank():05d}.pt")
    if isinstance(session.fsdp_model, FSDP):
        with FSDP.state_dict_type(session.fsdp_model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full = session.fsdp_model.state_dict()
        weights = {key.split("draft_model.", 1)[1]: value for key, value in full.items()
                   if "draft_model." in key} if rank() == 0 else None
    else:
        weights = session.draft.state_dict()
    if dist.is_initialized():
        dist.barrier()
    if rank() == 0:
        session.draft.save_pretrained(staging, state_dict=weights)
        (staging / "vlm_run.json").write_text(json.dumps(session.vlm_config, indent=2))
        (staging / "checkpoint.json").write_text(json.dumps(dict(world_size=world(), cursor=asdict(session.state))))
        os.replace(staging, destination)
        link = root / (".last-" + uuid.uuid4().hex)
        link.symlink_to(destination.name, target_is_directory=True)
        os.replace(link, root / "last")
        print(f"Saved checkpoint: {destination}", flush=True)
    if dist.is_initialized():
        dist.barrier()


def source_batches(session, phase):
    """Original DistributedSampler(drop_last=False) + DataLoader(drop_last=True)."""
    for epoch in range(session.state.epoch, phase.epochs):
        session.sampler.set_epoch(epoch)
        start = session.state.next_batch if epoch == session.state.epoch else 0
        for index, item in enumerate(session.sampler):
            if index < start:
                continue
            if session.state.phase_step >= phase.batch_count(session.steps_per_epoch):
                return
            yield prepare_example(session.processor, session.target.config, session.records[item],
                                  device=session.device, training=True, max_length=session.vlm_config["max_length"],
                                  base_dir=session.base_dir)


def log(session, window, grad_norm):
    metrics = window.metrics.reduce()
    scalars = {k: float(v) for k, v in metrics.items()}
    scalars.update({f"learning_rate/{k}": float(v)
                    for k, v in session.optimizer.get_learning_rates().items()})
    scalars.update(grad_norm=float(grad_norm),
                   epoch_progress=session.state.global_step / session.steps_per_epoch)
    session.tracker.log(scalars, step=session.state.optimizer_step)
    completed = session.state.global_step // session.steps_per_epoch
    if rank() == 0:
        print(json.dumps(dict(optimizer_step=session.state.optimizer_step,
                              micro_step=session.state.global_step, epoch=session.state.epoch,
                              completed_epochs=completed, loss=float(metrics["objective/loss"]),
                              grad_norm=grad_norm, learning_rates=session.optimizer.get_learning_rates(),
                              metrics={k: float(v) for k, v in metrics.items()})), flush=True)
    # Preserve upstream cross-epoch accumulation: export at the first update
    # boundary after an epoch, never discard or serialize partial gradients.
    if completed > session.last_exported_epoch:
        save(session, f"epoch_{completed}")
        session.last_exported_epoch = completed


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for flag in ("target-model", "train-data", "output-dir", "resume"):
        parser.add_argument("--" + flag)
    parser.add_argument("--max-steps", type=int, help="explicit debug optimizer-update cap; unset trains all six epochs")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--report-to", choices=["none", "tensorboard", "wandb"],
                        help="optional scalar tracking; defaults to YAML report_to")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.report_to is not None:
        cfg["report_to"] = args.report_to
    for key in ("target_model", "train_data", "output_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if (args.max_steps is not None and args.max_steps < 1) or args.max_samples < 0:
        raise ValueError("debug limits must be positive (max-samples=0 means all)")
    cfg.update(debug_max_steps=args.max_steps, debug_max_samples=args.max_samples)
    cfg["train_data"] = str(Path(cfg["train_data"]).resolve())
    cfg["output_dir"] = str(Path(cfg["output_dir"]).resolve())
    cfg["target_model"] = str(Path(cfg["target_model"]).resolve())
    device = cfg.get("device", "cuda")
    if device != "cpu":
        if torch.__version__.split("+")[0] != "2.9.1":
            raise ValueError("aligned CUDA Muon training requires torch==2.9.1; do not upgrade the working VLM environment")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = f"cuda:{torch.cuda.current_device()}"
    if "RANK" in os.environ:
        dist.init_process_group("gloo" if device == "cpu" else "nccl", timeout=timedelta(minutes=60))
    if device == "cpu" and world() > 1:
        raise ValueError("multi-rank training uses the CUDA/FSDP path")
    cfg["world_size"] = world()
    root = Path(cfg["output_dir"])
    if not args.resume and (root / "last").exists():
        raise ValueError("output has a checkpoint; choose a new output directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    target, processor = load_target(cfg["target_model"], device=device,
                                    dtype=torch.float32 if device == "cpu" else torch.bfloat16)
    path = Path(cfg["train_data"])
    records = list(read_records(path, args.max_samples or None))
    cfg["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    eligible = preflight(records, processor, target.config, cfg, path.parent)
    cfg["eligible_indices_sha256"] = hashlib.sha256(json.dumps(eligible["valid"]).encode()).hexdigest()
    records = [records[i] for i in eligible["valid"]]
    if not records:
        raise ValueError("no samples contain upstream-compatible supervised blocks")
    if rank() == 0:
        (root / "skipped_training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in eligible["skipped"]))
    from torch.utils.data.distributed import DistributedSampler
    # Upstream prepare_dp_dataloaders leaves DistributedSampler.seed at default 0.
    cfg["sampler_seed"] = 0
    sampler = DistributedSampler(records, num_replicas=world(), rank=rank(), shuffle=True,
                                 seed=cfg["sampler_seed"], drop_last=False)
    phase = Phase("sft", 0, cfg["num_epochs"], schedule_full_epochs=True,
                  max_batches=args.max_steps * cfg["accumulation_steps"] if args.max_steps else None)
    total = optimizer_step_count([phase], len(sampler), cfg["accumulation_steps"])
    cfg.update(eligible_samples=len(records), samples_per_rank=len(sampler), total_optimizer_steps=total)
    saved = None
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / "checkpoint.json").exists():
            raise ValueError("resume requires an aligned checkpoint; the old 2-layer/rank-64 smoke model is incompatible")
        saved = torch.load(checkpoint / f"training_state_rank_{rank():05d}.pt", map_location="cpu", weights_only=False)
        if saved["schema"] != "vlm_upstream_sft_v1" or saved["config"] != cfg:
            raise ValueError("resume configuration/data/world-size differs")
    torch.manual_seed(cfg["seed"])
    draft = make_draft(target, processor, cfg)
    draft.config._attn_implementation = "sdpa" if device == "cpu" else "flex_attention"
    draft.config.vlm_config["training"] = "upstream_sft_v1"
    if saved is not None:
        from litespec.modeling.draft.dlite import DLiteDraftModel
        loaded = DLiteDraftModel.from_pretrained(args.resume, attn_implementation="flex_attention")
        draft.load_state_dict(loaded.state_dict())
        del loaded
    online = make_online(target, draft, cfg)
    # Match upstream registration names and ownership before FSDP changes shapes.
    parameters, specs = register_optimizer_parameters(draft)
    fsdp = online
    layouts = None
    if device != "cpu":
        fsdp = FSDP(online, ignored_modules=[target.get_output_embeddings(), target.get_input_embeddings()],
                    use_orig_params=True, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
                    device_id=torch.cuda.current_device())
        layouts = build_fsdp_layouts(fsdp, parameters, specs)
    training_args = SimpleNamespace(**cfg)
    # Logging each optimizer boundary permits per-epoch exports without changing
    # the original runner's cross-epoch accumulation/tail semantics.
    if training_args.log_interval != 1:
        raise ValueError("aligned VLM epoch exports require log_interval=1 (operational logging only)")
    optimizer = BF16Optimizer(draft, cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                              max_grad_norm=cfg["max_grad_norm"], total_steps=total,
                              warmup_ratio=cfg["warmup_ratio"], parameters=parameters,
                              muon_config=muon_config_from_args(training_args) if cfg["use_muon"] else None,
                              parameter_specs=specs, parameter_layouts=layouts)
    session = TrainingSession(training_args, VLMAdapter(), [phase], draft=draft, draft_online=online,
                              fsdp_model=fsdp, target=target, optimizer=optimizer,
                              steps_per_epoch=len(sampler), total_steps=total)
    session.vlm_config, session.records, session.processor = cfg, records, processor
    session.sampler, session.device, session.base_dir = sampler, device, path.parent
    session.last_exported_epoch = 0
    session.tracker = SimpleNamespace(log=lambda *a, **kw: None, close=lambda: None)
    if cfg["report_to"] != "none":
        from litespec.training.logging import TrackerArgs, create_tracker, get_tracker_class
        if not dist.is_initialized():
            raise ValueError("external tracking requires a torchrun process group")
        for key, value in asdict(TrackerArgs()).items():
            if not hasattr(training_args, key):
                setattr(training_args, key, value)
        tracker_class = get_tracker_class(cfg["report_to"])
        if tracker_class is None:
            raise ValueError("unknown report_to tracker")
        tracker_class.validate_args(parser, training_args)
        session.tracker = create_tracker(training_args, cfg["output_dir"])
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        session.state = RunState(**saved["cursor"])
        session.last_exported_epoch = session.state.global_step // len(sampler)
        restore_rng(saved["rng"])
    else:
        random.seed(cfg["seed"] + rank())
        torch.manual_seed(cfg["seed"] + rank())
    optimizer.zero_grad()
    if rank() == 0:
        (root / "effective_config.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps(dict(config=cfg, global_batch=world() * cfg["accumulation_steps"],
                              full_epochs=cfg["num_epochs"], debug_cap=args.max_steps)), flush=True)
    try:
        run_draft(session, batch_source=source_batches, save=save, log=log)
    finally:
        session.tracker.close()


def main():
    configure_rank_caches()
    try:
        run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
