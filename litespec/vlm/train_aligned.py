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
from .progress import StepProgress


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def load_config(path):
    source = yaml.safe_load(Path(path).read_text())
    root = Path(__file__).resolve().parents[2]
    reference = root / source.pop("upstream_config")
    upstream = yaml.safe_load(reference.read_text())
    family = source.get("algorithm", "dlite")
    if family not in {"dlite", "dflash", "dspark"}:
        raise ValueError("unsupported VLM algorithm")
    if family == "dlite" and upstream.get("mode") != "sft":
        raise ValueError("this VLM adapter implements the upstream direct-SFT recipe")
    cfg = {**upstream["arguments"], **source}
    if not isinstance(cfg.get("batch_size"), int) or isinstance(cfg["batch_size"], bool) or cfg["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if cfg.get("model_role", "student") != "student":
        raise ValueError("VLM adapter requires the no-history student")
    if cfg["num_epochs"] < 1 or cfg["accumulation_steps"] < 1 or cfg["num_anchors"] < 1:
        raise ValueError("epochs, accumulation and anchors must be positive")
    cfg["upstream_config_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
    cfg["context_policy"] = dict(prefill_text="drop", prefill_vision="drop", decode_mode="none",
                                 window_size=0, implementation="compact")
    if family != "dlite":
        if upstream.get("algorithm") != family:
            raise ValueError("baseline recipe algorithm does not match")
        cfg["context_policy"].update(prefill_text="keep", prefill_vision="keep", decode_mode="all")
    if isinstance(cfg.get("target_layer_ids"), str):
        cfg["target_layer_ids"] = [int(x) for x in cfg["target_layer_ids"].split(",")]
    return cfg


def loss_mask(example):
    if hasattr(example, "supervision_mask"):
        return example.supervision_mask
    mask = torch.zeros_like(example.inputs["input_ids"], dtype=torch.float32)
    mask[:, example.prefill_length:] = 1
    return mask


def collate_examples(examples, pad_token_id=0):
    """Right-pad text; preserve per-image patch/grid order and answer boundaries."""
    if not examples:
        raise ValueError("cannot collate an empty VLM batch")
    if len(examples) == 1:
        return examples[0]
    width = max(ex.inputs["input_ids"].shape[1] for ex in examples)
    first = examples[0].inputs["input_ids"]
    ids = first.new_full((len(examples), width), pad_token_id)
    attention = torch.zeros_like(ids)
    supervision = torch.zeros_like(ids, dtype=torch.float32)
    token_types = torch.zeros_like(ids)
    for index, ex in enumerate(examples):
        size = ex.inputs["input_ids"].shape[1]
        ids[index, :size] = ex.inputs["input_ids"][0]
        attention[index, :size] = ex.inputs.get("attention_mask", torch.ones_like(ex.inputs["input_ids"]))[0]
        supervision[index, :size] = loss_mask(ex)[0] * attention[index, :size]
        if "mm_token_type_ids" in ex.inputs:
            token_types[index, :size] = ex.inputs["mm_token_type_ids"][0]
    inputs = dict(input_ids=ids, attention_mask=attention)
    if any("mm_token_type_ids" in ex.inputs for ex in examples):
        inputs["mm_token_type_ids"] = token_types
    for key in ("pixel_values", "image_grid_thw"):
        values = [ex.inputs[key] for ex in examples if key in ex.inputs]
        if values:
            inputs[key] = torch.cat(values, dim=0)
    # Let Qwen3.5 compute batched MRoPE from the image grids and padding mask.
    supported = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
    unexpected = set().union(*(ex.inputs.keys() for ex in examples)) - supported
    if unexpected:
        raise ValueError(f"unsupported batched processor inputs: {sorted(unexpected)}")
    return SimpleNamespace(inputs=inputs, supervision_mask=supervision,
                           sample_ids=[ex.sample_id for ex in examples])


def has_anchor(example, block_size):
    # Match setup._has_valid_anchor_supervision AND the upstream sampler.
    mask = loss_mask(example)
    if int(mask.sum().item()) < 2 * block_size:
        return False
    end = max(mask.size(1) - block_size, 0)
    return bool(((mask[:, :end + 1] > .5) & (mask[:, 1:end + 2] > .5)).any())


def reuse_validation(records, cfg, directory):
    """Recover the exact eligible set from a completed old preflight, not weights."""
    directory = Path(directory).expanduser().resolve()
    previous = json.loads((directory / "effective_config.json").read_text())
    for key in ("data_sha256", "train_data", "target_model", "max_length", "block_size", "debug_max_samples", "image_max_tokens"):
        if previous.get(key) != cfg.get(key):
            raise ValueError(f"validation reuse mismatch: {key}; run preflight again")
    skipped, excluded = [], set()
    with (directory / "skipped_training.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = row["index"]
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(records):
                raise ValueError("invalid index in saved validation")
            if index in excluded or row.get("id") != records[index]["id"]:
                raise ValueError("duplicate index or sample-id mismatch in saved validation")
            excluded.add(index)
            skipped.append(row)
    valid = [index for index in range(len(records)) if index not in excluded]
    digest = hashlib.sha256(json.dumps(valid).encode()).hexdigest()
    if len(valid) != previous.get("eligible_samples") or digest != previous.get("eligible_indices_sha256"):
        raise ValueError("saved validation is incomplete or its eligible-set hash differs")
    print(f"Reusing validation from {directory}: {len(valid)} eligible, {len(skipped)} skipped; no image re-encoding", flush=True)
    return dict(valid=valid, skipped=skipped)


def preflight(records, processor, config, cfg, base_dir, reuse_from=None):
    """Rank zero fixes the eligible set before equal-size distributed sampling."""
    packet = None
    if rank() == 0:
        try:
            if reuse_from is not None:
                packet = reuse_validation(records, cfg, reuse_from)
            else:
                packet = validate_records(records, processor, config, cfg, base_dir)
        except Exception as exc:
            packet = {"error": repr(exc)}
    if dist.is_initialized():
        payload = [packet]
        dist.broadcast_object_list(payload, src=0)
        packet = payload[0]
    if "error" in packet:
        raise RuntimeError(f"VLM input validation failed: {packet['error']}")
    return packet


def validate_records(records, processor, config, cfg, base_dir):
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
    return {"valid": valid, "skipped": skipped}


class CPUReferenceDLite(OnlineDLiteModel):
    """Numerical tests only: independent student blocks as an SDPA batch.

    Production CUDA uses the unchanged upstream FlexAttention implementation.
    """
    def forward_backbone(self, batch, *, seq_len, gdn_states=None):
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
                   attention_mask=None, is_causal=False,
                   gdn_states=tuple(s.reshape(bsz * blocks, 1, *s.shape[2:]) for s in gdn_states)
                   if gdn_states is not None else None)
        return d._prediction_hidden(output).reshape(bsz, blocks, d.proposal_length, -1)


def make_online(target, draft, cfg):
    if cfg.get("algorithm", "dlite") != "dlite":
        from .baselines import make_online as baseline_online
        return baseline_online(target, draft, cfg)
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
    from .baselines import is_baseline, prepare_forward as baseline_forward
    if is_baseline(online.draft_model):
        return baseline_forward(target, online, example, mask)
    anchor_mask = mask
    if hasattr(example, "supervision_mask"):
        lengths = example.inputs["attention_mask"].sum(-1)
        positions_in_sequence = torch.arange(ids.size(1), device=ids.device)
        anchor_mask = mask * (positions_in_sequence[None] <= lengths[:, None] - online.block_size + 1)
    anchors, keep = online.sample_anchor_positions(ids.size(1), anchor_mask)
    offsets = torch.arange(online.block_size - 1, device=ids.device)
    positions = anchors.unsqueeze(-1) + offsets
    # Project only distinct supervised text positions, then gather the block layout.
    unique, inverse = positions.reshape(-1).unique(sorted=True, return_inverse=True)
    from .gdn_capture import GDNAnchorCapture
    from contextlib import nullcontext
    reader = getattr(online.draft_model.config, "gdn_reader_config", None)
    capture = GDNAnchorCapture(target, reader["layer_ids"], anchors, keep) if reader else None
    with torch.no_grad(), capture if capture else nullcontext():
        output = target(**example.inputs, use_cache=False, output_hidden_states=True,
                        logits_to_keep=unique)
        indices = inverse.reshape(ids.size(0), -1)
        reference = output.logits.gather(1, indices.unsqueeze(-1).expand(-1, -1, output.logits.size(-1)))
        reference = reference.reshape(*positions.shape, -1)
        batch = online.prepare_batch(ids, output.hidden_states, mask,
                                     anchor_positions=anchors, block_keep_mask=keep)
        gdn_states = capture.states if capture else None
    return dict(gdn_states=gdn_states, prepared_batch=batch, seq_len=ids.size(1), target_prefill_logits=reference,
                target_logits_are_gathered=True)


class VLMAdapter:
    algorithm = "dlite_vlm"
    modes = ()

    def enter_phase(self, session, phase):
        session.draft.train()

    def compute_step(self, session, example, *, microbatch_start=None, microbatch_end=None,
                     collect_detailed_metrics=True):
        inputs = prepare_forward(session.target, session.draft_online, example)
        states = inputs.get("gdn_states") or ()
        session.gdn_anchor_bytes = sum(s.numel() * s.element_size() for s in states)
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
    """Atomic inference export, with optional rank-local resumable state."""
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
    if session.save_training_state:
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
        (staging / "checkpoint.json").write_text(json.dumps(dict(world_size=world(), cursor=asdict(session.state),
                                                               resumable=session.save_training_state)))
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
        items = list(session.sampler)
        batch_size = session.vlm_config["batch_size"]
        for index in range(len(items) // batch_size):
            if index < start:
                continue
            if session.state.phase_step >= phase.batch_count(session.steps_per_epoch):
                return
            examples = [prepare_example(session.processor, session.target.config, session.records[item],
                                        device=session.device, training=True, max_length=session.vlm_config["max_length"],
                                        base_dir=session.base_dir)
                        for item in items[index*batch_size:(index+1)*batch_size]]
            pad = getattr(getattr(session.processor, "tokenizer", None), "pad_token_id", None)
            if pad is None:
                pad = getattr(getattr(session.target.config, "text_config", session.target.config), "pad_token_id", 0)
            yield collate_examples(examples, pad_token_id=pad if pad is not None else 0)


def log(session, window, grad_norm):
    metrics = window.metrics.reduce()
    progress = session.progress.update(session.state.optimizer_step)
    scalars = {k: float(v) for k, v in metrics.items()}
    scalars.update({f"learning_rate/{k}": float(v)
                    for k, v in session.optimizer.get_learning_rates().items()})
    scalars["memory/gdn_anchor_mib"] = getattr(session, "gdn_anchor_bytes", 0) / 2**20
    if session.device != "cpu":
        scalars["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    scalars.update(grad_norm=float(grad_norm),
                   epoch_progress=session.state.global_step / session.steps_per_epoch)
    scalars.update({f"progress/{k}": v for k, v in progress.items() if isinstance(v, (int, float))})
    session.tracker.log(scalars, step=session.state.optimizer_step)
    completed = session.state.global_step // session.steps_per_epoch
    if rank() == 0:
        print(json.dumps(dict(optimizer_step=session.state.optimizer_step,
                              micro_step=session.state.global_step, epoch=session.state.epoch,
                              completed_epochs=completed, loss=float(metrics["objective/loss"]),
                              **progress,
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
    parser.add_argument("--reuse-validation-from", type=Path,
                        help="reuse effective_config.json + skipped_training.jsonl from a compatible run; images/processor must be unchanged")
    parser.add_argument("--save-training-state", action="store_true",
                        help="also save per-rank optimizer/RNG state for resume; default exports inference weights only")
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
    from .runtime import configure_image_token_budget
    configure_image_token_budget(processor, target.config, cfg.get("image_max_tokens"))
    path = Path(cfg["train_data"])
    records = list(read_records(path, args.max_samples or None))
    cfg["data_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    eligible = preflight(records, processor, target.config, cfg, path.parent,
                         reuse_from=args.reuse_validation_from)
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
    batches_per_rank = len(sampler) // cfg["batch_size"]
    if batches_per_rank < 1:
        raise ValueError("not enough eligible samples for one full batch per rank")
    total = optimizer_step_count([phase], batches_per_rank, cfg["accumulation_steps"])
    cfg.update(eligible_samples=len(records), samples_per_rank=len(sampler), total_optimizer_steps=total)
    saved = None
    if args.resume:
        checkpoint = Path(args.resume)
        if not (checkpoint / "checkpoint.json").exists():
            raise ValueError("resume requires an aligned checkpoint; the old 2-layer/rank-64 smoke model is incompatible")
        state_path = checkpoint / f"training_state_rank_{rank():05d}.pt"
        if not state_path.is_file():
            raise ValueError("checkpoint contains inference weights only; resume requires a checkpoint saved with --save-training-state")
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["schema"] != "vlm_upstream_sft_v1" or saved["config"] != cfg:
            raise ValueError("resume configuration/data/world-size differs")
    torch.manual_seed(cfg["seed"])
    if cfg.get("algorithm", "dlite") == "dlite":
        draft = make_draft(target, processor, cfg)
    else:
        from .baselines import make_draft as make_baseline
        draft = make_baseline(target, processor, cfg)
    draft.config._attn_implementation = "sdpa" if device == "cpu" else "flex_attention"
    draft.config.vlm_config["training"] = "upstream_sft_v1"
    if saved is not None:
        from .baselines import load_draft
        loaded = load_draft(args.resume, attn_implementation="sdpa" if device == "cpu" else "flex_attention")
        draft.load_state_dict(loaded.state_dict())
        del loaded
    online = make_online(target, draft, cfg)
    # Target activations already have the draft dtype. Disable FSDP root input
    # casts below so the detached FP32 GDN snapshots retain their precision.
    # Match upstream registration names and ownership before FSDP changes shapes.
    parameters, specs = register_optimizer_parameters(draft)
    fsdp = online
    layouts = None
    if device != "cpu":
        fsdp = FSDP(online, ignored_modules=[target.get_output_embeddings(), target.get_input_embeddings()],
                    use_orig_params=True, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16,
                                                   cast_root_forward_inputs=False),
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
                              steps_per_epoch=batches_per_rank, total_steps=total)
    session.vlm_config, session.records, session.processor = cfg, records, processor
    session.sampler, session.device, session.base_dir = sampler, device, path.parent
    session.last_exported_epoch = 0
    session.save_training_state = args.save_training_state
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
        session.last_exported_epoch = session.state.global_step // batches_per_rank
        restore_rng(saved["rng"])
    else:
        random.seed(cfg["seed"] + rank())
        torch.manual_seed(cfg["seed"] + rank())
    optimizer.zero_grad()
    if rank() == 0:
        (root / "effective_config.json").write_text(json.dumps(cfg, indent=2))
        print(json.dumps(dict(config=cfg, global_batch=world() * cfg["batch_size"] * cfg["accumulation_steps"],
                              full_epochs=cfg["num_epochs"], debug_cap=args.max_steps)), flush=True)
    session.progress = StepProgress(min(total, args.max_steps) if args.max_steps else total,
                                    session.state.optimizer_step)
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
