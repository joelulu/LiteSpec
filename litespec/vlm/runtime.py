"""Single-request HF multimodal reference path (training-independent).

This path prioritizes correctness: hybrid verification snapshots the target
cache and replays the committed prefix on rejection. All snapshot/replay cost
is included in total latency. It is not a continuous-batching serving engine.
Keep this file in sync with litespec/vlm/runtime.py.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .context import ContextPolicy, visual_positions


def clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def load_target(path, *, device="cuda", dtype=torch.bfloat16, attention="sdpa"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    target = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, attn_implementation=attention
    ).to(device).eval()
    target.requires_grad_(False)
    return target, AutoProcessor.from_pretrained(path)


@dataclass
class PreparedExample:
    inputs: dict
    prefill_length: int
    vision_mask: torch.Tensor
    sample_id: str


@dataclass
class HistoryBank:
    hidden: torch.Tensor
    positions: torch.Tensor

    def prune(self, policy, *, anchor, prefill_length, vision_mask):
        if policy.implementation != "compact":
            return
        # Keep the pivot for future rounds, and keep temporarily hidden recent
        # tokens in drop_recent mode: they can become eligible as decoding moves.
        positions = self.positions
        visual = (positions < prefill_length) & vision_mask[positions.clamp(0, prefill_length-1)]
        keep = ((positions < prefill_length) & ~visual & (policy.prefill_text == "keep")) | (
            visual & (policy.prefill_vision == "keep"))
        decode = positions >= prefill_length
        if policy.decode_mode == "none":
            decode = torch.zeros_like(decode)
        elif policy.decode_mode == "keep_last":
            decode &= positions >= anchor - 1 - policy.window_size
        keep |= decode
        self.positions, self.hidden = positions[keep], self.hidden[:, keep]


def prepare_example(processor, target_config, record, *, device, training=False,
                    max_length=None, base_dir=None):
    """Explicit schema: {id, image?: local path, question, answer?}.

    One image/one user turn only. No heuristic field extraction, automatic
    dataset conversion, truncation of image blocks, or network image fetches.
    """
    from PIL import Image

    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ValueError("each record needs a non-empty question")
    content = []
    if record.get("image"):
        path = Path(record["image"])
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        with Image.open(path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    content.append({"type": "text", "text": record["question"]})
    messages = [{"role": "user", "content": content}]

    def encode(turns, generation):
        return processor.apply_chat_template(
            turns, tokenize=True, add_generation_prompt=generation,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        )

    prefix = encode(messages, True)
    prefill_length = prefix["input_ids"].shape[1]
    encoded = prefix
    if training:
        if not isinstance(record.get("answer"), str) or not record["answer"].strip():
            raise ValueError("training records need a non-empty target-generated answer")
        encoded = encode(messages + [{"role": "assistant", "content": record["answer"]}], False)
        if not torch.equal(encoded["input_ids"][:, :prefill_length], prefix["input_ids"]):
            raise ValueError("chat template is not prefix-stable; cannot align answer supervision")
    length = encoded["input_ids"].shape[1]
    if max_length is not None and length > max_length:
        raise ValueError(f"sample {record.get('id', '?')} has {length} tokens > max_length={max_length}; not truncating visual blocks")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    vision = visual_positions(inputs["input_ids"][0, :prefill_length], target_config)
    return PreparedExample(inputs, prefill_length, vision, str(record.get("id", "")))


def selected_layers(states, ids, depth):
    if len(states) not in {depth, depth + 1}:
        raise ValueError("target did not return language-layer hidden states")
    offset = int(len(states) == depth + 1)
    return torch.stack([states[i + offset] for i in ids], dim=2)


def draft_block(draft, embedding, *, anchor_ids, pivot, history, anchor,
                prefill_length, vision_mask, policy):
    """One block, identical context selection during training and decoding."""
    student = draft.is_student
    if student and not policy.is_empty:
        raise ValueError("student checkpoints have no history interface; use dlite_no_history")
    device, dtype = pivot.device, pivot.dtype
    block_size = draft.block_size
    query_ids = torch.full((1, block_size), draft.mask_token_id, device=device, dtype=torch.long)
    query_ids[:, 0] = anchor_ids.reshape(1)
    query = draft.build_inference_query_embeddings(embedding, query_ids)
    physical = visible = 0
    if student:
        context_pos = torch.zeros((1, draft.condition_slot_count), device=device, dtype=torch.long)
        offset = 0 if draft.architecture_version == "dlite_v1" else 1
        q_pos = torch.arange(offset, offset + block_size, device=device).view(1, -1)
        kwargs = {"history_hidden": pivot.new_empty(1, 1, 0, pivot.size(-1))}
        attention_mask = None
    else:
        if history is None:
            raise ValueError("teacher requires a history bank")
        if isinstance(history, HistoryBank):
            available = history.positions < anchor - 1
            positions, raw = history.positions[available], history.hidden[:, available]
        else:
            if history.shape[1] < anchor:
                raise ValueError("training history must cover the committed pivot")
            positions, raw = torch.arange(anchor - 1, device=device), history[:, :anchor - 1]
        keep = policy.keep(positions, anchor=anchor, prefill_length=prefill_length,
                           vision_mask=vision_mask)
        visible = int(keep.sum().item())
        if policy.implementation == "compact":
            raw = raw[:, keep]
            positions = positions[keep]
            keep = torch.ones_like(positions, dtype=torch.bool)
        # Selection precedes trainable fusion and per-layer K/V projection.
        fused = draft.fuse_history_hidden(raw)
        physical = positions.numel()
        kwargs = {"history_hidden": fused.unsqueeze(1)}
        chs_pos = torch.full((draft.condition_slot_count,), anchor - 1, device=device)
        context_pos = torch.cat([positions, chs_pos]).long().view(1, -1)
        q_pos = torch.arange(anchor, anchor + block_size, device=device).view(1, -1)
        allow = torch.cat([keep, torch.ones(draft.condition_slot_count + block_size,
                                            device=device, dtype=torch.bool)])
        attention_mask = torch.zeros((1, 1, block_size, allow.numel()), device=device, dtype=dtype)
        attention_mask.masked_fill_(~allow.view(1, 1, 1, -1), torch.finfo(dtype).min)
    output = draft(
        noise_embedding=query, target_hidden=pivot, position_ids=q_pos,
        rotary_position_ids=torch.cat([context_pos, q_pos], dim=-1),
        attention_mask=attention_mask, is_causal=False, **kwargs,
    )
    return draft._prediction_hidden(output), {"physical_history": physical, "visible_history": visible}


@torch.no_grad()
def generate(target, draft, example, *, max_new_tokens=128, policy=None, verification="block"):
    """Greedy chain verification with exact accepted-prefix cache recovery."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if verification not in {"block", "sequential"}:
        raise ValueError("verification must be block or sequential")
    inputs = example.inputs
    if inputs["input_ids"].shape != (1, example.prefill_length):
        raise ValueError("generation accepts one unpadded prompt, not a training answer")
    from .baselines import is_baseline, DraftCache
    baseline = draft is not None and is_baseline(draft)
    policy = policy or (ContextPolicy() if baseline else ContextPolicy.no_history())
    if baseline and (policy.prefill_text != "keep" or policy.prefill_vision != "keep" or policy.decode_mode != "all"):
        raise ValueError("baseline cached inference currently requires full history")
    if draft is not None and not baseline and draft.is_student and not policy.is_empty:
        raise ValueError("non-empty history policy is incompatible with a student checkpoint")
    device = inputs["input_ids"].device
    started = clock(device)
    output = target(**inputs, use_cache=True,
                    output_hidden_states=draft is not None or verification == "sequential")
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("target must return a cache for incremental verification")
    prefill_seconds = clock(device) - started
    pending = output.logits[:, -1:].argmax(-1)
    anchor = example.prefill_length
    history = None
    pivot = None
    baseline_cache = DraftCache(draft, target, output, anchor) if baseline else None
    if draft is not None:
        pivot = selected_layers(output.hidden_states, draft.target_layer_ids,
                                draft.config.num_target_layers)[:, -1:].detach()
        if not baseline and not draft.is_student:
            history = HistoryBank(
                selected_layers(output.hidden_states, draft.history_layer_ids,
                                draft.config.num_target_layers).detach(),
                torch.arange(anchor, device=device),
            )
            history.prune(policy, anchor=anchor, prefill_length=example.prefill_length,
                          vision_mask=example.vision_mask)
    eos = getattr(target.config, "text_config", target.config).eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    emitted, rounds, context_counts = [], [], []
    draft_seconds = verify_seconds = recovery_seconds = 0.0
    decode_started = clock(device)
    while len(emitted) < max_new_tokens:
        if draft is None:
            token = int(pending.item())
            emitted.append(token)
            if token in eos or len(emitted) == max_new_tokens:
                break
            output = target(input_ids=pending, past_key_values=cache, use_cache=True,
                            output_hidden_states=verification == "sequential")
            cache = output.past_key_values
            pending = output.logits[:, -1:].argmax(-1)
            continue
        t = clock(device)
        if baseline:
            proposals, counts = baseline_cache.propose(draft, target, pending, anchor)
        else:
            hidden, counts = draft_block(
                draft, target.get_input_embeddings(), anchor_ids=pending, pivot=pivot,
                history=history, anchor=anchor, prefill_length=example.prefill_length,
                vision_mask=example.vision_mask, policy=policy,
            )
            proposals, _ = draft.sample_draft_tokens(
                draft_hidden=hidden, first_prev_token_ids=pending[:, 0], temperature=0.0,
            )
        candidates = torch.cat([pending, proposals], dim=1)
        candidates = candidates[:, :max_new_tokens - len(emitted)]
        draft_seconds += clock(device) - t
        t = clock(device)
        # Snapshot includes GDN convolution/recurrent state AND attention KV.
        before = None
        if verification == "sequential":
            # Diagnostic path: identical one-token calls to AR. Wrong proposals
            # never enter the hybrid cache. This is not a speed benchmark.
            from types import SimpleNamespace
            states, predictions = [], []
            for j in range(candidates.shape[1]):
                one = target(input_ids=candidates[:, j:j+1], past_key_values=cache,
                             use_cache=True, output_hidden_states=True)
                cache = one.past_key_values
                states.append(one.hidden_states)
                predictions.append(one.logits[:, -1:].argmax(-1))
                if j + 1 == candidates.shape[1] or not torch.equal(candidates[:, j+1:j+2], predictions[-1]):
                    break
            kept = len(states)
            predictions = torch.cat(predictions, dim=1)
            output = SimpleNamespace(hidden_states=tuple(torch.cat(parts, dim=1) for parts in zip(*states)))
        else:
            before = copy.deepcopy(cache)
            output = target(input_ids=candidates, past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
            cache = output.past_key_values
            predictions = output.logits.argmax(-1)
            accepted = 0
            for j in range(candidates.shape[1] - 1):
                if int(candidates[0, j + 1]) != int(predictions[0, j]):
                    break
                accepted += 1
            kept = accepted + 1
        verify_seconds += clock(device) - t
        if verification == "block" and kept != candidates.shape[1]:
            t = clock(device)
            restored = target(input_ids=candidates[:, :kept], past_key_values=before,
                              use_cache=True, output_hidden_states=True)
            cache = restored.past_key_values
            # The committed prefix is the authoritative state after replay.
            # Chunk-vs-prefix execution may differ numerically on hybrid targets.
            output = restored
            predictions = restored.logits.argmax(-1)
            recovery_seconds += clock(device) - t
        del before
        pivot = selected_layers(output.hidden_states, draft.target_layer_ids,
                                draft.config.num_target_layers)[:, kept - 1:kept].detach()
        if baseline:
            baseline_cache.commit(draft, output, kept)
        elif not draft.is_student:
            new = selected_layers(output.hidden_states, draft.history_layer_ids,
                                  draft.config.num_target_layers)[:, :kept].detach()
            history.hidden = torch.cat([history.hidden, new], dim=1)
            history.positions = torch.cat([history.positions, torch.arange(anchor, anchor+kept, device=device)])
        pending = predictions[:, kept - 1:kept]
        anchor += kept
        if history is not None:
            history.prune(policy, anchor=anchor, prefill_length=example.prefill_length,
                          vision_mask=example.vision_mask)
        context_counts.append(counts)
        previous_count = len(emitted)
        for token in candidates[0, :kept].tolist():
            emitted.append(token)
            if token in eos:
                break
        rounds.append(len(emitted) - previous_count)
        if emitted[-1] in eos:
            break
    decode_seconds = clock(device) - decode_started
    return {
        "token_ids": emitted, "accept_lengths": rounds, "context_counts": context_counts,
        "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
        "total_seconds": clock(device) - started, "draft_seconds": draft_seconds,
        "verify_seconds": verify_seconds, "recovery_seconds": recovery_seconds,
        "policy": policy.to_dict(), "prefill_tokens": example.prefill_length,
        "visual_tokens_with_delimiters": int(example.vision_mask.sum().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "verification": verification,
        "cache_recovery": "full_snapshot_and_prefix_replay" if verification == "block" else "never_commit_rejected_tokens",
    }


def read_records(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                record = json.loads(line)
                record.setdefault("id", str(index))
                yield record
