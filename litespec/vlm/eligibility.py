"""Shared CPU-safe supervision checks for VLM preparation and training.

Keep this module independent of trainers, distributed runtimes and optional
attention packages: importing them can initialize CUDA on a CUDA-enabled wheel.
"""
import torch


def loss_mask(example):
    if hasattr(example, "supervision_mask"):
        return example.supervision_mask
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
