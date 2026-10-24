"""Fixed prompt evidence, read once per pivot (never generated answer history)."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PromptMemory:
    values: torch.Tensor
    valid: torch.Tensor


@dataclass
class ProjectedMemory:
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class PivotMemoryReader(nn.Module):
    def __init__(self, hidden_size, rank=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def project(self, memory):
        x = self.norm(memory.values)
        return ProjectedMemory(self.key(x), self.value(x), memory.valid)

    def forward(self, pivot, memory):
        # pivot [B,N,S,H], one read per anchor, shared across its query tokens.
        if memory is None:
            raise ValueError("memory checkpoint requires explicit prompt memory")
        bank = self.project(memory) if isinstance(memory, PromptMemory) else memory
        q = pivot.mean(-2)
        available = bank.valid.any(-1)
        # Empty modality: safe dummy slot, followed by an exact zero residual.
        valid = bank.valid.clone()
        valid[:, 0] |= ~available
        read = F.scaled_dot_product_attention(
            self.query(self.norm(q)).unsqueeze(1), bank.keys.unsqueeze(1),
            bank.values.unsqueeze(1), attn_mask=valid[:, None, None, :],
            dropout_p=0., is_causal=False).squeeze(1)
        return self.output(read) * self.gate(q).sigmoid() * available[:, None, None]
