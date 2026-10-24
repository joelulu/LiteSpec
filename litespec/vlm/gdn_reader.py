"""ReaDy-style query-conditioned residual reader for native HF GDN states.

State input is a tuple (one tensor per selected target layer), each in HF
[B, anchors, heads, key, value] layout. The anchor axis is shared by all query
positions in that draft block; it never denotes future target states.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def reader_config(target_config, cfg):
    if not cfg.get("gdn_enabled", False):
        return None
    text = target_config.text_config
    layers = cfg.get("gdn_layer_ids", [14, 30])
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
    if not layers or len(layers) != len(set(layers)) or any(
        i < 0 or i >= text.num_hidden_layers or text.layer_types[i] != "linear_attention" for i in layers
    ):
        raise ValueError("gdn_layer_ids must select distinct Qwen3.5 linear-attention layers (zero based)")
    return dict(layer_ids=list(layers), heads=text.linear_num_value_heads,
                key_dim=text.linear_key_head_dim, value_dim=text.linear_value_head_dim,
                rank=int(cfg.get("gdn_rank", 8)), bottleneck=int(cfg.get("gdn_bottleneck", 128)),
                chunk_anchors=int(cfg.get("gdn_chunk_anchors", 16)),
                injection_layer=int(cfg.get("gdn_injection_layer", cfg["num_draft_layers"] - 1)))


class GDNStateReader(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.layer_ids = tuple(config["layer_ids"])
        self.heads, self.dk, self.dv = (config[k] for k in ("heads", "key_dim", "value_dim"))
        self.rank, self.chunk_anchors = config["rank"], config["chunk_anchors"]
        layers, heads, rank = len(self.layer_ids), self.heads, self.rank
        if min(layers, heads, self.dk, self.dv, rank, config["bottleneck"], self.chunk_anchors) < 1:
            raise ValueError("GDN reader dimensions must be positive")
        self.query = nn.Linear(hidden_size, layers * heads * rank, bias=False)
        self.query_up = nn.Parameter(torch.randn(layers, heads, rank, self.dk) / math.sqrt(rank))
        self.value_down = nn.Parameter(torch.randn(layers, heads, self.dv, rank) / math.sqrt(self.dv))
        self.down = nn.Linear(layers * heads * (rank + 1), config["bottleneck"], bias=False)
        self.up = nn.Linear(config["bottleneck"], hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def _read(self, hidden, *states):
        # hidden [B,A,Q,D]. Keep FP32 normalization, then use the draft dtype.
        b, a, n, d = hidden.shape
        query = self.query(F.rms_norm(hidden.float(), (d,)).to(hidden.dtype))
        query = query.reshape(b, a, n, len(states), self.heads, self.rank)
        keys = torch.einsum("banlhr,lhrk->banlhk", query, self.query_up)
        reads, scales = [], []
        for i, state in enumerate(states):
            sf = state.float()
            rms = (sf.square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
            # HF [K,V] -> ReaDy [V,K]; nonsquare state tests cover this transpose.
            normalized = (sf / rms).transpose(-1, -2).to(hidden.dtype)
            compressed = torch.einsum("hvr,bahvk->bahrk", self.value_down[i], normalized)
            reads.append(torch.einsum("bahrk,banhk->banhr", compressed, keys[:, :, :, i]) / math.sqrt(self.dk))
            scales.append(rms[..., 0, 0].log().to(hidden.dtype)[:, :, None, :, None].expand(b, a, n, self.heads, 1))
        features = torch.cat((torch.stack(reads, 3), torch.stack(scales, 3)), -1).flatten(3)
        features = F.rms_norm(features.float(), (features.shape[-1],)).to(hidden.dtype)
        return hidden + self.up(F.silu(self.down(features)))

    def forward(self, hidden, states):
        if states is None or len(states) != len(self.layer_ids):
            raise ValueError("enabled GDN reader requires one causally aligned state per selected layer")
        b, width, d = hidden.shape
        anchors = states[0].shape[1]
        expected = (b, anchors, self.heads, self.dk, self.dv)
        if anchors < 1 or width % anchors or any(tuple(s.shape) != expected for s in states):
            raise ValueError("GDN state/query block layout differs")
        query = hidden.reshape(b, anchors, width // anchors, d)
        outputs = []
        for start in range(0, anchors, self.chunk_anchors):
            chunk = query[:, start:start+self.chunk_anchors]
            values = tuple(s[:, start:start+self.chunk_anchors].detach() for s in states)
            if self.training and torch.is_grad_enabled():
                # Retain only the raw anchor states; recompute normalization and
                # small reader activations during backward, never the target.
                outputs.append(checkpoint(self._read, chunk, *values, use_reentrant=False))
            else:
                outputs.append(self._read(chunk, *values))
        return torch.cat(outputs, 1).reshape(b, width, d)


def install_reader(draft):
    config = getattr(draft.config, "gdn_reader_config", None)
    if config is None:
        return
    index = config["injection_layer"]
    if not 0 <= index < len(draft.layers):
        raise ValueError("invalid GDN reader injection layer")
    # Called AFTER the original model's post_init: paired backbones and the
    # subsequent RNG stream remain identical with the reader enabled/disabled.
    with torch.random.fork_rng(devices=[]):
        draft.layers[index].gdn_reader = GDNStateReader(draft.config.hidden_size, config)


def cache_reader_states(draft, cache):
    config = getattr(draft.config, "gdn_reader_config", None)
    if config is None:
        return None
    values = []
    for index in config["layer_ids"]:
        if not cache.has_previous_state(index, state_idx=0):
            raise RuntimeError(f"GDN layer {index} has no committed state")
        state = cache.layers[index].recurrent_states[0]
        # The reader is synchronous and the target cache is not mutated until
        # drafting has finished. No history cache or per-token state copies.
        values.append(state.detach().unsqueeze(1))
    return tuple(values)
