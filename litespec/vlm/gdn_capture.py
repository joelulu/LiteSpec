"""Sparse training anchor states without modifying the frozen target's output.

Hooks retain the native projections of selected GDN modules. A second causal
convolution and GDN-only recurrence reconstruct states at anchor-1, while the
native HF full-sequence forward still supplies all logits and hidden states.
No final-answer cache is used and no full target forward is replayed per anchor.
"""
import importlib
import torch


@torch.no_grad()
def reference_anchor_states(k, v, g, beta, anchors, keep):
    from .gdn_state import reference_recurrence
    b, _, h, dk = k.shape
    states = torch.zeros(b, anchors.shape[1], h, dk, v.shape[-1], device=k.device, dtype=torch.float32)
    state = states.new_zeros(b, h, dk, v.shape[-1])
    for t in range(k.shape[1]):
        _, state, _ = reference_recurrence(None, k[:, t:t+1], v[:, t:t+1], g[:, t:t+1], beta[:, t:t+1], state)
        rows, slots = ((anchors == t+1) & keep).nonzero(as_tuple=True)
        states[rows, slots] = state[rows]
    return states


@torch.no_grad()
def anchor_states(k, v, g, beta, anchors, keep):
    if k.device.type == "cuda":
        from .gdn_capture_triton import sparse_anchor_states
        return sparse_anchor_states(k, v, g, beta, anchors, keep)
    if k.device.type != "cpu":
        raise ValueError("GDN anchor capture supports CPU reference or CUDA Triton")
    return reference_anchor_states(k, v, g, beta, anchors, keep)


class GDNAnchorCapture:
    def __init__(self, target, layer_ids, anchors, keep):
        self.target, self.layer_ids, self.anchors, self.keep = target, tuple(layer_ids), anchors, keep
        if anchors.shape != keep.shape or anchors.ndim != 2:
            raise ValueError("invalid GDN anchor layout")
        if bool(((anchors < 1) & keep).any()):
            raise ValueError("GDN anchors must have a committed prefix")
        # Samplers sort valid anchors before masked filler slots.
        positions = torch.where(keep, anchors, torch.iinfo(anchors.dtype).max)
        if bool((positions[:, 1:] < positions[:, :-1]).any()):
            raise ValueError("GDN anchors must be sorted")
        self.handles, self.projections, self.records = [], {}, {}

    def __enter__(self):
        hf = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        selected = {m.layer_idx: m for m in self.target.modules()
                    if isinstance(m, hf.Qwen3_5GatedDeltaNet) and m.layer_idx in self.layer_ids}
        if set(selected) != set(self.layer_ids):
            raise ValueError("GDN reader requested nonexistent/native non-GDN layers")
        try:
            for index, layer in selected.items():
                self.projections[index] = {}
                for name in ("in_proj_qkv", "in_proj_a", "in_proj_b"):
                    def projection_hook(module, args, output, i=index, n=name):
                        if n in self.projections[i]:
                            raise RuntimeError("GDN training capture saw repeated projection")
                        self.projections[i][n] = output.detach()
                    self.handles.append(getattr(layer, name).register_forward_hook(projection_hook))
                self.handles.append(layer.register_forward_hook(self._collect, with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def _collect(self, layer, args, kwargs, output):
        if kwargs.get("cache_params") is not None or (len(args) > 1 and args[1] is not None):
            raise ValueError("training capture requires a fresh teacher-forcing forward, use_cache=False")
        if any(kwargs.get(k) is not None for k in ("cu_seq_lens_q", "cu_seqlens")):
            raise ValueError("packed GDN training capture is unsupported")
        p = self.projections[layer.layer_idx]
        mixed = p.pop("in_proj_qkv").transpose(1, 2)
        b, a = p.pop("in_proj_b"), p.pop("in_proj_a")
        if mixed.device.type == "cuda":
            from causal_conv1d import causal_conv1d_fn
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn
        mixed = causal_conv1d_fn(mixed, layer.conv1d.weight.squeeze(1), layer.conv1d.bias,
                                 activation=layer.activation).transpose(1, 2)
        batch, length, _ = mixed.shape
        if bool(((self.anchors > length) & self.keep).any()):
            raise ValueError("GDN anchor outside target sequence")
        _, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], -1)
        k = k.reshape(batch, length, layer.num_k_heads, layer.head_k_dim)
        v = v.reshape(batch, length, layer.num_v_heads, layer.head_v_dim)
        repeats = layer.num_v_heads // layer.num_k_heads
        k = k.repeat_interleave(repeats, dim=2) if repeats > 1 else k
        beta = b.sigmoid()
        g = -layer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer.dt_bias)
        self.records[layer.layer_idx] = anchor_states(*(x.contiguous() for x in (k, v, g, beta)), self.anchors, self.keep)

    @property
    def states(self):
        if set(self.records) != set(self.layer_ids):
            raise RuntimeError("incomplete GDN anchor capture")
        return tuple(self.records[i] for i in self.layer_ids)

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.projections.clear()
        # Returned tensors are owned by the caller, including after hook cleanup.
        self.records.clear()
