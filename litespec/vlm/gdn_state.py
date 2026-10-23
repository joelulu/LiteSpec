"""Experimental Qwen3.5 HF speculative state transactions (no weight changes).

``per_token`` saves every recurrent state during a fused recurrent verify.
``compact_replay`` saves the block-start state and post-conv k/v/g/beta; commit
replays only the accepted GDN updates. This is a ReplaySSM-inspired per-block
transaction, NOT the upstream circular-buffer/output-only ReplaySSM algorithm.

Both modes use the SAME verification recurrence. Native HF prefill is untouched.
CUDA uses our Triton kernel; CPU uses a correctness reference, never a speed path.
"""
from contextlib import contextmanager
import importlib
from types import MethodType

import torch

STATE_MODES = ("full_target_replay", "per_token", "compact_replay")


@torch.no_grad()
def warmup_state_kernels(target, *, batch_size, width, mode):
    """Compile all tail widths BEFORE timing; prefix length is runtime-valued."""
    if mode == "full_target_replay":
        return
    hf = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    seen = set()
    for layer in target.modules():
        if not isinstance(layer, hf.Qwen3_5GatedDeltaNet):
            continue
        weight = layer.in_proj_qkv.weight
        if weight.device.type != "cuda":
            continue
        shape = (layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, weight.dtype, weight.device)
        if shape in seen:
            continue
        seen.add(shape)
        heads, dk, dv, dtype, device = shape
        initial = torch.zeros(batch_size, heads, dk, dv, device=device, dtype=torch.float32)
        for length in range(1, width+1):
            k = torch.zeros(batch_size, length, heads, dk, device=device, dtype=dtype)
            v = torch.zeros(batch_size, length, heads, dv, device=device, dtype=dtype)
            g = torch.zeros(batch_size, length, heads, device=device, dtype=torch.float32)
            beta = torch.zeros(batch_size, length, heads, device=device, dtype=dtype)
            recurrence(k, k, v, g, beta, initial, save_states=mode == "per_token")
            if mode == "compact_replay":
                recurrence(None, k, v, g, beta, initial, out_state=initial, steps=length)
        torch.cuda.synchronize(device)


def reference_recurrence(q, k, v, g, beta, initial, *, save_states=False):
    """FP32 HF-layout [B,H,K,V] recurrence, including HF L2 epsilon."""
    k = k.float()
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    if q is not None:
        q = q.float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / q.shape[-1]**0.5
    state = initial.float().clone()
    outputs, states = [], []
    for i in range(k.shape[1]):
        state = state * g[:, i].float().exp()[..., None, None]
        residual = (v[:, i].float() - (state * k[:, i, :, :, None]).sum(-2)) * beta[:, i].float()[..., None]
        state = state + k[:, i, :, :, None] * residual[..., None, :]
        if q is not None:
            outputs.append((state * q[:, i, :, :, None]).sum(-2))
        if save_states:
            states.append(state.clone())
    output = torch.stack(outputs, 1).to(v.dtype) if outputs else None
    return output, state, torch.stack(states, 1) if save_states else None


def recurrence(q, k, v, g, beta, initial, *, save_states=False, out_state=None, steps=None):
    if steps is not None and (q is not None or save_states or not 1 <= steps <= k.shape[1]):
        raise ValueError("prefix steps are supported only for state-only replay")
    if k.device.type == "cuda":
        from .gdn_state_triton import run_recurrence
        return run_recurrence(q, k, v, g, beta, initial, save_states=save_states, out_state=out_state, steps=steps)
    if k.device.type != "cpu":
        raise ValueError("GDN state experiment supports CPU reference or CUDA Triton only")
    if steps is not None:
        k, v, g, beta = (x[:, :steps] for x in (k, v, g, beta))
    output, state, states = reference_recurrence(q, k, v, g, beta, initial, save_states=save_states)
    if out_state is not None:
        out_state.copy_(state)
        state = out_state
    return output, state, states


class GDNStateTransaction:
    """Instance-local adapter; restored on exit, including exceptions.

    For unpadded linear-chain HF static batches only. No global kernel patch,
    no training changes, no serving-engine or variable-length batching claim.
    """
    def __init__(self, target, mode):
        if mode not in {"per_token", "compact_replay"}:
            raise ValueError(f"unsupported state transaction {mode}")
        self.target, self.mode = target, mode
        self.active, self.records, self.installed = False, {}, []
        self.peak_record_bytes = 0
        self.module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")

    def __enter__(self):
        modules = [m for m in self.target.modules() if isinstance(m, self.module.Qwen3_5GatedDeltaNet)]
        if not modules:
            raise RuntimeError("state modes require native Transformers Qwen3.5 GDN modules")
        for module in modules:
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "layer_idx"):
                if not hasattr(module, name):
                    raise RuntimeError(f"unsupported Qwen3.5 HF GDN ABI: missing {name}")
        try:
            for module in modules:
                # Preserve whether forward was originally an instance attribute.
                had_local, local = "forward" in module.__dict__, module.__dict__.get("forward")
                original = module.forward
                def adapted(layer, hidden_states, cache_params=None, attention_mask=None, _original=original, **kwargs):
                    if not self.active:
                        return _original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
                    return self._forward(layer, hidden_states, cache_params, attention_mask, kwargs)
                module.forward = MethodType(adapted, module)
                self.installed.append((module, had_local, local))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        self.active = False
        self.records.clear()
        for module, had_local, local in reversed(self.installed):
            if had_local:
                module.forward = local
            else:
                del module.forward
        self.installed.clear()

    @contextmanager
    def capture(self, cache):
        if self.active or self.records:
            raise RuntimeError("unfinished GDN state transaction")
        if not hasattr(cache, "activate_past_recording"):
            raise RuntimeError("state experiment requires HF 5.x recordable hybrid cache")
        cache.activate_past_recording()
        self.active = True
        try:
            yield
            expected = {m.layer_idx for m, _, _ in self.installed}
            if set(self.records) != expected:
                raise RuntimeError("not every GDN layer participated in state capture")
            self.peak_record_bytes = max(self.peak_record_bytes, self.record_bytes())
        finally:
            self.active = False

    def record_bytes(self):
        # Retained transaction payload, not process/CUDA allocator peak memory.
        return sum(t.numel()*t.element_size() for record in self.records.values()
                   for t in record.values() if torch.is_tensor(t))

    def _forward(self, layer, hidden, cache, mask, kwargs):
        if cache is None or not cache.has_previous_state(layer.layer_idx, state_idx=0):
            raise RuntimeError("GDN capture requires a prefilled hybrid cache")
        if kwargs.get("cu_seq_lens_q") is not None or kwargs.get("cu_seqlens") is not None:
            raise ValueError("packed/variable-length GDN verify is not supported")
        if layer.layer_idx in self.records:
            raise RuntimeError("GDN layer called more than once in a verification transaction")
        hf = self.module
        hidden = hf.apply_mask_to_padding_states(hidden, mask)
        batch, length, _ = hidden.shape
        mixed = layer.in_proj_qkv(hidden).transpose(1, 2)
        z = layer.in_proj_z(hidden).reshape(batch, length, -1, layer.head_v_dim)
        b, a = layer.in_proj_b(hidden), layer.in_proj_a(hidden)
        # Same recorded-convolution branch as HF; crop later restores its prefix.
        mixed = cache.update_conv_state(mixed, layer.layer_idx, conv_kernel_size=layer.conv_kernel_size)
        if mixed.device.type == "cuda":
            # Call the installed CUDA extension explicitly: HF's class-local
            # Hub substitutions need not be present in module globals.
            try:
                from causal_conv1d import causal_conv1d_fn
            except ImportError as exc:
                raise RuntimeError("CUDA state experiment requires causal-conv1d; no silent Python fallback") from exc
            mixed = causal_conv1d_fn(mixed, layer.conv1d.weight.squeeze(1), layer.conv1d.bias,
                                    activation=layer.activation)
        else:
            mixed = hf.causal_conv1d_fn(mixed, layer.conv1d.weight.squeeze(1), layer.conv1d.bias,
                                      activation=layer.activation, **kwargs)
        mixed = mixed[:, :, -length:].transpose(1, 2)
        q, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], -1)
        q, k = (x.reshape(batch, length, -1, layer.head_k_dim) for x in (q, k))
        v = v.reshape(batch, length, -1, layer.head_v_dim)
        beta = b.sigmoid()
        g = -layer.A_log.float().exp() * torch.nn.functional.softplus(a.float() + layer.dt_bias)
        repeats = layer.num_v_heads // layer.num_k_heads
        if repeats > 1:
            q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
        # Contiguous records have no views retaining the q|k|v backing storage.
        q, g, beta = (x.contiguous() for x in (q, g, beta))
        k, v = (x.clone(memory_format=torch.contiguous_format) if self.mode == "compact_replay"
                else x.contiguous() for x in (k, v))
        initial = cache.layers[layer.layer_idx].recurrent_states[0]
        record = {}
        if self.mode == "compact_replay":
            record = dict(initial=initial.clone(), k=k, v=v, g=g, beta=beta)
        core, final, states = recurrence(q, k, v, g, beta, initial, save_states=self.mode == "per_token")
        if states is not None:
            record = dict(states=states)
        self.records[layer.layer_idx] = record
        cache.update_recurrent_state(final, layer.layer_idx)
        core = layer.norm(core.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
        return layer.out_proj(core.reshape(batch, length, -1))

    def commit(self, cache, kept, width):
        if not 1 <= kept <= width or not self.records:
            raise ValueError("invalid or empty GDN commit")
        if kept < width:
            for index, record in self.records.items():
                destination = cache.layers[index].recurrent_states[0]
                if self.mode == "per_token":
                    destination.copy_(record["states"][:, kept-1])
                else:
                    recurrence(None, *(record[n] for n in ("k", "v", "g", "beta")),
                               record["initial"], out_state=destination, steps=kept)
        self.records.clear()
