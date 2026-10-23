"""Fused linear-chain GDN verify / state-only replay, native HF [B,H,K,V].

The recurrence follows Transformers Qwen3.5 and vLLM's GDN MTP update: FP32
state, L2-normalized q/k, decay then delta update. No Python token loop on CUDA.
Unlike upstream ReplaySSM we materialize the final state every block and do not
implement its across-round circular log or output-only UT transform.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["STEPS"])
def _gdn_transaction(Q, Kp, Vp, G, Beta, Initial, Final, Out, States,
                     T: tl.constexpr, STEPS, H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr, OUTPUT: tl.constexpr, SAVE: tl.constexpr):
    tile, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ik, iv = tl.arange(0, BK), tile * BV + tl.arange(0, BV)
    state_offsets = ((b * H + h) * DK + ik[:, None]) * DV + iv[None, :]
    valid = (ik[:, None] < DK) & (iv[None, :] < DV)
    state = tl.load(Initial + state_offsets, valid, 0).to(tl.float32)
    for t in range(STEPS):
        k = tl.load(Kp + ((b*T+t)*H+h)*DK + ik, ik < DK, 0).to(tl.float32)
        k *= tl.rsqrt(tl.sum(k*k, 0) + 1e-6)
        v = tl.load(Vp + ((b*T+t)*H+h)*DV + iv, iv < DV, 0).to(tl.float32)
        decay = tl.exp(tl.load(G + (b*T+t)*H+h).to(tl.float32))
        beta = tl.load(Beta + (b*T+t)*H+h).to(tl.float32)
        state *= decay
        delta = (v - tl.sum(state * k[:, None], 0)) * beta
        state += k[:, None] * delta[None, :]
        if OUTPUT:
            q = tl.load(Q + ((b*T+t)*H+h)*DK + ik, ik < DK, 0).to(tl.float32)
            q *= tl.rsqrt(tl.sum(q*q, 0) + 1e-6) * (DK ** -0.5)
            o = tl.sum(state*q[:, None], 0)
            tl.store(Out + ((b*T+t)*H+h)*DV+iv, o, iv < DV)
        if SAVE:
            offsets = (((b*T+t)*H+h)*DK + ik[:, None])*DV + iv[None, :]
            tl.store(States + offsets, state, valid)
    tl.store(Final + state_offsets, state, valid)


def run_recurrence(q, k, v, g, beta, initial, *, save_states=False, out_state=None, steps=None):
    batch, length, heads, dim_k = k.shape
    dim_v = v.shape[-1]
    steps = length if steps is None else steps
    if not 1 <= steps <= length or (steps != length and (q is not None or save_states)):
        raise ValueError("invalid GDN prefix replay length")
    tensors = (k, v, g, beta, initial) + ((q,) if q is not None else ())
    if any(not t.is_contiguous() or t.device != k.device for t in tensors):
        raise ValueError("GDN Triton transaction needs contiguous tensors on one CUDA device")
    if initial.shape != (batch, heads, dim_k, dim_v) or g.shape != (batch, length, heads) or beta.shape != g.shape:
        raise ValueError("incompatible GDN transaction shapes")
    if initial.dtype != torch.float32:
        raise ValueError("GDN Triton state experiment requires FP32 recurrent cache")
    if q is not None and q.shape != k.shape:
        raise ValueError("GDN query/key shapes must agree")
    final = torch.empty_like(initial) if out_state is None else out_state
    if final.shape != initial.shape or final.dtype != initial.dtype or not final.is_contiguous():
        raise ValueError("invalid GDN state output buffer")
    output = torch.empty_like(v) if q is not None else None
    states = torch.empty((batch, length, heads, dim_k, dim_v), device=k.device, dtype=torch.float32) if save_states else None
    # Tile V for GPU occupancy at B=1; each program keeps one K x 16 state tile.
    with torch.cuda.device(k.device):
        _gdn_transaction[(triton.cdiv(dim_v, 16), batch, heads)](
            q if q is not None else k, k, v, g, beta, initial, final,
            output if output is not None else v, states if states is not None else final,
            length, steps, heads, dim_k, dim_v, triton.next_power_of_2(dim_k), 16,
            q is not None, save_states, num_warps=4, enable_fp_fusion=False)
    return output, final, states
