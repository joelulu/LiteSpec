"""FP32 sparse GDN snapshots. Allocate states only for selected anchors."""
import torch
import triton
import triton.language as tl


@triton.jit
def _capture(K, V, G, Beta, Anchors, Keep, States,
             T: tl.constexpr, A: tl.constexpr, H: tl.constexpr, DK: tl.constexpr,
             DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    tile, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ik, iv = tl.arange(0, BK), tile * BV + tl.arange(0, BV)
    valid = (ik[:, None] < DK) & (iv[None, :] < DV)
    state = tl.full((BK, BV), 0, tl.float32)
    slot = 0
    anchor = tl.load(Anchors + b*A, A > 0, T+1)
    active = tl.load(Keep + b*A, A > 0, 0)
    # Anchors denote the first unevaluated input. Capture after index anchor-1.
    t = 0
    while (t < T) & (slot < A) & active:
        k = tl.load(K + ((b*T+t)*H+h)*DK + ik, ik < DK, 0).to(tl.float32)
        k *= tl.rsqrt(tl.sum(k*k, 0) + 1e-6)
        v = tl.load(V + ((b*T+t)*H+h)*DV + iv, iv < DV, 0).to(tl.float32)
        state *= tl.exp(tl.load(G + (b*T+t)*H+h).to(tl.float32))
        delta = (v - tl.sum(state*k[:, None], 0)) * tl.load(Beta + (b*T+t)*H+h).to(tl.float32)
        state += k[:, None] * delta[None, :]
        if active and anchor == t+1:
            offsets = (((b*A+slot)*H+h)*DK + ik[:, None])*DV + iv[None, :]
            tl.store(States + offsets, state, valid)
            slot += 1
            anchor = tl.load(Anchors + b*A+slot, slot < A, T+1)
            active = tl.load(Keep + b*A+slot, slot < A, 0)
        t += 1


def sparse_anchor_states(k, v, g, beta, anchors, keep):
    b, t, h, dk = k.shape
    dv, a = v.shape[-1], anchors.shape[1]
    if a < 1 or anchors.shape != keep.shape or anchors.shape[0] != b:
        raise ValueError("invalid sparse GDN anchor layout")
    if v.shape[:3] != (b,t,h) or g.shape != (b,t,h) or beta.shape != g.shape:
        raise ValueError("incompatible sparse GDN update inputs")
    if any(x.device != k.device for x in (v,g,beta,anchors,keep)):
        raise ValueError("GDN inputs must be on the same CUDA device")
    k,v,g,beta,anchors,keep = (x.contiguous() for x in (k,v,g,beta,anchors,keep))
    states = torch.zeros(b,a,h,dk,dv,device=k.device,dtype=torch.float32)
    with torch.cuda.device(k.device):
        _capture[(triton.cdiv(dv,16),b,h)](k,v,g,beta,anchors,keep,states,
            t,a,h,dk,dv,triton.next_power_of_2(dk),16,num_warps=4,enable_fp_fusion=False)
    return states
