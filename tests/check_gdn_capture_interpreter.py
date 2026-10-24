"""CPU interpreter check of the actual sparse Triton kernel (no GPU claim).

Run: TRITON_INTERPRET=1 PYTHONPATH=. python tests/check_gdn_capture_interpreter.py
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get('TRITON_INTERPRET') != '1':
    raise RuntimeError('Set TRITON_INTERPRET=1 before importing Triton')
import torch
import triton
from litespec.vlm.gdn_capture import reference_anchor_states
from litespec.vlm.gdn_capture_triton import _capture


def check():
    torch.manual_seed(7)
    for dk,dv in ((7,19),(128,128)):
        b,t,h,a=3,17,2,4
        k=torch.randn(b,t,h,dk)
        v=torch.randn(b,t,h,dv)
        g=-torch.rand(b,t,h)
        beta=torch.rand(b,t,h)
        anchors=torch.tensor([[1,4,t,0],[2,8,0,0],[0,0,0,0]])
        keep=anchors>0
        expected=reference_anchor_states(k,v,g,beta,anchors,keep)
        observed=torch.zeros_like(expected)
        _capture[(triton.cdiv(dv,16),b,h)](k,v,g,beta,anchors,keep,observed,
            t,a,h,dk,dv,triton.next_power_of_2(dk),16)
        torch.testing.assert_close(observed,expected,atol=3e-6,rtol=3e-5)
        print(f'sparse Triton interpreter: K={dk}, V={dv}, mixed anchor counts: passed')


if __name__ == '__main__':
    check()
