"""Offline SM90 compilation catches Triton constructs the interpreter accepts."""
import pytest


@pytest.mark.parametrize('dtype,dk,dv', [('bf16',128,128),('fp32',7,19)])
def test_sparse_capture_compiles_without_gpu(dtype,dk,dv):
    triton=pytest.importorskip('triton')
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from litespec.vlm.gdn_capture_triton import _capture
    source=ASTSource(_capture,signature=dict(K='*'+dtype,V='*'+dtype,G='*fp32',
        Beta='*'+dtype,Anchors='*i64',Keep='*i1',States='*fp32'),
        constexprs=dict(T=257,A=4,H=32,DK=dk,DV=dv,BK=triton.next_power_of_2(dk),BV=16))
    compiled=triton.compile(source,target=GPUTarget('cuda',90,32),
        options=dict(num_warps=4,enable_fp_fusion=False))
    assert 'ptx' in compiled.asm
