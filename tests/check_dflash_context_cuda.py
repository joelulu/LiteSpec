"""Compile actual bounded-context CUDA FlexAttention forward/backward before training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from litespec.core.attention_masks import create_dflash_flex_mask,create_dflash_sdpa_mask
from litespec.modeling.draft.flex_attention import compile_friendly_flex_attention,flex_kernel_options


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback for this check')
    torch.manual_seed(42)
    for mode,slots,window in [('recent128',1,128),('pivot1',1,1),('pivot12',12,1)]:
        anchors=torch.tensor([[16,24]],device='cuda')
        keep=torch.ones_like(anchors,dtype=torch.bool)
        q=torch.randn(1,16,16,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn(1,4,32*slots+16,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        mask=create_dflash_flex_mask(anchors,keep,32,8,'cuda',context_window=window,context_slots=slots)
        actual=compile_friendly_flex_attention(q,k,v,block_mask=mask,enable_gqa=True,kernel_options=flex_kernel_options('TRITON'))
        dense=create_dflash_sdpa_mask(anchors,keep,32,8,'cuda',context_window=window,context_slots=slots)
        with torch.no_grad():expected=F.scaled_dot_product_attention(q,k,v,attn_mask=dense,enable_gqa=True)
        torch.testing.assert_close(actual,expected,atol=.04,rtol=.04)
        actual.float().square().mean().backward()
        torch.cuda.synchronize()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q,k,v))
        print(f'{mode}: CUDA Flex forward/backward passed',flush=True)


if __name__=='__main__':main()
