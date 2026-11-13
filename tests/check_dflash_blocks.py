"""Real four-variant prefix/reload and production FSDP optimizer smoke checks."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tiny(mode, block, device='cpu', dtype=None):
    import torch
    from transformers import Qwen3Config
    from litespec.modeling.draft.dflash import DFlashDraftModel
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=4096, layer_types=['full_attention']*5,
        attention_dropout=0.)
    cfg.architectures=['DFlashDraftModel']; cfg.block_size=block; cfg.num_target_layers=32
    cfg.dflash_config=dict(target_layer_ids=[1,8,15,22,29], mask_token_id=127)
    cfg.vlm_config=dict(draft_context=mode); cfg._attn_implementation='sdpa'
    return DFlashDraftModel(cfg).to(device=device, dtype=dtype or torch.float32)


def check(device='cpu', flex=False):
    import torch
    from litespec.core.attention_masks import create_dflash_sdpa_mask, create_dflash_flex_mask
    from litespec.core.dflash import OnlineDFlashModel
    from litespec.modeling.draft.flex_attention import flex_kernel_options
    from litespec.vlm.baselines import load_draft
    from dflash_30w import VARIANTS
    if flex and not str(device).startswith('cuda'): raise ValueError('Flex check requires CUDA')
    dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32
    torch.set_num_threads(1)
    for label,(mode,block) in VARIANTS.items():
        torch.manual_seed(42)
        draft=tiny(mode,block,device,dtype).eval()
        emb=torch.nn.Embedding(128,64,device=device,dtype=dtype).requires_grad_(False)
        head=torch.nn.Linear(64,128,bias=False,device=device,dtype=dtype).requires_grad_(False)
        online=OnlineDFlashModel(draft,head,emb,127,block_size=block,num_anchors=2,
            attention_backend='sdpa',selector_loss_alpha=0,loss_decay_gamma=4.)
        ids=torch.randint(0,120,(1,64),device=device)
        features=torch.randn(1,64,320,device=device,dtype=dtype)
        anchors=torch.tensor([[16,40]],device=device);keep=torch.ones_like(anchors,dtype=torch.bool)
        online._sample_anchor_positions=lambda *a,**kw:(anchors,keep)
        _,_,trained=online._forward_draft_blocks(ids,features,torch.ones_like(ids).float())
        for i,a in enumerate(anchors[0].tolist()):
            left=0 if mode=='full' else a-1
            query=ids.new_full((1,block),127);query[:,0]=ids[:,a]
            positions=torch.cat([torch.arange(left,a,device=device),torch.arange(a,a+block,device=device)])[None]
            probe=dict(position_ids=positions,target_hidden=features[:,left:a],
                noise_embedding=emb(query),is_causal=False)
            actual=draft(**probe)
            torch.testing.assert_close(actual,trained[:,i*block:(i+1)*block],atol=.025 if flex else 2e-6,rtol=.025 if flex else 2e-5)
        loss=online(input_ids=ids,hidden_states=features,loss_mask=torch.ones_like(ids).float()).loss
        assert torch.isfinite(loss); loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters())
        if flex:
            draft.zero_grad(set_to_none=True)
            query=torch.randn(1,2*block,64,device=device,dtype=dtype)
            positions=torch.cat([torch.arange(64,device=device)[None],
                (anchors[...,None]+torch.arange(block,device=device)).flatten(1)],1)
            kwargs=dict(position_ids=positions,target_hidden=features,noise_embedding=query,is_causal=False)
            window=None if mode=='full' else 1
            dense=create_dflash_sdpa_mask(anchors,keep,64,block,device,context_window=window)
            expected=draft(attention_mask=dense,**kwargs)
            draft.config._attn_implementation='flex_attention'
            mask=create_dflash_flex_mask(anchors,keep,64,block,device,context_window=window)
            actual=draft(attention_mask=mask,kernel_options=flex_kernel_options('TRITON'),**kwargs)
            torch.testing.assert_close(actual,expected,atol=.025,rtol=.025)
            actual.float().square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in draft.parameters())
            draft.config._attn_implementation='sdpa'
        with tempfile.TemporaryDirectory() as directory:
            draft.save_pretrained(directory)
            restored=load_draft(directory,attn_implementation='sdpa').to(device=device,dtype=dtype).eval()
            assert restored.block_size==block and restored.context_window==(None if mode=='full' else 1)
            with torch.no_grad():torch.testing.assert_close(restored(**probe),draft(**probe))
        print(f'{label}: training/inference prefix parity, finite loss/gradients, checkpoint block reload; Flex={flex}',flush=True)


def check_fsdp():
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from litespec.core.dflash import OnlineDFlashModel
    from litespec.training.optimizer import BF16Optimizer, register_optimizer_parameters
    from litespec.training.fsdp_layout import build_fsdp_layouts
    from dflash_30w import VARIANTS
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device=torch.device('cuda',torch.cuda.current_device())
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    try:
        assert dist.get_world_size()==4
        for label,(mode,block) in VARIANTS.items():
            torch.manual_seed(42)
            draft=tiny(mode,block,device,torch.bfloat16).train()
            draft.config._attn_implementation='flex_attention'
            emb=torch.nn.Embedding(128,64,device=device,dtype=torch.bfloat16).requires_grad_(False)
            head=torch.nn.Linear(64,128,bias=False,device=device,dtype=torch.bfloat16).requires_grad_(False)
            online=OnlineDFlashModel(draft,head,emb,127,block_size=block,num_anchors=2,
                attention_backend='flex_attention',selector_loss_alpha=0,loss_decay_gamma=4.)
            params,specs=register_optimizer_parameters(draft)
            fsdp=FSDP(online,ignored_modules=[emb,head],use_orig_params=True,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16),device_id=device)
            layouts=build_fsdp_layouts(fsdp,params,specs)
            optimizer=BF16Optimizer(draft,lr=.001,weight_decay=0,max_grad_norm=1,total_steps=2,
                warmup_ratio=0,parameters=params,parameter_specs=specs,parameter_layouts=layouts)
            with FSDP.summon_full_params(fsdp,writeback=False):
                before=draft.layers[0].mlp.down_proj.weight.detach().float().cpu().clone()
            torch.manual_seed(100+dist.get_rank())
            for step in range(2):
                optimizer.zero_grad()
                for micro in range(32):
                    length=(48,64)[micro%2]
                    ids=torch.randint(0,120,(2,length),device=device)
                    features=torch.randn(2,length,320,device=device,dtype=torch.bfloat16)
                    loss=fsdp(input_ids=ids,hidden_states=features,loss_mask=torch.ones_like(ids).float()).loss/32
                    assert torch.isfinite(loss);loss.backward()
                norm=torch.zeros((),device=device,dtype=torch.float64)
                for p in params:
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all();norm+=p.grad.detach().double().square().sum()
                dist.all_reduce(norm);assert norm.item()>0 and torch.isfinite(norm)
                optimizer.step()
            with FSDP.summon_full_params(fsdp,writeback=False):
                after=draft.layers[0].mlp.down_proj.weight.detach().float().cpu().clone()
            assert not torch.equal(before,after)
            if dist.get_rank()==0:print(f'{label}: 4-rank BF16/Flex/FSDP + FP32 master optimizer, 2x32 accumulation passed',flush=True)
            del optimizer,layouts,fsdp,online,draft,params
            torch.cuda.empty_cache();dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu');parser.add_argument('--flex',action='store_true')
    parser.add_argument('--fsdp',action='store_true');args=parser.parse_args()
    if args.fsdp:check_fsdp()
    else:check(args.device,args.flex)
