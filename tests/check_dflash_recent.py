"""Recent-window numerical/Flex and FSDP checks; no model weights or images."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_dflash_recent import tiny


MODES=(('recent256',256),('recent64',64),('recent0',0))


def check(device='cpu',flex=False):
    import torch
    from litespec.core.attention_masks import create_dflash_sdpa_mask,create_dflash_flex_mask
    from litespec.modeling.draft.flex_attention import flex_kernel_options
    from litespec.modeling.draft.dflash import DFlashDraftModel
    if flex and not device.startswith('cuda'):raise ValueError('--flex requires CUDA')
    dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32
    torch.manual_seed(42)
    for mode,window in MODES:
        draft=tiny(mode,device,width=64).to(dtype=dtype).eval()
        anchors=torch.tensor([[12,280]],device=device);keep=torch.ones_like(anchors,dtype=torch.bool)
        sequence=320 if window else 0
        context=torch.randn(1,sequence,320,device=device,dtype=dtype)
        query=torch.randn(1,16,64,device=device,dtype=dtype)
        positions=torch.cat([torch.arange(sequence,device=device)[None],
            (anchors[...,None]+torch.arange(8,device=device)).flatten(1)],1)
        dense=create_dflash_sdpa_mask(anchors,keep,sequence,8,device,context_window=window)
        kwargs=dict(position_ids=positions,target_hidden=context,noise_embedding=query,is_causal=False)
        expected=draft(attention_mask=dense,**kwargs)
        for i,anchor in enumerate(anchors[0].tolist()):
            left=max(0,anchor-window)
            hidden=context[:,left:anchor] if window else context
            pos=torch.cat([torch.arange(left,anchor,device=device) if window else torch.empty(0,device=device,dtype=torch.long),
                torch.arange(anchor,anchor+8,device=device)])[None]
            inferred=draft(position_ids=pos,target_hidden=hidden,noise_embedding=query[:,i*8:(i+1)*8],is_causal=False)
            torch.testing.assert_close(expected[:,i*8:(i+1)*8],inferred,atol=.025 if flex else 2e-6,rtol=.025 if flex else 2e-5)
        readout=torch.randn_like(expected)
        (expected.float()*readout.float()).mean().backward()
        dense_grad={name:p.grad.detach().clone() for name,p in draft.named_parameters()}
        assert all(torch.isfinite(g).all() for g in dense_grad.values())
        if flex:
            draft.zero_grad(set_to_none=True);draft.config._attn_implementation='flex_attention'
            mask=create_dflash_flex_mask(anchors,keep,sequence,8,device,context_window=window)
            actual=draft(attention_mask=mask,kernel_options=flex_kernel_options('TRITON'),**kwargs)
            torch.testing.assert_close(actual,expected,atol=.025,rtol=.025)
            (actual.float()*readout.float()).mean().backward()
            for name,p in draft.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                torch.testing.assert_close(p.grad,dense_grad[name],atol=.025,rtol=.08)
        if window==0:
            draft.config._attn_implementation='sdpa'
            changed=draft(attention_mask=dense,**dict(kwargs,target_hidden=torch.randn(1,999,320,device=device,dtype=dtype)))
            torch.testing.assert_close(changed,expected,atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as path:
            draft.save_pretrained(path)
            restored=DFlashDraftModel.from_pretrained(path,attn_implementation='sdpa').to(device=device,dtype=dtype).eval()
            torch.testing.assert_close(restored(attention_mask=dense,**kwargs),expected)
        print(f'{mode}: prefix parity, gradients, reload, zero-context isolation; Flex={flex} passed',flush=True)


def check_fsdp():
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
    from litespec.core.dflash import OnlineDFlashModel
    from litespec.training.optimizer import BF16Optimizer,register_optimizer_parameters
    from litespec.training.fsdp_layout import build_fsdp_layouts
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device=torch.device('cuda',torch.cuda.current_device())
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    try:
        if dist.get_world_size()!=8:raise ValueError('Supplement check requires exactly eight ranks')
        for mode,window in MODES:
            torch.manual_seed(42)
            draft=tiny(mode,device,width=64).to(dtype=torch.bfloat16).train()
            draft.config._attn_implementation='flex_attention'
            emb=torch.nn.Embedding(128,64,device=device,dtype=torch.bfloat16).requires_grad_(False)
            head=torch.nn.Linear(64,128,bias=False,device=device,dtype=torch.bfloat16).requires_grad_(False)
            online=OnlineDFlashModel(draft,head,emb,127,block_size=8,num_anchors=2,
                attention_backend='flex_attention',selector_loss_alpha=0,loss_decay_gamma=4.)
            parameters,specs=register_optimizer_parameters(draft)
            fsdp=FSDP(online,ignored_modules=[emb,head],use_orig_params=True,
                sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16),device_id=device)
            layouts=build_fsdp_layouts(fsdp,parameters,specs)
            optimizer=BF16Optimizer(draft,lr=.001,weight_decay=0,max_grad_norm=1,total_steps=2,warmup_ratio=0,
                parameters=parameters,parameter_specs=specs,parameter_layouts=layouts)
            with FSDP.summon_full_params(fsdp,writeback=False):
                before=draft.layers[0].mlp.down_proj.weight.detach().float().cpu().clone()
            torch.manual_seed(100+dist.get_rank())
            for step in range(2):
                optimizer.zero_grad()
                for micro in range(8):
                    length=(40,48)[micro%2]
                    ids=torch.randint(0,120,(2,length),device=device)
                    hidden=torch.randn(2,length,320,device=device,dtype=torch.bfloat16)
                    loss=fsdp(input_ids=ids,hidden_states=hidden,loss_mask=torch.ones_like(ids).float()).loss/8
                    assert torch.isfinite(loss),'nonfinite FSDP loss'
                    loss.backward()
                norm=torch.zeros((),device=device,dtype=torch.float64)
                for p in parameters:
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(),'nonfinite shard gradient'
                        norm+=p.grad.detach().double().square().sum()
                dist.all_reduce(norm);assert torch.isfinite(norm) and norm.item()>0
                optimizer.step()
            with FSDP.summon_full_params(fsdp,writeback=False):
                after=draft.layers[0].mlp.down_proj.weight.detach().float().cpu().clone()
            assert not torch.equal(before,after),'optimizer did not update Draft weights'
            if dist.get_rank()==0:print(f'{mode}: eight-rank BF16/Flex/FSDP, two optimizer steps x eight accumulation passed',flush=True)
            del optimizer,layouts,fsdp,online,draft,parameters
            torch.cuda.empty_cache();dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu');parser.add_argument('--flex',action='store_true')
    parser.add_argument('--fsdp',action='store_true');args=parser.parse_args()
    if args.fsdp:check_fsdp()
    else:check(args.device,args.flex)
