"""Fail-fast sparse capture + native-image train/export/reject CUDA check.

Uses a randomly initialized tiny Qwen3.5, never a model/data download. This
checks engineering correctness; it is not evidence of acceptance or speed gains.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from litespec.vlm import baselines, train_aligned as aligned
from litespec.vlm.context import visual_positions
from litespec.vlm.gdn_capture import anchor_states, reference_anchor_states
from litespec.vlm.runtime import PreparedExample
from litespec.vlm.static_batch import generate_static_batch


def check_sparse():
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for dk, dv, length in ((128,128,33), (7,13,17), (128,128,257)):
            b, h = 2, 4
            k = torch.randn(b,length,h,dk,dtype=dtype,device='cuda')
            v = torch.randn(b,length,h,dv,dtype=dtype,device='cuda')
            g = -torch.rand(b,length,h,device='cuda')
            beta = torch.rand(b,length,h,dtype=dtype,device='cuda')
            anchors = torch.tensor([[1,4,length,0], [2,length-1,0,0]], device='cuda')
            keep = anchors > 0
            expected = reference_anchor_states(*(x.cpu() for x in (k,v,g,beta,anchors,keep)))
            observed = anchor_states(k,v,g,beta,anchors,keep).cpu()
            torch.testing.assert_close(observed, expected, atol=3e-5, rtol=3e-4)
            results.append(dict(dtype=str(dtype), dk=dk, dv=dv, length=length,
                                max_abs_error=(observed-expected).abs().max().item()))
    return results


def check_image_pipeline(device="cuda", dtype=torch.bfloat16):
    config = Qwen3_5Config(text_config=dict(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=128,
        linear_value_head_dim=128, layer_types=['linear_attention']*3+['full_attention'],
        eos_token_id=120, pad_token_id=0,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[2,2,4])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        patch_size=2, temporal_patch_size=2, spatial_merge_size=1, out_hidden_size=64,
        num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=91, vision_start_token_id=90, vision_end_token_id=92)
    target = Qwen3_5ForConditionalGeneration(config).to(device=device,dtype=dtype).eval().requires_grad_(False)
    processor = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda:{str(i):i for i in range(121)}))
    inputs = dict(input_ids=torch.tensor([[1,90,91,91,91,91,92,2,4,5,6,7,8,9,10,11,120]],device=device),
        pixel_values=torch.randn(4,24,device=device,dtype=dtype),
        image_grid_thw=torch.tensor([[1,2,2]],device=device),
        mm_token_type_ids=torch.tensor([[0,0,1,1,1,1,0,0]+[0]*9],device=device))
    train = PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0,:8],config),'cuda-check')
    prompt = PreparedExample({k:(v[:,:8] if k in ('input_ids','mm_token_type_ids') else v) for k,v in inputs.items()},
                             8,train.vision_mask,'cuda-check')
    results = []
    for family in ('dflash','dlite'):
        source = 'sft_student_aligned.yaml' if family == 'dlite' else 'sft_dflash.yaml'
        cfg = aligned.load_config(Path(__file__).resolve().parents[1]/'scripts/config/vlm'/source)
        cfg.update(algorithm=family,num_draft_layers=1,chs_num_layers=2,target_layer_ids=[0,3],
            sequential_rank=8,block_size=4,num_anchors=4,flex_attention_backend='TRITON',
            gdn_experiment=True,gdn_enabled=True,gdn_layer_ids=[0,2],gdn_rank=2,
            gdn_bottleneck=8,gdn_chunk_anchors=2)
        draft = aligned.make_draft(target,processor,cfg) if family == 'dlite' else baselines.make_draft(target,processor,cfg)
        # This tiny check uses SDPA to isolate reader/capture math. Production
        # CUDA training continues to use the existing FlexAttention recipe.
        draft.config._attn_implementation='sdpa'
        if family == 'dlite':
            online = aligned.CPUReferenceDLite(draft,target.get_output_embeddings(),target.get_input_embeddings(),
                draft.mask_token_id,4,attention_backend='flex_attention',num_anchors=4,
                final_ce_weight=.1,l1_loss_weight=1,base_lm_ce_weight=0)
        else:
            online = baselines.make_online(target,draft,cfg)
            online.attention_backend='sdpa'
        kwargs = aligned.prepare_forward(target,online,train)
        assert all(s.dtype == torch.float32 for s in kwargs['gdn_states'])
        loss = online(**kwargs).loss
        loss.backward()
        assert torch.isfinite(loss) and draft.layers[0].gdn_reader.up.weight.grad.abs().sum() > 0
        with torch.no_grad():
            draft.layers[0].gdn_reader.up.weight.add_(.01)
        with tempfile.TemporaryDirectory() as directory:
            draft.save_pretrained(directory)
            loaded = baselines.load_draft(directory,attn_implementation='sdpa',dtype=dtype).to(device).eval()
            if family == 'dlite':
                loaded.sample_draft_tokens=lambda **kw:(torch.zeros(1,3,dtype=torch.long,device=device),None)
            else:
                loaded._propose_draft_tokens=lambda *a,**kw:(torch.zeros(1,3,dtype=torch.long,device=device),None,None)
            for mode in ('per_token','compact_replay'):
                actual = generate_static_batch(target,loaded,prompt,max_new_tokens=5,state_mode=mode)
                reference = generate_static_batch(target,None,prompt,max_new_tokens=5)
                if actual['token_ids'] != reference['token_ids']:
                    raise AssertionError(f'{family}/{mode}: tiny-image greedy output differs from AR')
        results.append(dict(family=family,loss=float(loss.detach()),passed=True))
    return results


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback for the launch preflight')
    torch.manual_seed(42)
    print(json.dumps(dict(sparse=check_sparse(), image_pipeline=check_image_pipeline()),indent=2))
