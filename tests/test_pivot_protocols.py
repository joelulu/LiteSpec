"""Regression checks for the expensive/full-prefix-training failure mode."""
from copy import deepcopy
import torch
import pytest

from litespec.vlm import train_aligned as aligned
from litespec.vlm.pivot_sft import OnlinePivotSFT
from litespec.vlm.pivot_training import OnlinePivot
from litespec.modeling.draft.dlite import DLiteDraftModel
from test_pivot_augmentation import draft_for, train_example
from test_vlm import tiny_qwen35, draft_config
from test_vlm_aligned import CONFIG
from pivot_screen_config import make_config


def config_for(draft,rollout):
    c=aligned.load_config(CONFIG)
    c.update(block_size=4,num_anchors=3,pivot_two_rounds=rollout,rollout_anchors=2,
        pivot_augmentation=draft.config.dlite_config['pivot_augmentation'])
    return c


@pytest.mark.parametrize('mode',['A','B','C','E'])
def test_shared_sft_one_target_pass_many_anchors_and_backward(mode):
    torch.set_num_threads(1);torch.manual_seed(17)
    target=tiny_qwen35().requires_grad_(False)
    if mode=='E':
        from transformers import Qwen3_5ForConditionalGeneration
        c=deepcopy(target.config);c.text_config.num_hidden_layers=32
        c.text_config.layer_types=(['linear_attention']*3+['full_attention'])*8
        target=Qwen3_5ForConditionalGeneration(c).eval().requires_grad_(False)
    draft=draft_for(mode)
    online=aligned.make_online(target,draft,config_for(draft,False))
    assert isinstance(online,OnlinePivotSFT)
    online.chunk_blocks=2  # exercises closure binding through checkpoint recompute
    calls=[]
    handle=target.register_forward_pre_hook(lambda m,a,k:calls.append(k['input_ids'].shape),with_kwargs=True)
    inputs=aligned.prepare_forward(target,online,train_example(target))
    from torch.distributed.utils import _cast_forward_inputs
    _,inputs=_cast_forward_inputs(torch.float32,**inputs)
    result=online(**inputs);result.loss.backward()
    handle.remove()
    assert calls==[torch.Size([1,20])]
    assert inputs['prepared_batch'].anchor_positions.size(1)==3
    assert 'round2/objective/final_ce' not in result.ratio_metrics
    assert all(p.grad is None for p in target.parameters())
    if mode!='A':
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in draft.pivot_augmentation.parameters())


@pytest.mark.parametrize('mode',['D0','D1','D2'])
@pytest.mark.parametrize('case',['image','text','mixed'])
def test_cached_matches_full_reference_and_encodes_original_images_once(mode,case):
    torch.set_num_threads(1);torch.manual_seed(11)
    target=tiny_qwen35().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    draft=draft_for(mode)
    online=aligned.make_online(target,draft,config_for(draft,True))
    assert isinstance(online,OnlinePivot)
    example=train_example(target)
    text_example=deepcopy(example)
    text_example.inputs={k:v for k,v in text_example.inputs.items() if k not in ('pixel_values','image_grid_thw')}
    text_example.inputs['input_ids'][:,:8]=3
    text_example.inputs['mm_token_type_ids'].zero_()
    if case=='text':
        example=text_example
    elif case=='mixed':
        for key in ('input_ids','mm_token_type_ids','attention_mask'):
            text_example.inputs[key]=text_example.inputs[key][:,:-2]
        example=aligned.collate_examples([example,text_example])
    kwargs=aligned.prepare_forward(target,online,example)
    rows=kwargs['rollout_inputs']['input_ids'].size(0)*2
    captures=[]
    original_loss=online.compute_supervised_loss
    def record(hidden,batch,reference,**kw):
        captures.append(dict(hidden=hidden.detach().clone(),pivot=batch.target_hidden.detach().clone(),
            reference=reference.detach().clone(),anchors=batch.anchor_positions.clone()))
        return original_loss(hidden,batch,reference,**kw)
    online.compute_supervised_loss=record
    # One row matches the first candidate; another misses, testing variable kept.
    def proposals(**kw):
        best=captures[0]['reference'][:,0,0].argmax(-1)
        assert not torch.isin(best,torch.tensor([91])).any()
        out=torch.zeros(best.size(0),3,dtype=torch.long)
        out[:,0]=best
        out[1::2,0]=(best[1::2]+1)%128
        return out,None
    draft.sample_draft_tokens=proposals
    shapes=[];vision=[]
    h=target.register_forward_pre_hook(lambda m,a,k:shapes.append(k['input_ids'].shape),with_kwargs=True)
    v=target.model.visual.register_forward_pre_hook(lambda m,a:vision.append(a[0].size(0)))
    torch.manual_seed(333)
    cached=online(**kwargs)
    cached_captures=captures[:]
    cached.loss.backward()
    h.remove();v.remove()
    assert len(shapes)==4
    assert shapes[1:]==[torch.Size([rows,3]),torch.Size([rows,4]),torch.Size([rows,7])]
    assert vision==([] if case=='text' else [4])  # original one-image microbatch, NOT two anchor copies
    assert cached.sum_metrics['rollout/visual_encoder_calls'].item()==(0 if case=='text' else 1)
    assert cached_captures[1]['anchors'][0].item()!=cached_captures[0]['anchors'][0].item()+1
    assert cached_captures[1]['anchors'][1].item()==cached_captures[0]['anchors'][1].item()+1
    online.zero_grad(set_to_none=True)
    captures.clear();online.rollout_backend='full_replay'
    torch.manual_seed(333)
    oracle=online(**kwargs)
    for actual,expected in zip(cached_captures,captures):
        torch.testing.assert_close(actual['anchors'],expected['anchors'],rtol=0,atol=0)
        for key in ['hidden','pivot','reference']:
            torch.testing.assert_close(actual[key],expected[key],rtol=5e-4,atol=3e-5)
    torch.testing.assert_close(cached.loss,oracle.loss,rtol=3e-5,atol=3e-6)


def test_protocol_budgets_are_separate_and_common_batch_is_64():
    for mode in ['A','B','C','E','D0','D1','D2']:
        c=make_config(mode,{})
        rollout=mode.startswith('D')
        assert c['pivot_two_rounds']==rollout
        assert c['num_anchors']==512
        assert c['transition_policy']=='trajectory'
        assert c['batch_size']*c['accumulation_steps']*8==64
        assert c['num_epochs']==2 and c['final_only']
        assert 'init_checkpoint' not in c
    assert make_config('C',{'NUM_ANCHORS':'64'})['num_anchors']==64
    assert make_config('D1',{'NUM_ANCHORS':'64'})['num_anchors']==64
