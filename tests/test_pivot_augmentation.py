from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm import train_aligned as aligned
from litespec.vlm.pivot_training import eligible_anchors, repeat_examples, transition
from litespec.vlm.pivot_augmentation import prefix_data, AugmentationState
from litespec.vlm.runtime import PreparedExample, visual_positions, draft_block
from litespec.vlm.context import ContextPolicy
from litespec.vlm.static_batch import generate_static_batch
from litespec.training.optimizer import register_optimizer_parameters
from test_vlm import draft_config, tiny_qwen35, tiny_image_inputs
from test_vlm_aligned import CONFIG, Processor


def draft_for(mode):
    cfg = draft_config()
    if mode == 'E':
        cfg.num_target_layers = 32
        cfg.dlite_config.update(chs_num_layers=3, target_layer_ids=[0,7,18])
    cfg.dlite_config['pivot_augmentation'] = dict(mode=mode, rank=8, history_positions=4)
    return DLiteDraftModel(cfg)


def train_example(target):
    inputs = tiny_image_inputs()
    answer = torch.tensor([[4,5,6,7,8,9,10,11,12,13,14,15]])
    inputs['input_ids'] = torch.cat([inputs['input_ids'],answer],1)
    inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(answer)],1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    return PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0,:8],target.config),'tiny')


def test_real_boundary_and_next_token_alignment():
    ids = torch.arange(20)[None]
    candidates = torch.tensor([[4,5,99,88]])
    predictions = torch.tensor([[5,6,77,66]])
    kept,a,bonus,corrected = transition(ids,torch.tensor([[4]]),candidates,predictions)
    assert kept.item()==2 and a.item()==6 and bonus.item()==6
    assert corrected.tolist()==ids.tolist()
    d=draft_for('D2')
    q=torch.randn(1,4,32);pivot=torch.randn(1,1,2,32)
    data=dict(feedback=torch.randn_like(pivot),correct_embedding=torch.randn(1,32),
              wrong_embedding=torch.randn(1,32),distance=torch.ones(1,dtype=torch.long),
              feedback_valid=torch.ones(1,dtype=torch.bool))
    actual_pivot,actual_q=d.pivot_augmentation.inputs(pivot,q,data)
    torch.testing.assert_close(actual_pivot,pivot,rtol=0,atol=0)
    torch.testing.assert_close(actual_q[:,[0,2,3]],q[:,[0,2,3]],rtol=0,atol=0)
    assert not torch.equal(actual_q[:,1],q[:,1])
    data['feedback_valid'].zero_()
    _,actual_q=d.pivot_augmentation.inputs(pivot,q,data)
    torch.testing.assert_close(actual_q,q,rtol=0,atol=0)


def test_history_excludes_pivot_future_and_delta_is_adjacent():
    for mode in ['B','C']:
        d=draft_for(mode)
        src={i:torch.arange(10)[None,:,None].expand(1,-1,32).float() for i in d.target_layer_ids}
        data=prefix_data(src,torch.tensor([[6]]),d)
        if mode=='B':
            assert data['history'][0,:,0,0].tolist()==[1,2,3,4]
        else:
            assert data['previous'][0,0,0,0].item()==4
        for x in src.values(): x[:,6:]+=100000
        newer=prefix_data(src,torch.tensor([[6]]),d)
        for key in data: torch.testing.assert_close(data[key],newer[key],rtol=0,atol=0)


def test_common_initialization_and_optimizer_ownership():
    torch.manual_seed(42);plain=DLiteDraftModel(draft_config())
    for mode in ['A','B','C','D1','D2']:
        torch.manual_seed(42);d=draft_for(mode)
        for key,value in plain.state_dict().items():
            torch.testing.assert_close(value,d.state_dict()[key],rtol=0,atol=0)
        register_optimizer_parameters(d)
    register_optimizer_parameters(draft_for('E'))


@pytest.mark.parametrize('mode',['D0','D1','D2'])
def test_real_two_round_image_training_backward_and_frozen_target(mode):
    torch.set_num_threads(1);torch.manual_seed(42)
    target=tiny_qwen35().requires_grad_(False)
    if mode=='E':
        from transformers import Qwen3_5ForConditionalGeneration
        config=deepcopy(target.config)
        config.text_config.num_hidden_layers=32
        config.text_config.layer_types=['linear_attention']*3+['full_attention']
        config.text_config.layer_types*=8
        target=Qwen3_5ForConditionalGeneration(config).eval().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    d=draft_for(mode)
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,num_anchors=2,pivot_two_rounds=True,rollout_anchors=2)
    online=aligned.make_online(target,d,cfg)
    inputs=aligned.prepare_forward(target,online,train_example(target))
    from torch.distributed.utils import _cast_forward_inputs
    args,kwargs=_cast_forward_inputs(torch.float32,**inputs)
    output=online(**kwargs)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(p.grad is None for p in target.parameters())
    assert d.sequential_head.output_proj.weight.grad.abs().sum()>0
    if mode not in ('A','D0'):
        grads=[p.grad for p in d.pivot_augmentation.parameters()]
        assert any(g is not None and g.abs().sum()>0 for g in grads)
    assert not any(k.startswith('_frozen_target') for k in online.state_dict())


@pytest.mark.parametrize('mode',['A','B','C','D1','D2','E'])
def test_reload_and_disabled_path(mode,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(42)
    d=draft_for(mode).eval()
    d.save_pretrained(tmp_path)
    loaded,info=DLiteDraftModel.from_pretrained(tmp_path,output_loading_info=True,attn_implementation='sdpa')
    assert not info['missing_keys'] and not info['unexpected_keys']
    for k,v in d.state_dict().items(): torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
    d.pivot_augmentation.enabled=False
    cfg=deepcopy(d.config);cfg.dlite_config.pop('pivot_augmentation')
    plain=DLiteDraftModel(cfg).eval()
    plain.load_state_dict({k:v for k,v in d.state_dict().items() if not k.startswith('pivot_augmentation.')})
    pivot=torch.randn(1,1,d.chs_num_layers,32)
    kwargs=dict(embedding=torch.nn.Embedding(128,32),anchor_ids=torch.tensor([4]),pivot=pivot,
        history=None,anchor=7,prefill_length=7,vision_mask=torch.zeros(7,dtype=torch.bool),policy=ContextPolicy.no_history())
    a,_=draft_block(d,**kwargs);b,_=draft_block(plain,**kwargs)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('mode',['A','B','C','D1','D2'])
@torch.no_grad()
def test_per_token_image_decode_and_no_stale_feedback(mode):
    torch.set_num_threads(1);torch.manual_seed(7)
    target=tiny_qwen35().eval();target.config.text_config.eos_token_id=-1
    draft=draft_for(mode).eval()
    # Force rejection repeatedly, including the shortened final block.
    draft.sample_draft_tokens=lambda **kw:(torch.zeros(kw['draft_hidden'].size(0),3,dtype=torch.long),None)
    inputs=tiny_image_inputs()
    ex=PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),'tiny')
    ar=generate_static_batch(target,None,ex,max_new_tokens=7)
    spec=generate_static_batch(target,draft,ex,max_new_tokens=7,state_mode='per_token')
    assert spec['token_ids']==ar['token_ids']
    assert spec['recovery_replays']==0


def test_e_stage_query_gates_and_backward():
    d=draft_for('E')
    q=torch.randn(2,4,32,requires_grad=True);pivot=torch.randn(2,1,3,32)
    cp=torch.zeros(2,3,dtype=torch.long);qp=torch.arange(1,5)[None].expand(2,-1)
    out=d(noise_embedding=q,target_hidden=pivot,history_hidden=q.new_empty(2,1,0,32),
        position_ids=qp,rotary_position_ids=torch.cat([cp,qp],-1),augmentation_data={})
    out.square().mean().backward()
    for reader in d.pivot_augmentation.reads[0].readers:
        assert reader.k.weight.grad is not None
    assert d.pivot_augmentation.reads[0].selector.weight.grad.abs().sum()>0


@pytest.mark.parametrize('mode',['A','B','C','D1','D2'])
@torch.no_grad()
def test_same_prefix_training_inference_alignment(mode):
    torch.set_num_threads(1);torch.manual_seed(42)
    target=tiny_qwen35().requires_grad_(False)
    d=draft_for(mode).eval()
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,pivot_two_rounds=True,rollout_anchors=1)
    online=aligned.make_online(target,d,cfg).eval()
    ex=train_example(target);a=torch.tensor([[10]])
    source,_=online.target_pass(ex.inputs,a)
    batch=online.make_batch(ex.inputs['input_ids'],source,aligned.loss_mask(ex),a)
    feedback={}
    if mode in ('D1','D2'):
        feedback=dict(feedback=torch.randn(1,1,2,32),feedback_valid=torch.ones(1,dtype=torch.bool),
            wrong_embedding=torch.randn(1,32),correct_embedding=torch.randn(1,32),distance=torch.ones(1,dtype=torch.long))
    expected=online.backbone(batch,source,feedback)[:,0]
    states=target(**ex.inputs,use_cache=False,output_hidden_states=True).hidden_states
    state=AugmentationState(d,target,tuple(x[:,:10] for x in states),10)
    state.record=feedback
    pivot=torch.stack([source[i][:,9:10] for i in d.target_layer_ids],2)
    actual,_=draft_block(d,target.get_input_embeddings(),anchor_ids=ex.inputs['input_ids'][:,10:11],
        pivot=pivot,history=None,anchor=10,prefill_length=8,vision_mask=ex.vision_mask,
        policy=ContextPolicy.no_history(),augmentation_data=state.data())
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


def test_eligibility_contiguous_and_replicated_image_order():
    mask=torch.zeros(2,20);mask[:,8:]=1
    mask[1,12]=0
    valid=eligible_anchors(mask,4)
    assert valid[0,8] and not valid[1].any()
    x=tiny_image_inputs();x['input_ids']=x['input_ids'].repeat(2,1)
    x['pixel_values']=torch.cat([x['pixel_values'],x['pixel_values']+10])
    x['image_grid_thw']=x['image_grid_thw'].repeat(2,1)
    x['mm_token_type_ids']=x['mm_token_type_ids'].repeat(2,1)
    y=repeat_examples(x,3)
    torch.testing.assert_close(y['pixel_values'],x['pixel_values'].repeat(3,1))
    assert y['input_ids'].size(0)==6


@pytest.mark.parametrize('protocol_mode',['A','D0'])
def test_final_only_runner_exports_one_model(monkeypatch,tmp_path,protocol_mode):
    torch.set_num_threads(1)
    monkeypatch.setattr(aligned,'load_target',lambda *a,**kw:(tiny_qwen35().requires_grad_(False),Processor()))
    data=tmp_path/'data.jsonl'
    data.write_text(json.dumps(dict(id='one',question='q',answer='a'))+'\n')
    cfg=tmp_path/'config.yaml'
    import yaml
    c=yaml.safe_load(CONFIG.read_text());c.update(device='cpu',num_epochs=2,batch_size=1,
        accumulation_steps=1,num_anchors=1,block_size=4,num_draft_layers=1,chs_num_layers=2,
        sequential_rank=8,target_layer_ids=[0,3],use_muon=False,pivot_two_rounds=protocol_mode=='D0',
        pivot_augmentation=dict(mode=protocol_mode,rank=8),rollout_anchors=1,final_only=True,
        report_to='none',save_interval=1)
    cfg.write_text(yaml.safe_dump(c))
    root=tmp_path/'out'
    aligned.run(['--config',str(cfg),'--train-data',str(data),'--output-dir',str(root),'--target-model',str(tmp_path)])
    assert (root/'last/config.json').is_file()
    assert not (root/'last').is_symlink()
    assert not list(root.glob('step_*'))
    assert not list(root.rglob('training_state_rank*'))
