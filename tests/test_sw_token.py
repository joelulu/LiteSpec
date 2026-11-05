from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from litespec.vlm.sw_token import RecentTargetWindow, gather_window, set_window
from litespec.vlm.runtime import draft_block, PreparedExample, selected_layers, visual_positions
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.context import ContextPolicy
from litespec.vlm import train_aligned as aligned
from test_vlm import tiny_qwen35, tiny_image_inputs, draft_config
from test_vlm_aligned import fixture_model, CONFIG
from litespec.modeling.draft.dlite import DLiteDraftModel


def test_window_order_padding_and_accepted_only():
    draft=set_window(DLiteDraftModel(draft_config()),5)
    def states(start,end):
        return tuple(torch.full((1,end-start,32),float(i*100))+torch.arange(start,end)[None,:,None] for i in range(5))
    window=RecentTargetWindow(draft,states(0,2),2)
    ctx,keep=window.context()
    assert keep.tolist()==[[False]*6+[True]*4]
    assert ctx.shape==(1,1,10,32)
    window.commit(states(2,6),2,2)
    assert window.committed==4
    assert window.values[0,:,0,0].tolist()==[100,101,102,103]  # selected layer 0 uses output offset 1
    poisoned=list(states(4,8));poisoned[1][:,2:]+=9999
    window.commit(poisoned,1,4)
    assert window.values[0,:,0,0].tolist()==[100,101,102,103,104]
    with pytest.raises(ValueError): window.commit(states(0,2),1,4)
    source={0:states(0,9)[1],3:states(0,9)[4]}
    context,valid=gather_window(source,torch.tensor([[2,5]]),[0,3],5)
    assert valid[0,0].tolist()==[False]*6+[True]*4
    assert context[0,1,:,0].tolist()==[100,400,101,401,102,402,103,403,104,404]


@pytest.mark.parametrize('k',[1,2,5,10])
@torch.no_grad()
def test_training_and_inference_same_prefix(k):
    target,_,_,example=fixture_model()
    draft=set_window(DLiteDraftModel(draft_config()),k).eval()
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,num_anchors=3)
    online=aligned.make_online(target,draft,cfg).eval()
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    anchors=torch.tensor([[2,8,11]])
    mask=torch.ones_like(example.inputs['input_ids'],dtype=torch.float32)
    batch=online.prepare_batch(example.inputs['input_ids'],states,mask,anchor_positions=anchors,block_keep_mask=torch.ones_like(anchors,dtype=torch.bool))
    expected=online.forward_backbone(batch,seq_len=example.inputs['input_ids'].size(1))
    for index,a in enumerate(anchors[0].tolist()):
        prefix=tuple(x[:,:a] for x in states)
        window=RecentTargetWindow(draft,prefix,a)
        context,keep=window.context()
        pending=example.inputs['input_ids'][:,a:a+1]
        actual,_=draft_block(draft,target.get_input_embeddings(),anchor_ids=pending,pivot=context,
            pivot_keep=keep,history=None,anchor=a,prefill_length=a,vision_mask=example.vision_mask,policy=ContextPolicy.no_history())
        torch.testing.assert_close(actual,expected[:,index],rtol=2e-4,atol=2e-5)


@pytest.mark.parametrize('k',[1,2,5,10])
@torch.no_grad()
def test_real_image_hybrid_static_window_decode(k):
    torch.set_num_threads(1);torch.manual_seed(7)
    target=tiny_qwen35().eval();target.config.text_config.eos_token_id=-1
    draft=set_window(DLiteDraftModel(draft_config()),k).eval()
    inputs=tiny_image_inputs()
    example=PreparedExample(inputs,inputs['input_ids'].size(1),visual_positions(inputs['input_ids'][0],target.config),'tiny')
    reference=generate_static_batch(target,None,example,max_new_tokens=9)
    result=generate_static_batch(target,draft,example,max_new_tokens=9,state_mode='per_token')
    assert result['token_ids']==reference['token_ids']
    assert result['context_window_size']==k
    assert result['context_slots']==k*2
    assert result['recent_feature_bytes']<=k*2*32*4


def test_chunked_checkpoint_gradient_and_future_isolation():
    target,_,_,example=fixture_model()
    draft=set_window(DLiteDraftModel(draft_config()),5)
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,num_anchors=3)
    online=aligned.make_online(target,draft,cfg)
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    anchors=torch.tensor([[2,8,11]])
    batch=online.prepare_batch(example.inputs['input_ids'],states,torch.ones_like(example.inputs['input_ids'],dtype=torch.float),anchor_positions=anchors,block_keep_mask=torch.ones_like(anchors,dtype=torch.bool))
    online.chunk_blocks=2
    a=online.forward_backbone(batch,seq_len=17)
    a.float().square().sum().backward()
    grads={n:p.grad.clone() for n,p in draft.named_parameters() if p.grad is not None}
    draft.zero_grad(set_to_none=True)
    online.chunk_blocks=256
    b=online.forward_backbone(batch,seq_len=17)
    b.float().square().sum().backward()
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    for n,p in draft.named_parameters():
        if n in grads: torch.testing.assert_close(grads[n],p.grad,rtol=3e-3,atol=1e-4)
    assert all(p.grad is None for p in target.parameters())
    online.eval()
    expected=online.forward_backbone(batch,seq_len=17)[:,:1]
    batch.sw_source={i:h.clone() for i,h in batch.sw_source.items()}
    for h in batch.sw_source.values(): h[:,2:]+=9999
    actual=online.forward_backbone(batch,seq_len=17)[:,:1]
    torch.testing.assert_close(expected,actual)


def test_one_position_has_no_parameter_or_output_change():
    target,legacy,_,example=fixture_model()
    updated=set_window(deepcopy(legacy),1)
    assert legacy.state_dict().keys()==updated.state_dict().keys()
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    pivot=selected_layers(states,legacy.target_layer_ids,4)[:,7:8]
    args=dict(anchor_ids=example.inputs['input_ids'][:,8:9],pivot=pivot,history=None,anchor=8,prefill_length=8,vision_mask=example.vision_mask,policy=ContextPolicy.no_history())
    a=draft_block(legacy,target.get_input_embeddings(),**args)[0]
    b=draft_block(updated,target.get_input_embeddings(),**args)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('k',[1,5])
def test_actual_warm_start_training_exports_weights_only(k,tmp_path,monkeypatch):
    import json,yaml
    from test_vlm_aligned import Processor
    from litespec.vlm.sw_token import load_warm_start,warm_start_config
    target,base,_,_=fixture_model()
    target_path=tmp_path/'target';target_path.mkdir()
    base.config.vlm_config=dict(target_model=str(target_path),context_policy=ContextPolicy.no_history().__dict__)
    parent=tmp_path/'parent';base.save_pretrained(parent)
    config=dict(upstream_config='scripts/config/dlite_sft.yaml',target_model=str(target_path),
        train_data=str(tmp_path/'train.jsonl'),output_dir=str(tmp_path/'run'),device='cpu',model_role='student',
        report_to='none',log_interval=1,save_interval=4000,num_epochs=1,batch_size=1,accumulation_steps=1,
        num_anchors=3,use_muon=False,context_window_size=k,sw_train_chunk_blocks=2,init_checkpoint=str(parent))
    warm_start_config(config,parent)
    loaded=load_warm_start(parent,target,config)
    assert loaded.context_window_size==k
    for key,value in base.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
    data=tmp_path/'train.jsonl';data.write_text('\n'.join(json.dumps(dict(id=str(i),question='describe',answer='answer')) for i in range(4)))
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(aligned,'load_target',lambda *a,**kw:(target,Processor()))
    aligned.run(['--config',str(path),'--max-steps','2'])
    last=tmp_path/'run/last'
    assert (last/'model.safetensors').is_file()
    assert json.loads((last/'config.json').read_text())['dlite_config']['context_window_size']==k
    assert not list((tmp_path/'run').rglob('training_state_rank_*'))
    exported=DLiteDraftModel.from_pretrained(last,attn_implementation='sdpa')
    assert any(not torch.equal(value,exported.state_dict()[key]) for key,value in base.state_dict().items())


def test_inference_overrides_are_saved_separately(tmp_path):
    draft=DLiteDraftModel(draft_config())
    draft.save_pretrained(tmp_path/'parent')
    loaded=DLiteDraftModel.from_pretrained(tmp_path/'parent',attn_implementation='sdpa')
    set_window(loaded,10)
    assert loaded.condition_slot_count==20
    import json
    assert json.loads((tmp_path/'parent/config.json').read_text())['dlite_config'].get('context_window_size',1)==1
    loaded.save_pretrained(tmp_path/'child')
    assert DLiteDraftModel.from_pretrained(tmp_path/'child',attn_implementation='sdpa').context_window_size==10


def test_chunked_one_position_control_matches_original_training():
    target,draft,original,example=fixture_model()
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,num_anchors=3,sw_chunked_training=True)
    chunked=aligned.make_online(target,draft,cfg)
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    anchors=torch.tensor([[2,8,11]])
    kwargs=dict(anchor_positions=anchors,block_keep_mask=torch.ones_like(anchors,dtype=torch.bool))
    mask=torch.ones_like(example.inputs['input_ids'],dtype=torch.float)
    before=original.prepare_batch(example.inputs['input_ids'],states,mask,**kwargs)
    after=chunked.prepare_batch(example.inputs['input_ids'],states,mask,**kwargs)
    a=original.forward_backbone(before,seq_len=17)
    b=chunked.forward_backbone(after,seq_len=17)
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)



@pytest.mark.parametrize('k',[1,2])
def test_fsdp_input_traversal_preserves_window_source_and_backward(k):
    """Exercise the real FSDP dataclass cast that dropped ad-hoc attributes."""
    from dataclasses import fields, replace
    from torch.distributed.utils import _cast_forward_inputs
    from torch.distributed.utils import _apply_to_tensors
    target,_,_,example=fixture_model()
    draft=set_window(DLiteDraftModel(draft_config()),k)
    cfg=aligned.load_config(CONFIG)
    cfg.update(block_size=4,num_anchors=3,sw_chunked_training=True)
    online=aligned.make_online(target,draft,cfg)
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    anchors=torch.tensor([[2,8,11]])
    batch=online.prepare_batch(example.inputs['input_ids'],states,
        torch.ones_like(example.inputs['input_ids'],dtype=torch.float32),
        anchor_positions=anchors,block_keep_mask=torch.ones_like(anchors,dtype=torch.bool))
    assert 'sw_source' in {f.name for f in fields(batch)}
    assert replace(batch).sw_source is batch.sw_source
    moved=_apply_to_tensors(lambda x:x.clone(),batch)
    assert moved.sw_source is not None
    assert moved.sw_source.keys()==batch.sw_source.keys()
    for layer in moved.sw_source:
        torch.testing.assert_close(moved.sw_source[layer],batch.sw_source[layer])
        assert not moved.sw_source[layer].requires_grad
    # This is the helper used by FSDP's mixed-precision forward pre-hook.
    args,kwargs=_cast_forward_inputs(torch.bfloat16,prepared_batch=moved)
    assert not args
    converted=kwargs['prepared_batch']
    assert converted.sw_source is not None
    assert all(x.dtype==torch.bfloat16 for x in converted.sw_source.values())
    assert converted.query_embeddings.dtype==torch.bfloat16
    assert converted.anchor_positions.dtype==torch.long
    # Forward/backward after a transport round trip, including chunk recomputation.
    _,kwargs=_cast_forward_inputs(torch.float32,prepared_batch=moved)
    online.chunk_blocks=2
    output=online.forward_backbone(kwargs['prepared_batch'],seq_len=17)
    output.float().square().sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in draft.parameters())
    assert all(p.grad is None for p in target.parameters())


def test_missing_window_source_has_actionable_error():
    target,_,_,example=fixture_model()
    draft=set_window(DLiteDraftModel(draft_config()),2)
    cfg=aligned.load_config(CONFIG);cfg.update(block_size=4,num_anchors=3)
    online=aligned.make_online(target,draft,cfg)
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    batch=online.prepare_batch(example.inputs['input_ids'],states,
        torch.ones_like(example.inputs['input_ids'],dtype=torch.float32))
    batch.sw_source=None
    with pytest.raises(ValueError,match='prepared sw_source'):
        online.forward_backbone(batch,seq_len=17)
