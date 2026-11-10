import copy
from types import SimpleNamespace
import pytest
import torch
from transformers import Qwen3Config
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.attention_masks import create_dflash_sdpa_mask
from litespec.vlm.baselines import DraftCache, load_draft
from litespec.vlm.dflash_context import intervene_visual, propose_features


def make(mode):
    c=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,layer_types=['full_attention'])
    c.architectures=['DFlashDraftModel']; c.num_target_layers=32; c.block_size=4
    c.dflash_config=dict(target_layer_ids=list(range(12 if mode=='pivot12' else 5)),mask_token_id=127)
    c.vlm_config=dict(draft_context=mode); c._attn_implementation='sdpa'
    return DFlashDraftModel(c)


@pytest.mark.parametrize('mode,window,slots', [('full',None,1),('recent128',128,1),('pivot1',1,1),('pivot5',1,5),('pivot12',1,12)])
def test_train_export_and_cache_matches_uncached(mode,window,slots,tmp_path):
    torch.manual_seed(42)
    draft=make(mode)
    target=SimpleNamespace(get_input_embeddings=lambda:emb, get_output_embeddings=lambda:head)
    emb=torch.nn.Embedding(128,32).requires_grad_(False)
    head=torch.nn.Linear(32,128,bias=False).requires_grad_(False)
    online=OnlineDFlashModel(draft,head,emb,127,block_size=4,num_anchors=3,attention_backend='sdpa',selector_loss_alpha=0)
    ids=torch.randint(0,100,(2,14)); features=torch.randn(2,14,len(draft.target_layer_ids)*32)
    loss=online(input_ids=ids,hidden_states=features,loss_mask=torch.ones_like(ids).float()).loss
    assert torch.isfinite(loss); loss.backward()
    assert draft.fc.weight.grad.abs().sum()>0
    if slots>1: assert draft.context_depth.weight.grad.abs().sum()>0
    draft.eval(); draft.save_pretrained(tmp_path)
    draft=load_draft(tmp_path,attn_implementation='sdpa').eval()
    assert draft.context_window==window and draft.context_slots==slots
    states=tuple(torch.randn(1,140,32) for _ in range(33))
    prefill=SimpleNamespace(hidden_states=states)
    cache=DraftCache(draft,target,prefill,140)
    pending=torch.tensor([[4]])
    all_features=cache.features(draft,prefill)
    for _ in range(3):
        actual=cache.propose(draft,target,pending,all_features.size(1))[0]
        expected=propose_features(draft,target,all_features,pending)[0]
        assert torch.equal(actual,expected)
        # Only accepted verifier inputs are appended; rejects are never retained.
        output=SimpleNamespace(hidden_states=tuple(torch.randn(1,4,32) for _ in range(33)))
        all_features=torch.cat([all_features,cache.features(draft,output)[:,:2]],1)
        cache.commit(draft,output,2)
        if window is not None: assert cache.hidden.size(1)<=window


def test_mask_visibility_no_future_and_same_query_block():
    anchors=torch.tensor([[3,7]])
    for slots,window in [(1,None),(1,128),(1,1),(5,1),(12,1)]:
        mask=create_dflash_sdpa_mask(anchors,torch.tensor([[True,True]]),8,4,'cpu',context_window=window,context_slots=slots)[0,0]
        for q in range(8):
            a=int(anchors[0,q//4]); left=0 if window is None else max(0,a-window)
            expected=list(range(left*slots,a*slots))+list(range(8*slots+(q//4)*4,8*slots+(q//4+1)*4))
            assert mask[q].nonzero().flatten().tolist()==expected


def test_visual_interventions_preserve_text_and_positions():
    f=torch.randn(1,7,20); mask=torch.tensor([False,True,True,False,False,False,False])
    donor=torch.randn(1,3,20)
    for condition in ('correct','wrong','mask'):
        out,keep,meta=intervene_visual(f,mask,condition,donor)
        assert torch.equal(out[:,~mask],f[:,~mask])
        assert torch.equal(f,intervene_visual(f,mask,'correct')[0])
        assert meta['visual_slots']==2
        if condition=='mask': assert torch.equal(keep[0],~mask)
        if condition=='wrong':
            assert torch.equal(out[0,mask],donor[0,[0,2]])
            assert keep.all()
    with pytest.raises(ValueError): intervene_visual(f,mask,'wrong')


@pytest.mark.parametrize('mode',['full','recent128','pivot1','pivot5','pivot12'])
def test_real_hybrid_static_decode_and_visual_intervention(mode):
    from test_vlm import tiny_qwen35,tiny_image_inputs
    from test_vlm_aligned import Processor
    from litespec.vlm import baselines
    from litespec.vlm.runtime import PreparedExample
    from litespec.vlm.context import visual_positions
    from litespec.vlm.static_batch import generate_static_batch
    from litespec.vlm.benchmark import RequestPositionState
    torch.manual_seed(7)
    target=tiny_qwen35().requires_grad_(False)
    config=dict(algorithm='dflash',draft_context=mode,num_draft_layers=1,block_size=4,
        target_model='tiny',context_policy=dict(prefill_text='keep',prefill_vision='keep',decode_mode='all'))
    if mode=='pivot5':config['target_layer_ids']=[0,1,2,3,3]
    if mode=='pivot12':config['target_layer_ids']=[0,1,2,3]*3
    draft=baselines.make_draft(target,Processor(),config).eval()
    inputs=tiny_image_inputs()
    ex=PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),'image')
    with RequestPositionState(target).activate():
        result=generate_static_batch(target,draft,ex,max_new_tokens=7,state_mode='per_token')
    assert len(result['token_ids'])>0 and result['recovery_replays']==0
    if mode=='full':
        for condition in ['wrong','mask']:
            donor=torch.randn(1,int(ex.vision_mask.sum()),len(draft.target_layer_ids)*32)
            with RequestPositionState(target).activate():
                changed=generate_static_batch(target,draft,ex,max_new_tokens=7,state_mode='per_token',
                    visual_condition=condition,donor_features=donor)
            assert changed['visual_intervention']['condition']==condition
            assert changed['visual_intervention']['visual_slots']>0


@pytest.mark.parametrize('mode',['full','recent128','pivot1','pivot5','pivot12'])
def test_training_block_hidden_equals_same_prefix_inference(mode):
    torch.manual_seed(19)
    d=make(mode).eval()
    emb=torch.nn.Embedding(128,32).requires_grad_(False)
    head=torch.nn.Linear(32,128,bias=False).requires_grad_(False)
    online=OnlineDFlashModel(d,head,emb,127,block_size=4,num_anchors=2,attention_backend='sdpa',selector_loss_alpha=0)
    ids=torch.randint(0,90,(1,12)); features=torch.randn(1,12,len(d.target_layer_ids)*32)
    anchors=torch.tensor([[4,8]]);keep=torch.ones_like(anchors,dtype=torch.bool)
    online._sample_anchor_positions=lambda *a,**kw:(anchors,keep)
    _,_,training=online._forward_draft_blocks(ids,features,torch.ones_like(ids).float())
    for block,anchor in enumerate([4,8]):
        left=0 if d.context_window is None else max(0,anchor-d.context_window)
        context=features[:,left:anchor]
        query=ids.new_full((1,4),127);query[:,0]=ids[:,anchor]
        pos=torch.cat([torch.arange(left,anchor).repeat_interleave(d.context_slots),torch.arange(anchor,anchor+4)])[None]
        inference=d(position_ids=pos,target_hidden=context,noise_embedding=emb(query),is_causal=False)
        torch.testing.assert_close(training[:,block*4:(block+1)*4],inference,atol=1e-6,rtol=1e-5)


def test_flex_mask_rule_matches_dense(monkeypatch):
    from litespec.core import attention_masks as m
    captured={}
    def capture(rule,**kw): captured.update(rule=rule,**kw); return captured
    monkeypatch.setattr(m,'FLEX_ATTENTION_AVAILABLE',True)
    monkeypatch.setattr(m,'create_flex_block_mask',capture)
    for slots,window in [(1,None),(1,128),(1,1),(5,1),(12,1)]:
        anchors=torch.tensor([[3,7]]);keep=torch.tensor([[True,False]])
        m.create_dflash_flex_mask(anchors,keep,8,4,'cpu',context_window=window,context_slots=slots)
        q=torch.arange(8)[:,None];kv=torch.arange(captured['KV_LEN'])[None,:]
        actual=captured['rule'](0,0,q,kv)
        expected=m.create_dflash_sdpa_mask(anchors,keep,8,4,'cpu',context_window=window,context_slots=slots)[0,0]
        assert torch.equal(actual,expected)


def test_visual_probe_cli_with_real_tiny_target(tmp_path,monkeypatch):
    import json
    import sys
    import probe_dflash_visual as cli
    from test_vlm import tiny_qwen35,tiny_image_inputs
    from test_vlm_aligned import Processor
    from litespec.vlm.baselines import make_draft
    from litespec.vlm.runtime import PreparedExample
    from litespec.vlm.context import visual_positions
    target=tiny_qwen35().requires_grad_(False);target.config.text_config.eos_token_id=None
    d=make_draft(target,Processor(),dict(algorithm='dflash',num_draft_layers=1,block_size=4,target_model='tiny',context_policy={}))
    path=tmp_path/'model';d.save_pretrained(path)
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,Processor()))
    monkeypatch.setattr(cli,'configure_image_tokens',lambda *a: {})
    monkeypatch.setattr(cli,'select_data',lambda args:{'tiny':dict(source=str(tmp_path/'data.jsonl'),serialized='{"id":"a"}\n{"id":"b"}\n',samples=2)})
    def prepare(processor,config,record,**kw):
        inputs=tiny_image_inputs()
        if record['id']=='b':inputs['pixel_values']=inputs['pixel_values']+1
        return PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),record['id'])
    monkeypatch.setattr(cli,'prepare_example',prepare)
    result=tmp_path/'probe'
    monkeypatch.setattr(sys,'argv',['probe','--target-model','tiny','--checkpoint','new',str(path),
        '--checkpoint','old',str(path),'--output-dir',str(result),'--samples','2',
        '--max-new-tokens','8','--probes-per-sample','2','--stride','2','--device','cpu'])
    cli.main()
    rows=json.loads((result/'summary.json').read_text())
    assert len(rows)==6 and all(r['probes']==4 for r in rows)
    assert all(r['candidate_change']==0 for r in rows if r['condition']=='correct')
    assert (result/'diagnostics.md').is_file()


def test_benchmark_cli_wrong_and_mask_reuse_ar(tmp_path,monkeypatch):
    import json
    import sys
    import benchmark_vlm_static_batch as cli
    from test_vlm import tiny_qwen35,tiny_image_inputs
    from test_vlm_aligned import Processor
    from litespec.vlm.baselines import make_draft
    from litespec.vlm.runtime import PreparedExample
    from litespec.vlm.context import visual_positions
    torch.manual_seed(21)
    target=tiny_qwen35().requires_grad_(False)
    draft=make_draft(target,Processor(),dict(algorithm='dflash',num_draft_layers=1,block_size=4,target_model='tiny',context_policy={}))
    checkpoint=tmp_path/'checkpoint';draft.save_pretrained(checkpoint)
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,Processor()))
    monkeypatch.setattr(cli,'read_eval_records',lambda *a:[dict(id='a'),dict(id='b')])
    def prepare(processor,config,record,**kw):
        inputs=tiny_image_inputs()
        if record['id']=='b':inputs['pixel_values']=inputs['pixel_values']+1
        return PreparedExample(inputs,8,visual_positions(inputs['input_ids'][0],target.config),record['id'])
    monkeypatch.setattr(cli,'prepare_example',prepare)
    for condition in ('correct','wrong','mask'):
        argv=['bench','--target-model','tiny','--checkpoint','full',str(checkpoint),
            '--data',str(tmp_path/'data.jsonl'),'--output-dir',str(tmp_path/condition),'--max-samples','2',
            '--batch-sizes','1','--max-new-tokens','7','--warmup','0','--device','cpu',
            '--attention-backend','sdpa','--state-mode','per_token','--visual-condition',condition]
        if condition!='correct':argv+=['--ar-reference',str(tmp_path/'correct')]
        monkeypatch.setattr(sys,'argv',argv)
        cli.main()
        rows=json.loads((tmp_path/condition/'summary.json').read_text())
        assert len(rows)==2 and rows[1]['visual_condition']==condition
        assert (tmp_path/condition/'summary.csv').is_file()


def test_flex_callback_has_no_mutation_after_functionalization(monkeypatch):
    """CUDA lowering rejects copy_ introduced by functionalizing &= in mask_mod."""
    import ast, inspect
    from torch.fx.experimental.proxy_tensor import make_fx
    from litespec.core import attention_masks as m
    assert not any(isinstance(node,ast.AugAssign) for node in ast.walk(ast.parse(inspect.getsource(m.create_dflash_flex_mask))))
    captured={}
    def capture(rule,**kw): captured['rule']=rule
    monkeypatch.setattr(m,'FLEX_ATTENTION_AVAILABLE',True)
    monkeypatch.setattr(m,'create_flex_block_mask',capture)
    for slots,window,sliding in [(1,128,None),(1,1,None),(5,1,None),(12,1,None),(1,None,4)]:
        m.create_dflash_flex_mask(torch.tensor([[3,7]]),torch.tensor([[True,True]]),8,4,'cpu',
            context_window=window,context_slots=slots,sliding_window=sliding)
        rule=captured['rule']
        callback=lambda q,k: rule(0,0,q,k)
        graph=make_fx(torch.func.functionalize(callback))(torch.tensor([2]),torch.tensor([1]))
        mutations=[n.target for n in graph.graph.nodes if n.op=='call_function' and
                   getattr(getattr(n.target,'_schema',None),'is_mutable',False)]
        assert mutations==[], mutations


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA FlexAttention')
@pytest.mark.parametrize('slots,window',[(1,128),(1,1),(5,1),(12,1)])
def test_cuda_flex_context_forward_backward(slots,window):
    from litespec.core.attention_masks import create_dflash_flex_mask
    from litespec.modeling.draft.flex_attention import compile_friendly_flex_attention
    q=torch.randn(1,2,8,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(1,1,17*slots+8,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    mask=create_dflash_flex_mask(torch.tensor([[8,13]],device='cuda'),torch.ones(1,2,device='cuda',dtype=torch.bool),
        17,4,'cuda',context_window=window,context_slots=slots)
    result=compile_friendly_flex_attention(q,k,v,block_mask=mask,enable_gqa=True)
    result.float().square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q,k,v))


def test_pivot5_uses_pivot1_sources_and_pivot12_slot_projector():
    from test_vlm import tiny_qwen35
    from test_vlm_aligned import Processor
    from litespec.vlm.baselines import make_draft
    target=tiny_qwen35()
    target.config.text_config.num_hidden_layers=32
    config=dict(algorithm='dflash',num_draft_layers=5,block_size=8,target_model='tiny',context_policy={})
    one=make_draft(target,Processor(),dict(config,draft_context='pivot1'))
    five=make_draft(target,Processor(),dict(config,draft_context='pivot5'))
    twelve=make_draft(target,Processor(),dict(config,draft_context='pivot12'))
    assert one.target_layer_ids==five.target_layer_ids==[1,8,15,22,29]
    assert five.context_slots==5 and five.context_window==1 and five.block_size==8
    assert five.fc.in_features==twelve.fc.in_features==32
    assert one.fc.in_features==5*32
    assert five.context_depth.weight.shape==(5,32)
    with pytest.raises(ValueError,match='exactly 5'):
        make_draft(target,Processor(),dict(config,draft_context='pivot5',target_layer_ids=[1,8]))


def test_pivot5_eval_optional_and_external_checkpoint(tmp_path):
    from evaluate_dflash_context import context_checkpoints
    old=context_checkpoints(tmp_path, {})
    assert list(old)==['full','recent128','pivot1','pivot12']
    path=tmp_path/'pivot5'/'last';path.mkdir(parents=True);(path/'config.json').write_text('{}')
    assert list(context_checkpoints(tmp_path,{}))==['full','recent128','pivot1','pivot5','pivot12']
    override=tmp_path/'another'/'last'
    assert context_checkpoints(tmp_path,{'PIVOT5_CHECKPOINT':str(override)})['pivot5']==override
    assert list(context_checkpoints(tmp_path,{'EVAL_CONTEXTS':'full pivot1 pivot5'}))==['full','pivot1','pivot5']


def test_pivot5_configuration_keeps_matched_budget(monkeypatch):
    from dflash_context_config import build
    monkeypatch.setenv('EPOCHS','1');monkeypatch.setenv('MICRO_BATCH_SIZE','4');monkeypatch.setenv('ACCUMULATION_STEPS','2')
    config=build('pivot5')
    assert config['block_size']==8 and config['num_epochs']==1
    assert config['batch_size']==4 and config['accumulation_steps']==2
    assert config['draft_context']=='pivot5' and config['final_only']


def test_video_probe_selects_frozen_mixed_packets(monkeypatch,tmp_path):
    import json
    import probe_dflash_visual as cli
    import video_benchmark_data as video
    calls=[]
    def images(args):
        assert args.tests==['textvqa']
        return dict(textvqa=dict(source='image.jsonl',serialized='{"id":"im"}\n'))
    def videos(root,names,count,seed):
        calls.append((root,names,count,seed))
        return {name:[dict(id=f'{name}:a',video='a.mp4'),dict(id=f'{name}:b',video='b.mp4')] for name in names}
    monkeypatch.setattr(cli,'select_data',images)
    monkeypatch.setattr(video,'select_data',videos)
    packets=cli.select_packets(SimpleNamespace(tests=['textvqa','Video-MME','VideoDetailCaption'],samples=50,seed=42,video_data_root=tmp_path))
    assert list(packets)==['textvqa','Video-MME','VideoDetailCaption']
    assert calls==[(tmp_path,['Video-MME','VideoDetailCaption'],50,42)]
    assert len(packets['Video-MME']['serialized'].splitlines())==2


def test_video_probe_preparation_and_real_target_collection(monkeypatch):
    import probe_dflash_visual as cli
    import video_runtime
    from test_vlm import tiny_qwen35,tiny_image_inputs
    from test_vlm_aligned import Processor
    from litespec.vlm.baselines import make_draft
    from litespec.vlm.runtime import PreparedExample
    target=tiny_qwen35().requires_grad_(False)
    target.config.video_token_id=93
    target.config.text_config.eos_token_id=None
    inputs=tiny_image_inputs()
    inputs['input_ids'][inputs['input_ids']==91]=93
    inputs['mm_token_type_ids']*=2
    inputs['pixel_values_videos']=inputs.pop('pixel_values')
    inputs['video_grid_thw']=inputs.pop('image_grid_thw')
    initial=PreparedExample(inputs,8,inputs['input_ids'][0]==93,'video:1')
    def prepare(processor,config,record,frames,max_pixels):
        assert frames==16 and max_pixels==200704
        return initial,dict(actual_frames=frames,video_tokens=4)
    monkeypatch.setattr(video_runtime,'prepare_video',prepare)
    args=SimpleNamespace(device='cpu',frames=16,max_pixels_per_frame=200704)
    example,metadata,fingerprint=cli.prepare_probe_input(Processor(),target.config,dict(video='a.mp4'),args,'source')
    assert int(example.vision_mask.sum())==6  # includes both delimiters
    assert metadata['modality']=='video' and metadata['intervention_slots']==6
    assert len(fingerprint)==64
    draft=make_draft(target,Processor(),dict(algorithm='dflash',num_draft_layers=1,block_size=4,target_model='tiny',context_policy={}))
    item=cli.collect(target,draft,example,8)
    assert len(item['tokens'])==8 and item['features'].shape[1]==15
