"""Joint SFT/rollout, trajectory boundaries and paired-feedback regression tests."""
from copy import deepcopy
import os
import torch
import pytest
from litespec.core.outputs import TrainingOutput,RatioTerms
from litespec.vlm import train_aligned as aligned
from litespec.vlm.pivot_training import choose_transition,joint_objective
from litespec.vlm.pivot_augmentation import gather_positions
from litespec.vlm.sw_token import source_layers
from test_pivot_augmentation import draft_for,train_example
from test_pivot_protocols import config_for
from test_vlm import tiny_qwen35
from pivot_feedback_probe import collect_feedback,evaluate_feedback,paired_rows


def test_trajectory_transition_keeps_recorded_answer_and_logs_real_disagreement():
    ids=torch.arange(20)[None]
    anchors=torch.tensor([[4]])
    candidates=torch.tensor([[4,5,99,88]])
    predictions=torch.tensor([[5,77,66,55]])
    kept,nxt,bonus,corrected,metrics=choose_transition(ids,anchors,candidates,predictions)
    assert (kept.item(),nxt.item(),bonus.item())==(2,6,6)
    torch.testing.assert_close(corrected,ids,rtol=0,atol=0)
    assert metrics['trajectory/verifier_bonus_disagreement'].numerator.item()==1
    old=choose_transition(ids,anchors,candidates,predictions,'verifier')
    assert old[2].item()==77 and old[3][0,6].item()==77
    # Forced disagreement before any accepted proposal changes boundary too.
    different=choose_transition(ids,anchors,candidates,torch.tensor([[12,77,66,55]]))
    assert different[-1]['trajectory/verifier_kept_disagreement'].numerator.item()==1
    with pytest.raises(ValueError): choose_transition(ids,anchors,candidates,predictions,'unknown')


def test_joint_mean_preserves_additive_runner_gradient_and_zero_local_rollout():
    x=torch.tensor(4.,requires_grad=True);y=torch.tensor(9.,requires_grad=True)
    a=TrainingOutput(x/2,RatioTerms(x,torch.tensor(2.)))
    b=TrainingOutput(y/3,RatioTerms(y,torch.tensor(3.)))
    result=joint_objective(a,b,.25)
    expected=(x/2+.25*y/3)/1.25
    torch.testing.assert_close(result.loss,expected)
    actual_grads=torch.autograd.grad(result.loss_terms.numerator/result.loss_terms.denominator,(x,y),retain_graph=True)
    expected_grads=torch.autograd.grad(expected,(x,y))
    for g,h in zip(actual_grads,expected_grads): torch.testing.assert_close(g,h)


@pytest.mark.parametrize('mode',['D0','D1','D2'])
def test_joint_real_qwen_image_forward_backward_and_single_visual_encode(mode):
    torch.set_num_threads(1);torch.manual_seed(42)
    target=tiny_qwen35().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    draft=draft_for(mode)
    cfg=config_for(draft,True)
    cfg.update(training_protocol='joint_sft_rollout_v3',num_anchors=3,
        transition_policy='trajectory',rollout_loss_weight=.25)
    online=aligned.make_online(target,draft,cfg)
    calls=[];vision=[]
    h=target.register_forward_pre_hook(lambda m,a,k:calls.append(k['input_ids'].shape),with_kwargs=True)
    v=target.model.visual.register_forward_pre_hook(lambda m,a:vision.append(a[0].size(0)))
    kwargs=aligned.prepare_forward(target,online,train_example(target))
    assert kwargs['prepared_batch'].anchor_positions.size(1)==3
    from torch.distributed.utils import _cast_forward_inputs
    _,kwargs=_cast_forward_inputs(torch.float32,**kwargs)
    assert kwargs['prepared_batch'].sw_source is not None
    result=online(**kwargs)
    result.loss.backward()
    h.remove();v.remove()
    assert len(calls)==5 and vision==[4]
    assert calls[0]==torch.Size([1,20])
    assert calls[2:]==[torch.Size([2,3]),torch.Size([2,4]),torch.Size([2,7])]
    assert 'sft/loss' in result.ratio_metrics and 'rollout/loss' in result.ratio_metrics
    assert result.sum_metrics['rollout/visual_encoder_calls'].item()==0
    assert result.sum_metrics['sft/visual_encoder_calls'].item()==1
    assert result.sum_metrics['supervision/sft_positions'].item()>0
    assert all(p.grad is None for p in target.parameters())
    assert draft.sequential_head.output_proj.weight.grad.abs().sum()>0
    if mode!='D0': assert any(p.grad is not None and p.grad.abs().sum()>0 for p in draft.pivot_augmentation.parameters())
    # Both cached and full replay obey exactly the same explicit trajectory.
    online.zero_grad(set_to_none=True);online.rollout_backend='full_replay'
    torch.manual_seed(111);a=online(**kwargs)
    online.rollout_backend='cached';torch.manual_seed(111);b=online(**kwargs)
    torch.testing.assert_close(a.loss,b.loss,rtol=5e-4,atol=5e-5)


@pytest.mark.parametrize('mode',['D0','D1','D2'])
def test_paired_feedback_off_exact_and_cross_sample_shuffled(mode):
    torch.set_num_threads(1);torch.manual_seed(5)
    target=tiny_qwen35().requires_grad_(False)
    draft=draft_for(mode).eval()
    example=train_example(target)
    states=target(**example.inputs,use_cache=False,output_hidden_states=True).hidden_states
    pivot=gather_positions(source_layers(states,draft),torch.tensor([[9]]),draft.target_layer_ids)
    embedding=target.get_input_embeddings()
    probe=dict(sample_id='one',anchor=10,pivot=pivot,pending=example.inputs['input_ids'][:,10:11],reference=[5,6,7],
        data=dict(feedback=torch.randn_like(pivot),feedback_valid=torch.ones(1,dtype=torch.bool),
            wrong_embedding=embedding(torch.tensor([10])),correct_embedding=embedding(torch.tensor([11])),distance=torch.ones(1,dtype=torch.long)))
    donor=deepcopy({k:v for k,v in probe.items() if k!='data'})
    donor['sample_id']='two';donor['data']={**probe['data'],'feedback':torch.randn_like(pivot)}
    # Larger gate exposes wiring, rather than relying on a random token flip.
    if mode!='D0': draft.pivot_augmentation.gate.weight.data.fill_(1.)
    results={c:evaluate_feedback(draft,target,probe,c,donor) for c in ('on','off','shuffled')}
    rows=paired_rows(results)
    off=next(r for r in rows if r['condition']=='off')
    assert off['candidate_change']==0 and off['mean_abs_logit_delta']==0
    assert off['applied_residual_ratio']==0
    if mode=='D0':
        for r in rows: assert r['applied_residual_ratio']==0
    else:
        assert results['on']['applied_residual_ratio']>0
        assert not torch.equal(results['on']['logits'],results['shuffled']['logits'])
    assert 'inputs' not in draft.pivot_augmentation.__dict__ or callable(draft.pivot_augmentation.inputs)
    with pytest.raises(ValueError): evaluate_feedback(draft,target,probe,'shuffled',probe)


def test_collect_feedback_uses_real_verifier_and_clean_second_pivot():
    torch.set_num_threads(1);torch.manual_seed(6)
    target=tiny_qwen35().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    draft=draft_for('D1').eval()
    example=train_example(target)
    # Evaluation input is prompt only, as in production prepare_example(training=False).
    example.inputs={k:(v[:,:8] if k in ('input_ids','attention_mask','mm_token_type_ids') else v)
                    for k,v in example.inputs.items()}
    def bad(**kwargs):
        ids=torch.zeros(kwargs['draft_hidden'].size(0),draft.proposal_length,dtype=torch.long)
        return ids,None
    draft.sample_draft_tokens=bad
    tokens,probes=collect_feedback(target,draft,example,cap=12,stride=2,max_probes=3)
    assert len(tokens)==12 and probes
    for p in probes:
        assert p['reference']==tokens[p['offset']+1:p['offset']+1+draft.proposal_length]
        assert p['pending'].item()==tokens[p['offset']]
        clean=dict(example.inputs)
        prefix=torch.tensor(tokens[:p['offset']])[None]
        clean['input_ids']=torch.cat([clean['input_ids'],prefix],1)
        clean['attention_mask']=torch.ones_like(clean['input_ids'])
        clean['mm_token_type_ids']=torch.cat([clean['mm_token_type_ids'],torch.zeros_like(prefix)],1)
        s=target(**clean,use_cache=False,output_hidden_states=True).hidden_states
        expected=gather_positions(source_layers(s,draft),torch.tensor([[p['anchor']-1]]),draft.target_layer_ids)
        torch.testing.assert_close(p['pivot'],expected,rtol=5e-4,atol=3e-5)


def test_diagnostic_cli_outputs_complete_paired_report_without_updates(tmp_path,monkeypatch):
    import json
    import sys
    import diagnose_pivot_feedback as cli
    torch.set_num_threads(1);torch.manual_seed(15)
    target=tiny_qwen35().requires_grad_(False)
    target.config.text_config.eos_token_id=-1
    draft=draft_for('D1').eval()
    path=tmp_path/'checkpoint';draft.save_pretrained(path)
    monkeypatch.setattr(cli,'load_target',lambda *a,**k:(target,object()))
    monkeypatch.setattr(cli,'configure_image_tokens',lambda *a: {})
    monkeypatch.setattr(cli,'load_draft',lambda *a,**k:draft)
    packet=dict(source=str(tmp_path/'unused.jsonl'),serialized='\n'.join(json.dumps(dict(id=str(i))) for i in range(2))+'\n')
    monkeypatch.setattr(cli,'select_data',lambda *a:{'gsm8k':packet})
    def prepare(*args,**kwargs):
        example=train_example(target);example.sample_id=args[2]['id']
        example.inputs={k:(v[:,:8] if k in ('input_ids','attention_mask','mm_token_type_ids') else v)
                        for k,v in example.inputs.items()}
        return example
    monkeypatch.setattr(cli,'prepare_example',prepare)
    before={k:v.clone() for k,v in draft.state_dict().items()}
    root=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['probe','--target-model','tiny','--checkpoint','D1',str(path),
        '--output-dir',str(root),'--device','cpu','--samples','2','--max-new-tokens','12','--stride','2','--probes-per-sample','2'])
    cli.main()
    summary=json.loads((root/'summary.json').read_text())
    assert {r['condition'] for r in summary}=={'on','off','shuffled'}
    rows=[json.loads(x) for x in (root/'probes.jsonl').read_text().splitlines()]
    assert all(r['sample_id']!=r['donor'] for r in rows)
    assert (root/'diagnostics.md').is_file() and (root/'settings.json').is_file()
    for key,value in before.items(): torch.testing.assert_close(value,draft.state_dict()[key],rtol=0,atol=0)


def test_joint_two_epoch_runner_saves_only_last(monkeypatch,tmp_path):
    import json,yaml
    from test_vlm_aligned import CONFIG,Processor
    torch.set_num_threads(1)
    monkeypatch.setattr(aligned,'load_target',lambda *a,**k:(tiny_qwen35().requires_grad_(False),Processor()))
    data=tmp_path/'data.jsonl';data.write_text(json.dumps(dict(id='one',question='q',answer='a'))+'\n')
    c=yaml.safe_load(CONFIG.read_text());c.update(device='cpu',num_epochs=2,batch_size=1,
        accumulation_steps=1,num_anchors=2,block_size=4,num_draft_layers=1,chs_num_layers=2,
        sequential_rank=8,target_layer_ids=[0,3],use_muon=False,pivot_two_rounds=True,
        pivot_augmentation=dict(mode='D1',rank=8),rollout_anchors=1,final_only=True,
        training_protocol='joint_sft_rollout_v3',transition_policy='trajectory',rollout_loss_weight=.25,
        report_to='none',save_interval=1)
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(c));root=tmp_path/'out'
    aligned.run(['--config',str(cfg),'--train-data',str(data),'--output-dir',str(root),'--target-model',str(tmp_path)])
    saved=json.loads((root/'last/config.json').read_text())
    assert saved['vlm_config']['training_protocol']=='joint_sft_rollout_v3'
    assert saved['vlm_config']['training_recipe']['transition_policy']=='trajectory'
    assert not (root/'last').is_symlink()
    assert not list(root.glob('step_*')) and not list(root.rglob('training_state_rank*'))


def _distributed_joint_worker(rank,init_file):
    import torch.distributed as dist
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+init_file)
    try:
        for empty_rank in (False,True):
            x=torch.tensor(2.,requires_grad=True);y=torch.tensor(3.,requires_grad=True)
            sd=2. if rank==0 else 6.
            rd=3. if rank==0 else (0. if empty_rank else 1.)
            shared=TrainingOutput(x*x,RatioTerms(sd*x*x,torch.tensor(sd)))
            rollout=TrainingOutput(y*y,RatioTerms(rd*y*y,torch.tensor(rd)))
            result=joint_objective(shared,rollout,.25)
            result.loss_terms.numerator.backward()
            packed=torch.stack([x.grad,y.grad,result.loss_terms.denominator])
            dist.all_reduce(packed)
            gradient=packed[:2]/packed[2]
            torch.testing.assert_close(gradient,torch.tensor([4/1.25,.25*6/1.25]))
    finally: dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('RUN_GLOO_TESTS')!='1',reason='opt-in real sockets: RUN_GLOO_TESTS=1')
def test_two_rank_joint_denominator_and_empty_local_rollout(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_joint_worker,args=(str(tmp_path/'gloo_init'),),nprocs=2,join=True)


@pytest.mark.parametrize('empty_rank',[False,True])
def test_rank_global_balance_algebra_with_unequal_counts(monkeypatch,empty_rank):
    import torch.distributed as dist
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda group=None:2)
    sds=[2.,6.];rds=[3.,0. if empty_rank else 1.]
    scale=.25*sum(sds)/sum(rds)
    gradients=[];denominators=[];losses=[]
    for rank in range(2):
        def reduce(tensor,op=None,group=None):
            other=1-rank
            tensor.add_(tensor.new_tensor([sds[other],rds[other]]) if tensor.ndim else sds[other]+scale*rds[other])
        monkeypatch.setattr(dist,'all_reduce',reduce)
        x=torch.tensor(2.,requires_grad=True);y=torch.tensor(3.,requires_grad=True)
        a=TrainingOutput(x*x,RatioTerms(sds[rank]*x*x,torch.tensor(sds[rank])))
        b=TrainingOutput(y*y,RatioTerms(rds[rank]*y*y,torch.tensor(rds[rank])))
        result=joint_objective(a,b,.25)
        gradients.append(torch.stack(torch.autograd.grad(result.loss_terms.numerator,(x,y))))
        denominators.append(result.loss_terms.denominator);losses.append(result.loss.detach())
    torch.testing.assert_close(sum(gradients)/sum(denominators),torch.tensor([4/1.25,.25*6/1.25]))
    torch.testing.assert_close(sum(losses)/2,torch.tensor((4+.25*9)/1.25))
