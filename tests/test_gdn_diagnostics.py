"""Read-only diagnostics on actual tiny native-image target/draft checkpoints."""
from pathlib import Path

import pytest
import torch

import diagnose_gdn as diag
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.gdn_reader import cache_reader_states
from test_gdn_reader import draft_for
from test_vlm_aligned import fixture_model
from test_vlm_static_batch import example_for


@pytest.mark.parametrize('family',['dlite','dflash'])
def test_checkpoint_audit_load_and_zero_initialization(family,tmp_path):
    from safetensors.torch import load_file,save_file
    target,_,_,_=fixture_model()
    draft,_=draft_for(target,family,True)
    draft.save_pretrained(tmp_path)
    loaded=diag.load_draft(tmp_path,attn_implementation='sdpa')
    audit=diag.checkpoint_audit(tmp_path,loaded)
    assert audit['passed']
    assert all(r['exact_after_load_cast'] for r in audit['parameters'].values())
    up=next(r for k,r in audit['parameters'].items() if k.endswith('.up.weight'))
    assert up['nonzero']==0 and not up['changed_from_zero_initialization']
    assert audit['historical_reader_gradient_status']=='not_recorded_by_original_trainer'
    with torch.no_grad():diag.reader_of(loaded).up.weight.add_(.1)
    assert not diag.checkpoint_audit(tmp_path,loaded)['passed']
    loaded.save_pretrained(tmp_path)
    assert diag.checkpoint_audit(tmp_path,loaded)['passed']
    weights=load_file(tmp_path/'model.safetensors')
    weights.pop(next(k for k in weights if k.endswith('gdn_reader.up.weight')))
    save_file(weights,tmp_path/'model.safetensors')
    assert not diag.checkpoint_audit(tmp_path,loaded)['passed']


@pytest.mark.parametrize('family',['dlite','dflash'])
def test_modes_gradient_probe_and_actual_draft_proposal_parity(family):
    from litespec.vlm.baselines import DraftCache
    from litespec.vlm.context import ContextPolicy
    from litespec.vlm.runtime import draft_block,selected_layers
    target,_,_,_=fixture_model()
    draft,_=draft_for(target,family,True);draft.eval()
    torch.nn.init.normal_(diag.reader_of(draft).up.weight,std=.5)
    example=example_for(target,True)
    reference=[3,4,5,6,7,8]
    with RequestPositionState(target).activate():
        with torch.no_grad():
            output=target(**example.inputs,use_cache=True,output_hidden_states=True)
            states=tuple(s.clone() for s in cache_reader_states(draft,output.past_key_values))
            verified=target(input_ids=torch.tensor([reference[:4]]),past_key_values=output.past_key_values,use_cache=True)
            teacher=verified.logits[:,:3]
        wrong=tuple(s*3+1 for s in states)
        before={k:v.clone() for k,v in draft.state_dict().items()}
        normal,logits,candidates=diag.run_mode(target,draft,example,output,states,reference,0,'normal',teacher)
        off,off_logits,off_candidates=diag.run_mode(target,draft,example,output,states,reference,0,'off',teacher)
        altered,wrong_logits,wrong_candidates=diag.run_mode(target,draft,example,output,wrong,reference,0,'wrong',teacher)
        assert normal['residual']['relative_norm']>0
        assert off['residual']['relative_norm']==0
        assert not torch.equal(logits,off_logits) and not torch.equal(logits,wrong_logits)
        # Diagnostic proposals equal the existing production proposal boundary.
        with torch.no_grad():
            pending=torch.tensor([[reference[0]]])
            if family=='dflash':
                expected,_=DraftCache(draft,target,output,example.prefill_length).propose(
                    draft,target,pending,example.prefill_length,gdn_states=states)
            else:
                pivot=selected_layers(output.hidden_states,draft.target_layer_ids,draft.config.num_target_layers)[:,-1:]
                hidden,_=draft_block(draft,target.get_input_embeddings(),anchor_ids=pending,pivot=pivot,
                    history=None,anchor=example.prefill_length,prefill_length=example.prefill_length,
                    vision_mask=example.vision_mask,policy=ContextPolicy.no_history(),gdn_states=states)
                expected=draft.sample_draft_tokens(draft_hidden=hidden,first_prev_token_ids=pending[:,0],temperature=0.)[0]
        torch.testing.assert_close(expected,candidates)
        probe=diag.gradient_probe(target,draft,example,output,states,reference,0,teacher)
        assert not probe['optimizer_step_performed']
        assert all(r['connected'] and r['gradient']['finite'] for r in probe['parameters'].values())
        assert probe['parameters']['up.weight']['gradient']['norm']>0
        assert all(p.grad is None for p in draft.parameters())
        assert all(torch.equal(v,draft.state_dict()[k]) for k,v in before.items())
        assert all(p.grad is None for p in target.parameters())
        assert torch.equal(diag.run_mode(target,draft,example,output,states,reference,0,'normal',teacher)[1],logits)


def test_reader_mode_restores_after_exception_and_rejects_identical_donors():
    target,_,_,_=fixture_model();draft,_=draft_for(target,'dlite',True)
    reader=diag.reader_of(draft)
    assert 'forward' not in reader.__dict__
    with pytest.raises(RuntimeError):
        with diag.reader_mode(reader,'off'):raise RuntimeError('abort')
    assert 'forward' not in reader.__dict__
    with pytest.raises(ValueError):diag.donor_index([dict(image='same',question='a'),dict(image='same',question='b')],0)
    assert diag.donor_index([dict(question='a'),dict(question='b')],0)==1


def test_position_weighted_summary():
    fields=('ce_target_greedy','ce_reference_tokens','l1_to_target','fixed_reference_committed_length',
        'logits_mean_abs_change','logits_max_abs_change','kl_normal_to_mode',
        'teacher_forced_top1_change_fraction','free_running_candidate_change_fraction')
    rows=[]
    for i,n in enumerate([1,3]):
        rows.append(dict(model='dlite_gdn',dataset='vision',mode='off',sample_id=str(i),positions=n,
            residual=dict(relative_norm=0),**{k:float(i) for k in fields}))
    result=diag.summarize(rows)[0]
    assert result['ce_target_greedy']==.75 and result['fixed_reference_committed_length']==.5


def test_full_cli_real_image_target_and_two_checkpoint_report(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from test_vlm_aligned import Processor
    from PIL import Image
    target,_,_,_=fixture_model()
    target.config.text_config.eos_token_id=None
    processor=Processor()
    processor.image_processor=SimpleNamespace(size=dict(shortest_edge=16,longest_edge=65536))
    monkeypatch.setattr(diag,'load_target',lambda *a,**kw:(target,processor))
    run=tmp_path/'run'
    cfg=dict(target_model='tiny',image_max_tokens=1024)
    for family in ('dlite','dflash'):
        draft,_=draft_for(target,family,True)
        with torch.no_grad():diag.reader_of(draft).up.weight.add_(.1)
        path=run/(family+'_gdn')/'last'
        draft.save_pretrained(path)
        (path/'vlm_run.json').write_text(json.dumps(cfg))
        (path/'checkpoint.json').write_text(json.dumps(dict(cursor=dict(optimizer_step=2))))
    inputs=run/'evaluation/inputs';inputs.mkdir(parents=True)
    records=[]
    for index in range(2):
        image=tmp_path/f'image_{index}.png'
        Image.new('RGB',(4,4),(index*100,0,0)).save(image)
        records.append(dict(id=str(index),image=str(image),question=f'q{index}'))
    (inputs/'vision.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    output=tmp_path/'diagnostics'
    diag.main(['--run-group',str(run),'--output-dir',str(output),'--device','cpu',
        '--samples-per-dataset','2','--reference-tokens','6','--offsets','0','2','4'])
    rows=[json.loads(line) for line in (output/'anchors.jsonl').read_text().splitlines()]
    assert len(rows)==36
    assert all(r['sample_id']!=r['donor_sample_id'] for r in rows)
    assert all(r['residual']['relative_norm']==0 for r in rows if r['mode']=='off')
    assert all(r['logits_max_abs_change']==0 for r in rows if r['mode']=='normal')
    assert (output/'report.md').is_file() and len(json.loads((output/'summary.json').read_text()))==6
    audits=json.loads((output/'checkpoint_audit.json').read_text())
    assert all(a['passed'] for a in audits.values())
    assert all(a['historical_reader_gradient_status']=='not_recorded_by_original_trainer' for a in audits.values())
    assert len(json.loads((output/'gradient_probe.json').read_text()))==2
