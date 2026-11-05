from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch

from test_vlm_aligned import fixture_model, Processor
from litespec.vlm.sw_token import RecentTargetWindow, set_window
from sw_token_probe import (AttentionMass, observe_attention, run_condition,
                            compare_conditions, collect_ar_prefixes, CONDITIONS)


def fixed_probe(target,draft,example):
    output=target(**example.inputs,use_cache=False,output_hidden_states=True)
    set_window(draft,2)
    source=tuple(x[:,:8] for x in output.hidden_states)
    window=RecentTargetWindow(draft,source,8)
    context,keep=window.context()
    set_window(draft,1)
    return dict(context=context,keep=keep,pending=example.inputs['input_ids'][:,8:9],
                anchor=8,offset=0,reference=[5,6,7])


@torch.no_grad()
def test_masked_old_slots_recover_k1_and_observer_is_passive():
    target,draft,_,example=fixture_model()
    draft.eval();probe=fixed_probe(target,draft,example)
    original=deepcopy(draft.config.dlite_config)
    base=run_condition(draft,target.get_input_embeddings(),probe,'k1')
    again=run_condition(draft,target.get_input_embeddings(),probe,'k1',capture=False)
    torch.testing.assert_close(base['hidden'],again['hidden'],rtol=0,atol=0)
    for name in ('k2_pivot_only','k2_pivot_only_origin0'):
        result=run_condition(draft,target.get_input_embeddings(),probe,name)
        torch.testing.assert_close(base['hidden'],result['hidden'],rtol=1e-5,atol=1e-6)
        assert compare_conditions(base,result,atol=1e-5,rtol=1e-4)['first_proposal_equal']
        assert all(abs(a['old_mass'])<1e-8 for a in result['attention'])
    poisoned={**probe,'context':probe['context'].clone()}
    poisoned['context'][:,:,:draft.chs_num_layers]+=1000
    restored=run_condition(draft,target.get_input_embeddings(),poisoned,'k2_pivot_only_origin0')
    torch.testing.assert_close(base['hidden'],restored['hidden'],rtol=1e-5,atol=1e-6)
    assert draft.context_window_size==1
    assert draft.config.dlite_config==original
    assert all(not hasattr(layer.self_attn,'_sw_attention_observer') for layer in draft.layers)


@torch.no_grad()
def test_attention_mass_matches_manual_grouped_softmax():
    from types import SimpleNamespace
    torch.manual_seed(3)
    q=torch.randn(1,4,3,2);k=torch.randn(1,2,7,2)
    module=SimpleNamespace(num_key_value_groups=2,scaling=2**-0.5,sliding_window=None,layer_idx=0)
    keep=torch.tensor([[False,True,True,True]])
    mask=torch.zeros(1,1,3,7);mask[:,:,:,0]=torch.finfo(torch.float32).min
    obs=AttentionMass(2,keep);obs(module,q,k,mask,4)
    w=((q@k.repeat_interleave(2,1).transpose(-1,-2))*module.scaling+mask).softmax(-1)[0].mean(0)
    for i,row in enumerate(obs.rows):
        assert row['old_mass']==pytest.approx(float(w[i,:2].sum()))
        assert row['pivot_mass']==pytest.approx(float(w[i,2:4].sum()))
        assert row['block_mass']==pytest.approx(float(w[i,4:].sum()))
        assert row['mass_sum']==pytest.approx(1.)
        assert row['old_valid_slots']==1
        assert row['pivot_valid_slots']==2


def test_observer_is_restored_on_failure():
    target,draft,_,example=fixture_model()
    prior=lambda *args:None
    draft.layers[0].self_attn._sw_attention_observer=prior
    with pytest.raises(RuntimeError):
        with observe_attention(draft,object()):
            raise RuntimeError('probe failed')
    assert draft.layers[0].self_attn._sw_attention_observer is prior
    assert all(not hasattr(layer.self_attn,'_sw_attention_observer') for layer in draft.layers[1:])


@torch.no_grad()
def test_ar_prefixes_match_reference_and_candidates_never_enter_target():
    from litespec.vlm.benchmark import RequestPositionState
    from litespec.vlm.static_batch import generate_static_batch
    target,draft,_,example=fixture_model()
    # Use prompt only, with the original multimodal type/mask prefix.
    example.inputs={k:(v[:,:8] if torch.is_tensor(v) and v.ndim==2 and v.shape[1]==17 else v) for k,v in example.inputs.items()}
    target.eval();target.config.text_config.eos_token_id=-1
    with RequestPositionState(target).activate():
        expected=generate_static_batch(target,None,example,max_new_tokens=9)['token_ids']
    with RequestPositionState(target).activate():
        tokens,probes=collect_ar_prefixes(target,draft,example,cap=9,stride=2,max_probes=4)
    assert tokens==expected
    assert [p['offset'] for p in probes]==[0,2,4,6]
    for probe in probes:
        assert int(probe['pending'])==tokens[probe['offset']]
        assert probe['reference']==tokens[probe['offset']+1:probe['offset']+1+draft.proposal_length]
    assert draft.context_window_size==1


def test_summary_fails_closed_for_empty_or_bad_alignment():
    from diagnose_sw_token import summarize
    assert not summarize([],[],expected_probes=0,tolerances={})['alignment_passed']
    rows=[]
    for name in CONDITIONS:
        rows.append(dict(condition=name,hidden_close=True,logits_close=True,first_proposal_equal=True,
            hidden_max_abs=0.,logits_max_abs=0.,first_hit=True,accepted=2,ce=1.,candidate_count=3,
            first_repair=False,first_break=False,proposal_agreement=1.))
    assert summarize(rows,[],expected_probes=1,tolerances={})['alignment_passed']
    rows[-1]['first_proposal_equal']=False
    assert not summarize(rows,[],expected_probes=1,tolerances={})['alignment_passed']


def test_actual_diagnostic_cli_exports_prefixes_attention_and_gate(tmp_path,monkeypatch):
    import diagnose_sw_token as cli
    target,draft,_,_=fixture_model();target.eval();target.config.text_config.eos_token_id=-1
    parent=tmp_path/'parent';draft.save_pretrained(parent)
    processor=Processor()
    processor.image_processor=type('Image',(),{'size':dict(shortest_edge=16,longest_edge=1024)})()
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,processor))
    monkeypatch.setattr(cli,'select_data',lambda args:{'textvqa':dict(source='fixture',serialized=json.dumps(dict(id='one',question='what?'))+'\n')})
    output=tmp_path/'output'
    cli.main(['--workspace',str(tmp_path),'--checkpoint',str(parent),'--target-model','tiny',
        '--output-dir',str(output),'--device','cpu','--samples','1','--max-new-tokens','9',
        '--stride','1','--probes-per-sample','3'])
    summary=json.loads((output/'summary.json').read_text())
    assert summary['alignment_passed']
    assert summary['expected_probes']==3
    assert len(summary['conditions'])==4
    assert (output/'diagnostics.md').is_file()
    rows=[json.loads(s) for s in (output/'attention.jsonl').read_text().splitlines()]
    assert all(abs(r['mass_sum']-1.)<1e-6 for r in rows)
    assert set(r['condition'] for r in rows)==set(CONDITIONS)
    assert not list(parent.glob('training_state_rank_*'))


def test_training_pair_stops_before_training_if_gate_fails(tmp_path):
    import os,subprocess,sys
    root=tmp_path/'diagnostic';root.mkdir()
    (root/'summary.json').write_text(json.dumps(dict(schema='sw_textvqa_probe_v1',alignment_passed=False)))
    destination=tmp_path/'must_not_be_created'
    env=dict(os.environ,DIAGNOSTICS_DIR=str(root),PYTHON_BIN=sys.executable,RUN_GROUP=str(destination))
    result=subprocess.run(['bash',str(Path(__file__).with_name('train_sw_token_pair.sh'))],env=env,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'Alignment must PASS' in result.stderr
    assert not destination.exists()


def test_training_pair_routes_two_windows_and_exact_evaluation_manifest(tmp_path):
    import os,subprocess,sys,shutil
    # Isolated shell fixture: real gate, fake trainer/evaluator, no GPU or updates.
    repo=tmp_path/'repo';tests=repo/'tests';tests.mkdir(parents=True)
    for name in ('train_sw_token_pair.sh',):
        shutil.copy(Path(__file__).with_name(name),tests/name)
    (tests/'check_sw_token_source.py').write_text('pass\n')
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'config.json').write_text('{}')
    (parent/'model.safetensors').write_bytes(b'fixture')
    target=tmp_path/'target';target.mkdir()
    (target/'config.json').write_text(json.dumps(dict(vision_config=dict(patch_size=16,spatial_merge_size=2))))
    from diagnose_sw_token import checkpoint_identity
    import hashlib
    diag=tmp_path/'diag';diag.mkdir();(diag/'inputs.jsonl').write_text('{"id":"one"}\n')
    (diag/'summary.json').write_text(json.dumps(dict(schema='sw_textvqa_probe_v1',alignment_passed=True,
        checkpoint=checkpoint_identity(parent),selection_sha256=hashlib.sha256((diag/'inputs.jsonl').read_bytes()).hexdigest())))
    (diag/'settings.json').write_text(json.dumps(dict(arguments=dict(target_model=str(target),samples=1,
        max_new_tokens=9,image_max_tokens=1024,seed=42))))
    (tests/'train_sw_token.sh').write_text('#!/bin/bash\nset -eu\nmkdir -p "$RUN_GROUP"\nprintf "%s\\n" "$WINDOW_SIZES" "$INIT_CHECKPOINT" "$TARGET_MODEL" "$SAVE_TRAINING_STATE" > "$RUN_GROUP/routed.txt"\n')
    (tests/'benchmark_vlm_static_batch.py').write_text('import json,sys,os\nfrom pathlib import Path\n(Path(os.environ["RUN_GROUP"])/"eval.json").write_text(json.dumps(sys.argv[1:]))\n')
    group=tmp_path/'group'
    env=dict(os.environ,DIAGNOSTICS_DIR=str(diag),PYTHON_BIN=sys.executable,RUN_GROUP=str(group),TARGET_MODEL='wrong_target')
    result=subprocess.run(['bash',str(tests/'train_sw_token_pair.sh')],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert (group/'routed.txt').read_text().splitlines()==['1 2',str(parent),str(target),'0']
    arguments=json.loads((group/'eval.json').read_text())
    assert arguments[arguments.index('--data')+1]==str(diag/'inputs.jsonl')
    assert arguments[arguments.index('--state-mode')+1]=='per_token'
    assert arguments.count('--checkpoint')==3


@torch.no_grad()
def test_bfloat16_masked_controls_restore_first_proposal():
    target,draft,_,example=fixture_model()
    draft.eval();probe=fixed_probe(target,draft,example)
    target.to(dtype=torch.bfloat16);draft.to(dtype=torch.bfloat16)
    probe['context']=probe['context'].to(dtype=torch.bfloat16)
    base=run_condition(draft,target.get_input_embeddings(),probe,'k1')
    for condition in ('k2_pivot_only','k2_pivot_only_origin0'):
        current=run_condition(draft,target.get_input_embeddings(),probe,condition)
        result=compare_conditions(base,current,atol=0.05,rtol=0.01)
        assert result['hidden_close'] and result['logits_close'] and result['first_proposal_equal']
        assert all(a['old_mass']==0. for a in current['attention'])



@torch.no_grad()
def test_draft_only_float32_preserves_target_and_bf16_context():
    target,draft,_,example=fixture_model()
    draft.eval();probe=fixed_probe(target,draft,example)
    target.to(dtype=torch.bfloat16);draft.to(dtype=torch.bfloat16)
    rounded=draft.layers[0].self_attn.q_proj.weight.detach().clone()
    draft.to(dtype=torch.float32)
    torch.testing.assert_close(draft.layers[0].self_attn.q_proj.weight,rounded.float(),rtol=0,atol=0)
    probe['context']=probe['context'].to(dtype=torch.bfloat16)
    context=probe['context'].clone()
    embed=target.get_input_embeddings();weight=embed.weight.detach().clone()
    from sw_token_probe import DraftPrecisionEmbedding
    wrapper=DraftPrecisionEmbedding(embed,draft.dtype)
    assert wrapper.weight is embed.weight
    assert wrapper(probe['pending']).dtype==torch.float32
    base=run_condition(draft,embed,probe,'k1')
    for name in ('k2_pivot_only','k2_pivot_only_origin0'):
        result=run_condition(draft,embed,probe,name)
        assert result['hidden'].dtype==torch.float32
        compared=compare_conditions(base,result,atol=1e-5,rtol=1e-4)
        assert compared['hidden_close'] and compared['logits_close'] and compared['first_proposal_equal']
    assert target.dtype==torch.bfloat16
    torch.testing.assert_close(embed.weight,weight,rtol=0,atol=0)
    torch.testing.assert_close(probe['context'],context,rtol=0,atol=0)


def test_reused_prefixes_reject_changed_ar_tokens(tmp_path,monkeypatch):
    import diagnose_sw_token as cli
    target,draft,_,_=fixture_model();target.eval();target.config.text_config.eos_token_id=-1
    parent=tmp_path/'parent';draft.save_pretrained(parent)
    processor=Processor()
    raw_apply=processor.apply_chat_template
    def deterministic_apply(*args,**kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(123)
            return raw_apply(*args,**kwargs)
    processor.apply_chat_template=deterministic_apply
    processor.image_processor=type('Image',(),{'size':dict(shortest_edge=16,longest_edge=1024)})()
    monkeypatch.setattr(cli,'load_target',lambda *a,**kw:(target,processor))
    monkeypatch.setattr(cli,'select_data',lambda args:{'textvqa':dict(source='fixture',serialized=json.dumps(dict(id='one',question='what?'))+'\n')})
    common=['--workspace',str(tmp_path),'--checkpoint',str(parent),'--target-model','tiny',
        '--device','cpu','--samples','1','--max-new-tokens','9','--stride','1','--probes-per-sample','3']
    prior=tmp_path/'prior'
    cli.main(common+['--output-dir',str(prior)])
    monkeypatch.setattr(cli,'select_data',lambda *a:(_ for _ in ()).throw(AssertionError('must reuse selection')))
    out=tmp_path/'promoted'
    cli.main(common+['--output-dir',str(out),'--draft-dtype','float32','--reuse-prefixes-from',str(prior)])
    summary=json.loads((out/'summary.json').read_text())
    assert summary['prefixes_verified_against_previous']
    assert summary['draft_compute_dtype']=='torch.float32'
    bad=json.loads((prior/'ar_sequences.jsonl').read_text())
    bad['tokens'][0]=(bad['tokens'][0]+1)%128
    (prior/'ar_sequences.jsonl').write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='Target AR prefix changed'):
        cli.main(common+['--output-dir',str(tmp_path/'bad'),'--draft-dtype','float32','--reuse-prefixes-from',str(prior)])
    assert (tmp_path/'bad/prefix_mismatch.json').is_file()
