"""Real tiny Qwen3.5 equivalence, selected retention and cleanup diagnostics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_vlm import tiny_qwen35,tiny_image_inputs
from test_vlm_static_batch import make_draft,example_for
from litespec.vlm.selected_prefill import target_prefill
from litespec.vlm.runtime import selected_layers
from litespec.vlm.baselines import DraftCache
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.benchmark import RequestPositionState
from sample_memory import measured_request


@pytest.mark.parametrize('image',[False,True])
@pytest.mark.parametrize('ids',[[0,1],[1,3]])
def test_selected_hidden_and_logits_equal_full_capture_including_final_norm(image,ids):
    torch.set_num_threads(1);torch.manual_seed(42)
    target=tiny_qwen35()
    draft=SimpleNamespace(target_layer_ids=ids,config=SimpleNamespace(num_target_layers=4))
    inputs=tiny_image_inputs() if image else dict(input_ids=torch.tensor([[1,2,3]]))
    with torch.no_grad():
        dense=target_prefill(target,draft,inputs,None,needs_hidden=True,mode='all')
        slim=target_prefill(target,draft,inputs,None,needs_hidden=True,mode='selected')
    torch.testing.assert_close(dense.logits,slim.logits,rtol=0,atol=0)
    torch.testing.assert_close(selected_layers(dense.hidden_states,ids,4),selected_layers(slim.hidden_states,ids,4),rtol=0,atol=0)
    assert len(slim.hidden_states)==4
    assert [i for i,h in enumerate(slim.hidden_states) if h is not None]==ids
    assert sum(h.numel() for h in slim.hidden_states if h is not None)<sum(h.numel() for h in dense.hidden_states)


@pytest.mark.parametrize('image',[False,True])
@pytest.mark.parametrize('context',['full','pivot1'])
@pytest.mark.parametrize('state_mode',['per_token','full_target_replay'])
def test_complete_static_generation_unchanged(image,context,state_mode):
    torch.set_num_threads(1);torch.manual_seed(7)
    target=tiny_qwen35().requires_grad_(False)
    draft=make_draft(target,'dflash')
    if context=='pivot1':draft.context_window=1
    example=example_for(target,image)
    with RequestPositionState(target).activate():
        dense=generate_static_batch(target,draft,example,max_new_tokens=8,state_mode=state_mode,prefill_hidden_mode='all')
    with RequestPositionState(target).activate():
        slim=generate_static_batch(target,draft,example,max_new_tokens=8,state_mode=state_mode,prefill_hidden_mode='selected')
    assert slim['token_ids']==dense['token_ids']
    assert slim['accept_lengths']==dense['accept_lengths']


def test_slice_features_before_stack_retains_only_window():
    draft=SimpleNamespace(target_layer_ids=[0,2],config=SimpleNamespace(num_target_layers=4))
    output=SimpleNamespace(hidden_states=tuple(torch.randn(1,200,8) for _ in range(5)))
    expected=DraftCache.features(draft,output)[:,-1:]
    from litespec.vlm import runtime
    with patch('litespec.vlm.baselines.selected_layers',wraps=runtime.selected_layers) as collect:
        actual=DraftCache.features(draft,output,token_slice=slice(-1,None))
    assert all(h.size(1)==1 for h in collect.call_args.args[0])
    torch.testing.assert_close(actual,expected)


def test_failure_diagnostic_contains_exact_sample_and_does_not_swallow_oom():
    from litespec.vlm.runtime import PreparedExample
    example=PreparedExample(dict(input_ids=torch.tensor([[1,2,3]])),3,torch.tensor([0,1,0],dtype=torch.bool),'doc:9')
    def failing(*args):raise torch.OutOfMemoryError('test OOM')
    with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
        path=Path(temp)/'progress.jsonl'
        with patch('torch.cuda.is_available',return_value=False),pytest.raises(torch.OutOfMemoryError):
            measured_request(failing,example,256,path=path,label='full',index=7,count=16,batch_size=1,pages=90)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['event'] for r in rows]==['start','failed']
        assert rows[-1]['sample_id']=='doc:9' and rows[-1]['index']==7
        assert rows[-1]['pages']==90 and rows[-1]['input_tokens']==3
