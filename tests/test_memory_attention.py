"""Numerical controls for concentration, masking and actual-reader hooks."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from memory_attention import attention_summary
from memory_probe import ReaderCapture


def reader_types():
    # Load the small reader standalone; these tests do not need target weights.
    name = 'attention_test_reader'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1]/'litespec/modeling/draft/pivot_memory.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    m = sys.modules[name]
    return m.PivotMemoryReader, m.PromptMemory


def test_uniform_attention_is_not_low_total_mass():
    q, k, v = torch.zeros(1, 1, 4), torch.randn(1, 128, 4), torch.randn(1, 128, 4)
    s, top = attention_summary(q, k, v, torch.ones(1, 128, dtype=torch.bool), v.mean(1, keepdim=True))
    assert s['attention_max_weight'] == pytest.approx(1/128)
    assert s['attention_peak_vs_uniform'] == pytest.approx(1)
    assert s['attention_entropy_normalized'] == pytest.approx(1)
    assert s['attention_effective_tokens'] == pytest.approx(128)
    assert s['attention_top16_mass'] == pytest.approx(16/128)
    assert s['attention_readout_uniform_delta_ratio'] < 1e-6
    assert len(top[0]['slots']) == 8


def test_selective_attention_and_padding():
    q = torch.tensor([[[20., 0.]]])
    k = torch.tensor([[[1., 0.], [-1., 0.], [1000., 0.]]])
    v = torch.tensor([[[1., 2.], [-2., 1.], [999., 999.]]])
    valid = torch.tensor([[True, True, False]])
    scores = (q @ k.transpose(-1, -2))/2**.5
    w = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
    s, top = attention_summary(q, k, v, valid, w @ v)
    assert s['attention_valid_tokens'] == 2
    assert s['attention_max_weight'] > .999
    assert s['attention_entropy_normalized'] < .001
    assert s['attention_effective_tokens'] == pytest.approx(1)
    assert 2 not in top[0]['slots']


def test_identical_values_make_different_attention_readouts_identical():
    q = torch.ones(1, 1, 2)*10
    k = torch.tensor([[[1., 0.], [-1., 0.], [0., 0.]]])
    v = torch.ones(1, 3, 2)
    s, _ = attention_summary(q, k, v, torch.ones(1, 3, dtype=torch.bool), v[:, :1])
    assert s['attention_peak_vs_uniform'] > 2.9
    assert s['attention_value_relative_spread'] == 0
    assert s['attention_readout_uniform_delta_ratio'] == 0


def test_value_cancellation_can_give_zero_readout_despite_mass_one():
    q, k = torch.zeros(1, 1, 2), torch.zeros(1, 2, 2)
    v = torch.tensor([[[1., 0.], [-1., 0.]]])
    s, _ = attention_summary(q, k, v, torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 1, 2))
    assert s['attention_readout_value_ratio'] == 0
    assert s['attention_entropy_normalized'] == pytest.approx(1)


@pytest.mark.parametrize('projected', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_capture_actual_reader_without_changing_outputs(projected, dtype):
    Reader, Memory = reader_types()
    torch.manual_seed(12)
    r = Reader(16, 8).to(dtype).eval()
    raw = Memory(torch.randn(2, 5, 16).to(dtype), torch.tensor([[1,1,1,0,0], [0,0,0,0,0]]).bool())
    pivot = torch.randn(2, 1, 3, 16).to(dtype)
    bank = r.project(raw) if projected else raw
    with torch.no_grad():
        expected = r(pivot, bank)
        with ReaderCapture(r) as capture:
            actual = r(pivot, bank)
        stats = capture.summarize(torch.randn(2, 8, 16).to(dtype))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats['attention_valid_tokens'] == 3
    assert stats['attention_reconstruction_relative_error'] < .05
    assert not r._forward_hooks and not r._forward_pre_hooks
    assert not r.query._forward_hooks and not r.key._forward_hooks


def test_empty_and_single_token_are_finite():
    q = torch.zeros(1, 1, 2)
    k = v = torch.ones(1, 1, 2)
    s, top = attention_summary(q, k, v, torch.zeros(1, 1, dtype=torch.bool), v)
    assert all(x is None for x in s.values()) and top == []
    s, top = attention_summary(q, k, v, torch.ones(1, 1, dtype=torch.bool), v)
    assert s['attention_entropy_normalized'] == 0
    assert s['attention_effective_tokens'] == 1


def test_report_prints_attention_table(tmp_path):
    from diagnose_memory import report
    s, _ = attention_summary(torch.zeros(1, 1, 2), torch.ones(1, 4, 2),
        torch.ones(1, 4, 2), torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 1, 2))
    row = dict(dataset='charxiv_val', condition='correct', sample_id='a', gate_mean=.1, gate_min=.1, gate_max=.1,
        residual_query_norm_ratio=.2, applied_query_norm_ratio=.2, query_changed_fraction=1.,
        proposal_token_agreement=1., logits_mean_abs_delta=0., logits_max_abs_delta=0., **s)
    report(tmp_path, [row], [])
    table = (tmp_path/'attention.md').read_text()
    assert 'charxiv_val' in table and 'Top16 mass' in table
