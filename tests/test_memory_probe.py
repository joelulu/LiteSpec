from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.pivot_memory import PivotMemoryReader, PromptMemory
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.static_batch import generate_static_batch
from memory_probe import ReaderCapture, compare_logits, compare_conditions, gradient_probe, probe_reference
from test_vlm import draft_config, tiny_qwen35
from test_vlm_static_batch import example_for


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    target = tiny_qwen35().requires_grad_(False)
    config = draft_config()
    config.dlite_config.update(memory_mode='vision', memory_layer_ids=[0, 3], memory_max_tokens=4, memory_rank=8)
    draft = DLiteDraftModel(config).eval()
    return target, draft, example_for(target, True)


def test_logit_comparison_detects_probability_and_token_changes():
    a = torch.tensor([[[1., 0., -1.], [0., 2., 1.]]])
    same = compare_logits(a, a)
    assert same['proposal_block_exact'] and same['probability_l1'] == 0
    b = a.clone()
    b[0, 0, 2] += 5
    changed = compare_logits(a, b)
    assert changed['proposal_token_agreement'] == .5
    assert changed['probability_l1'] > 0 and changed['kl_correct_to_condition'] > 0


def test_capture_detects_bfloat16_residual_rounded_away_and_removes_hooks():
    reader = PivotMemoryReader(8, 4)
    query = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    with ReaderCapture(reader) as capture:
        capture.gate = torch.tensor([.1])
        capture.ungated = torch.ones(1, 1, 8)*1e-7
        capture.residual = torch.ones(1, 1, 8, dtype=torch.bfloat16)*1e-8
        stats = capture.summarize(query)
    assert stats['residual_query_norm_ratio'] > 0
    assert stats['applied_query_norm_ratio'] == stats['query_changed_fraction'] == 0
    assert not reader._forward_hooks and not reader.gate._forward_hooks
    with pytest.raises(RuntimeError):
        with ReaderCapture(reader):
            raise RuntimeError('diagnostic interrupted')
    assert not reader.output._forward_hooks


def test_common_pivot_probe_detects_disabled_reader_and_zero_branch():
    target, draft, ex = fixture()
    pivot = torch.randn(1, 1, 2, 32)
    banks = {name:draft.memory_reader.project(PromptMemory(torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool)))
             for name in ('correct', 'wrong_image')}
    rows = compare_conditions(draft, target, ex, pivot, torch.tensor([[4]]), ex.prefill_length, banks)
    indexed = {r['condition']:r for r in rows}
    assert indexed['correct']['gate_mean'] == indexed['wrong_image']['gate_mean']
    assert indexed['correct']['residual_query_norm_ratio'] > 0
    assert indexed['wrong_image']['logits_mean_abs_delta'] > 0
    assert indexed['disabled']['gate_mean'] is None
    with torch.no_grad():
        draft.memory_reader.output.weight.zero_()
    rows = compare_conditions(draft, target, ex, pivot, torch.tensor([[4]]), ex.prefill_length, banks)
    assert all(r['logits_max_abs_delta'] == 0 for r in rows)


def test_gradient_probe_does_not_update_weights_or_target():
    target, draft, ex = fixture()
    raw = PromptMemory(torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool))
    before = {n:p.detach().clone() for n,p in draft.named_parameters()}
    report = gradient_probe(draft, target, ex, torch.randn(1, 1, 2, 32), [4, 5, 6, 7], raw)
    assert not report['historical_training_gradient'] and not report['optimizer_step']
    assert report['parameters']['output.weight']['gradient_norm'] > 0
    assert report['parameters']['key.weight']['gradient_norm'] > 0
    assert all(p.grad is None for p in target.parameters())
    assert all(p.grad is None for p in draft.parameters())
    for n,p in draft.named_parameters():
        torch.testing.assert_close(before[n], p, rtol=0, atol=0)


def test_reference_walk_has_no_condition_specific_target_calls(monkeypatch):
    target, draft, ex = fixture()
    with RequestPositionState(target).activate():
        trace = generate_static_batch(target, None, ex, max_new_tokens=7)['token_ids']
    donor = replace(ex, inputs={**ex.inputs, 'pixel_values':ex.inputs['pixel_values']+1})
    original = target.forward
    calls = []
    def forward(*a, **kw):
        calls.append(kw['input_ids'].shape[1])
        return original(*a, **kw)
    monkeypatch.setattr(target, 'forward', forward)
    rows, grad = probe_reference(target, draft, ex, donor, trace, stride=2, gradients=True)
    assert len(calls) == len(trace)+1  # donor, primary prefill, AR prefix advancement
    assert len(rows) == ((len(trace)+1)//2)*3
    assert all(r['logits_mean_abs_delta'] == 0 for r in rows if r['condition'] == 'correct')
    assert grad['parameters']['value.weight']['gradient_present']
    assert not draft.memory_reader._forward_hooks


def test_full_diagnostic_cli_three_conditions_and_fixed_prefix(monkeypatch, tmp_path):
    import sys
    import diagnose_memory as diag
    import benchmark_vlm_static_batch as benchmark
    import litespec.vlm.runtime as runtime
    target, draft, ex = fixture()
    checkpoint = tmp_path/'checkpoint'
    draft.save_pretrained(checkpoint)
    evaluation = tmp_path/'group/evaluation'
    prior = evaluation/'charxiv_val_correct'
    prior.mkdir(parents=True)
    records = [dict(id=str(i), question='same question', image=str(tmp_path/f'{i}.png')) for i in range(2)]
    (evaluation/'charxiv_val.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    size = dict(shortest_edge=16, longest_edge=1024)
    (prior/'settings.json').write_text(json.dumps(dict(sample_ids=['0', '1'], target_model='tiny',
        effective_image_processor_size=size, actual_attention_backend='sdpa', seed=42, max_new_tokens=5)))
    (prior/'summary.json').write_text(json.dumps([dict(model='vision', checkpoint=str(checkpoint))]))
    processor = SimpleNamespace(image_processor=SimpleNamespace(size=dict(size)))
    def load(*a, **kw):
        return target, processor
    def prepare(processor, config, record, **kw):
        return replace(ex, sample_id=record['id'])
    monkeypatch.setattr(runtime, 'load_target', load)
    monkeypatch.setattr(runtime, 'prepare_example', prepare)
    monkeypatch.setattr(benchmark, 'load_target', load)
    monkeypatch.setattr(benchmark, 'prepare_example', prepare)
    def run(cmd, directory):
        monkeypatch.setattr(sys, 'argv', ['benchmark', *cmd[2:]])
        benchmark.main()
    monkeypatch.setattr(diag, 'run_logged', run)
    out = tmp_path/'diagnosis'
    monkeypatch.setattr(sys, 'argv', ['diagnose', '--run-group', str(tmp_path/'group'),
        '--output-dir', str(out), '--datasets', 'charxiv_val', '--samples', '2',
        '--max-new-tokens', '5', '--probe-stride', '2', '--device', 'cpu'])
    diag.main()
    summaries = json.loads((out/'paired_summary.json').read_text())
    assert {r['condition'] for r in summaries} == {'correct', 'wrong_image', 'disabled'}
    assert all(r['probes'] == 6 for r in summaries)
    assert len(json.loads((out/'closed_loop_summary.json').read_text())) == 4
    assert (out/'diagnostics.md').is_file()
