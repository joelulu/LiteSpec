"""Profile clocks, fallback detection and actual tiny Qwen3.5 decode parity."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from litespec.vlm.inference_profile import DecodeTimeline, gdn_kernel_probe
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.benchmark import RequestPositionState
from report_inference_profile import summarize_profile, report
from test_vlm_static_batch import make_draft, example_for
from test_vlm import tiny_qwen35


def test_cuda_events_are_resolved_only_at_end():
    events = []
    class Event:
        def __init__(self, **kwargs):
            self.index = len(events)
            events.append(self)
        def record(self, stream):
            assert stream == 'stream'
        def elapsed_time(self, other):
            assert len(events) == 3
            return (other.index-self.index)*2.0
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event, current_stream=lambda device:'stream'))
    timeline = DecodeTimeline('cuda:0', torch_module=fake)
    timeline.mark('draft')
    timeline.mark('target_verify')
    timeline.mark(None)
    values = timeline.resolve(.005)
    assert values['stage_seconds'] == dict(draft=.002, target_verify=.002)
    assert values['unassigned_seconds'] == pytest.approx(.001)


def test_actual_python_fallback_probe_restores_existing_callback():
    filename = '/package/transformers/models/qwen3_5/modeling_qwen3_5.py'
    namespace = {}
    exec(compile('def torch_chunk_gated_delta_rule(query):\n return 1\n', filename, 'exec'), namespace)
    previous = sys.getprofile()
    with gdn_kernel_probe() as calls:
        namespace['torch_chunk_gated_delta_rule'](torch.zeros(1,8,1,2))
    assert dict(calls) == {'torch_chunk_gated_delta_rule:seq=8':1}
    assert sys.getprofile() is previous


@pytest.mark.parametrize('family', ['ar', 'dflash', 'dlite_v2'])
@pytest.mark.parametrize('image', [False, True])
def test_real_qwen35_profile_preserves_tokens_and_recovery(family, image, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    draft = make_draft(target, family)
    example = example_for(target, image)
    def run(profile):
        with RequestPositionState(target).activate():
            return generate_static_batch(target, draft, example, batch_size=2,
                                         max_new_tokens=6, profile=profile)
    plain, profiled = run(False), run(True)
    for field in ('token_ids','accept_lengths','target_decode_calls','recovery_replays'):
        assert profiled[field] == plain[field]
    assert 'profile' not in plain
    stages = profiled['profile']['stage_seconds']
    assert all(value >= 0 for value in stages.values())
    if family == 'ar':
        assert 'ar_target_decode' in stages and 'target_replay' not in stages
    else:
        assert 'target_replay' in stages and 'state_snapshot' in stages
        assert profiled['profile']['stage_calls']['target_replay'] == profiled['recovery_replays']
        assert profiled['profile']['stage_calls']['target_verify'] == len(profiled['accept_lengths'])
    summary = dict(model=family, batch_size=2, **summarize_profile([profiled],[plain]))
    assert summary['tokens_match_unprofiled']
    (tmp_path/'decode_profile.json').write_text(json.dumps([summary]))
    text = report(tmp_path).read_text()
    assert family in text and 'Profile/normal time' in text


def test_cli_keeps_normal_rows_separate_from_profile(monkeypatch, tmp_path):
    import benchmark_vlm_static_batch as cli
    from test_vlm_aligned import Processor
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path/'draft'
    make_draft(target, 'dflash').save_pretrained(checkpoint)
    manifest = tmp_path/'prompts.jsonl'
    manifest.write_text(json.dumps(dict(id='sample', question='hello'))+'\n')
    monkeypatch.setattr(cli, 'load_target', lambda *a,**kw:(target,Processor()))
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--target-model','tiny', '--checkpoint','dflash',str(checkpoint),
        '--data',str(manifest),'--output-dir',str(tmp_path/'results'),'--batch-sizes','1',
        '--max-samples','1','--max-new-tokens','4','--device','cpu','--profile-decode',
        '--profile-samples','1','--profile-kernels'])
    cli.main()
    rows = json.loads((tmp_path/'results/summary.json').read_text())
    diagnostics = json.loads((tmp_path/'results/decode_profile.json').read_text())
    assert len(rows) == len(diagnostics) == 2
    assert all('profile_decode_seconds' not in row for row in rows)
    assert all(row['tokens_match_unprofiled'] for row in diagnostics)
    assert all(Path(row['kernel_probe']['trace_file']).is_file() for row in diagnostics)
    assert any(row['kernel_probe']['python_fallback_calls'] for row in diagnostics)
