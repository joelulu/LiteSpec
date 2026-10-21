from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from litespec.vlm.benchmark import RequestPositionState, run_closed_loop, summarize
from litespec.vlm.runtime import PreparedExample, generate, generate_steps
from litespec.vlm.context import visual_positions
from test_vlm import draft_config, tiny_image_inputs, tiny_qwen35
from litespec.modeling.draft.dlite import DLiteDraftModel


@pytest.mark.parametrize("concurrency", [1, 4, 8])
def test_scheduler_request_accounting_and_wall_throughput(concurrency):
    order = []
    def factory(index, example):
        def steps():
            for _ in range(2):
                order.append(index)
                yield
            return dict(token_ids=[index, index], accept_lengths=[2])
        return steps(), SimpleNamespace(activate=nullcontext)
    rows, elapsed, peak = run_closed_loop(list(range(16)), factory, concurrency)
    assert peak == concurrency
    assert [row["result"]["token_ids"][0] for row in rows] == list(range(16))
    assert order[:concurrency] == list(range(concurrency))
    stats = summarize(rows, elapsed)
    assert stats["output_tokens"] == 32
    assert stats["output_tokens_per_second"] == 32 / elapsed
    assert stats["mean_committed_length"] == 2
    assert stats["mean_accepted_draft_tokens"] == 1


def test_mrope_state_restores_even_on_exception():
    target = torch.nn.Module()
    target.rope_deltas = "outer"
    one, two = RequestPositionState(target), RequestPositionState(target)
    with one.activate():
        assert target.rope_deltas is None
        target.rope_deltas = "one"
    assert target.rope_deltas == "outer"
    with two.activate():
        assert target.rope_deltas is None
        target.rope_deltas = "two"
    with pytest.raises(RuntimeError), one.activate():
        assert target.rope_deltas == "one"
        raise RuntimeError()
    assert target.rope_deltas == "outer"


@pytest.mark.parametrize("family", ["ar", "dlite", "dlite_v1", "dflash"])
def test_interleaved_real_image_and_text_requests_match_single_request(family):
    from litespec.vlm import baselines, train_aligned
    from test_vlm_aligned import Processor, ROOT
    torch.set_num_threads(1)
    torch.manual_seed(16)
    target = tiny_qwen35().requires_grad_(False)
    if family.startswith("dlite"):
        config = draft_config()
        if family == "dlite_v1":
            config.dlite_config["architecture_version"] = "dlite_v1"
        draft = DLiteDraftModel(config).eval()
    elif family == "dflash":
        cfg = train_aligned.load_config(ROOT / "scripts/config/vlm/sft_dflash.yaml")
        cfg.update(num_draft_layers=1, block_size=4)
        draft = baselines.make_draft(target, Processor(), cfg).eval()
    else:
        draft = None
    image = tiny_image_inputs()
    text = dict(input_ids=torch.tensor([[1, 2, 3]]), mm_token_type_ids=torch.zeros(1, 3, dtype=torch.long))
    examples = [PreparedExample(image, 8, visual_positions(image["input_ids"][0], target.config), "image"),
                PreparedExample(text, 3, torch.zeros(3, dtype=torch.bool), "text")]
    examples = examples * 2
    def factory(index, example):
        state = RequestPositionState(target)
        assert state.slots, "test must exercise actual Qwen3.5 mutable MRoPE state"
        return generate_steps(target, draft, example, max_new_tokens=6), state
    expected = [generate(target, draft, example, max_new_tokens=6)["token_ids"] for example in examples]
    results, _, peak = run_closed_loop(examples, factory, 4)
    assert peak == 4
    assert [row["result"]["token_ids"] for row in results] == expected


def test_benchmark_cli_writes_comparable_csv_and_request_records(monkeypatch, tmp_path):
    import json
    import sys
    import benchmark_vlm_checkpoints as cli
    from test_vlm_aligned import Processor
    torch.set_num_threads(1)
    torch.manual_seed(4)
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path / "draft"
    config = draft_config()
    config.dlite_config["architecture_version"] = "dlite_v1"
    DLiteDraftModel(config).save_pretrained(checkpoint)
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(''.join(json.dumps(dict(id=str(i), question="describe")) + '\n' for i in range(2)))
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, Processor()))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--target-model", "tiny", "--checkpoint", "dlite", str(checkpoint),
                                     "--data", str(manifest), "--output-dir", str(tmp_path / "results"),
                                     "--device", "cpu", "--max-samples", "2", "--max-new-tokens", "3",
                                     "--concurrency", "1", "2", "--warmup", "0"])
    cli.main()
    summaries = json.loads((tmp_path / "results/summary.json").read_text())
    assert len(summaries) == 4
    assert all(row["matched"] == row["requests"] == 2 for row in summaries)
    assert all(row["output_tokens_per_second"] > 0 and row["speedup_valid"] for row in summaries)
    assert (tmp_path / "results/summary.csv").is_file()
    assert len((tmp_path / "results/dlite_c2.jsonl").read_text().splitlines()) == 2


def test_profile_and_first_mismatch():
    from benchmark_vlm_checkpoints import mismatch_details
    records = [dict(latency_seconds=5, result=dict(token_ids=[1, 2], accept_lengths=[2],
                    prefill_seconds=1, draft_seconds=.5, verify_seconds=2, recovery_seconds=1))]
    row = summarize(records, 5)
    assert row["speculative_rounds"] == 1
    assert row["verify_and_snapshot_seconds"] == 2
    assert row["other_seconds"] == .5
    assert mismatch_details([1, 3], [1, 2]) == dict(position=1, ar=[2], spec=[3])
    assert mismatch_details([1], [1, 2])["position"] == 1
    assert mismatch_details([1], [1]) is None


@pytest.mark.parametrize("verification", ["block", "sequential"])
def test_native_text_original_entry_matches_ar_and_rejects_images(verification):
    from litespec.vlm.native_text import generate_native_text
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    config = draft_config()
    config.dlite_config["architecture_version"] = "dlite_v1"
    draft = DLiteDraftModel(config).eval()
    example = PreparedExample(dict(input_ids=torch.tensor([[1, 2, 3]])), 3,
                              torch.zeros(3, dtype=torch.bool), "text")
    ar = generate_native_text(target, None, example, max_new_tokens=8)
    spec = generate_native_text(target, draft, example, max_new_tokens=8, verification=verification)
    assert spec["token_ids"] == ar["token_ids"]
    assert sum(spec["accept_lengths"]) == len(spec["token_ids"])
    assert not spec["stage_profile_available"]
    assert spec["total_seconds"] >= spec["decode_seconds"] >= 0
    example.inputs["pixel_values"] = torch.zeros(1)
    with pytest.raises(ValueError, match="text-only"):
        generate_native_text(target, draft, example)


def test_text_gate_requires_correct_native_speedup_and_reference():
    from report_dlite_modalities import text_ready
    rows = [dict(stage=stage, model="dlite_original", valid_token_match=True,
                 speedup_valid=True, speedup_vs_ar=1.2)
            for stage in ("text_native", "text_reference")]
    assert text_ready(rows)
    rows[0]["speedup_vs_ar"] = .9
    assert not text_ready(rows)
    rows[0]["speedup_vs_ar"] = 1.2
    rows[1]["valid_token_match"] = False
    assert not text_ready(rows)


def test_native_text_cli_exports_original_results(monkeypatch, tmp_path):
    import json
    import sys
    import benchmark_vlm_checkpoints as cli
    class TextProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return dict(input_ids=torch.tensor([[1, 2, 3]]))
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    config = draft_config()
    config.dlite_config["architecture_version"] = "dlite_v1"
    checkpoint = tmp_path / "draft"
    DLiteDraftModel(config).save_pretrained(checkpoint)
    data = tmp_path / "text.jsonl"
    data.write_text(json.dumps(dict(id="text", question="explain")) + "\n")
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, TextProcessor()))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--target-model", "tiny", "--checkpoint", "dlite_original", str(checkpoint),
                                     "--data", str(data), "--output-dir", str(tmp_path / "results"), "--device", "cpu",
                                     "--max-samples", "1", "--max-new-tokens", "4", "--concurrency", "1",
                                     "--warmup", "0", "--execution", "native_text"])
    cli.main()
    rows = json.loads((tmp_path / "results/summary.json").read_text())
    assert len(rows) == 2
    assert all(row["valid_token_match"] and row["execution"] == "native_text" for row in rows)
    assert not any(row["stage_profile_available"] for row in rows)
