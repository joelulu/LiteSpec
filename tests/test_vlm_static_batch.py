import json
from pathlib import Path

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm import baselines, train_aligned
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.context import visual_positions
from litespec.vlm.runtime import PreparedExample, generate
from litespec.vlm.static_batch import PreallocatedAttentionLayer, generate_static_batch, replicate_inputs
from test_vlm import draft_config, tiny_image_inputs, tiny_qwen35
from test_vlm_aligned import Processor, ROOT


def make_draft(target, family):
    if family == "ar":
        return None
    if family.startswith("dlite"):
        cfg = draft_config()
        cfg.dlite_config["architecture_version"] = family
        return DLiteDraftModel(cfg).eval()
    cfg = train_aligned.load_config(ROOT/f"scripts/config/vlm/sft_{family}.yaml")
    cfg.update(num_draft_layers=1, block_size=4)
    return baselines.make_draft(target, Processor(), cfg).eval()


def example_for(target, image):
    inputs = tiny_image_inputs() if image else dict(input_ids=torch.tensor([[1, 2, 3]]))
    return PreparedExample(inputs, inputs["input_ids"].shape[1],
                           visual_positions(inputs["input_ids"][0], target.config), "sample")


def test_preallocated_kv_keeps_storage_and_overwrites_rejected_suffix():
    layer = PreallocatedAttentionLayer(12)
    keys = torch.arange(12, dtype=torch.float).reshape(2, 1, 3, 2)
    layer.update(keys, keys+1)
    pointers = layer.keys.data_ptr(), layer.values.data_ptr()
    layer.crop(-2)
    new = torch.ones(2, 1, 2, 2)*99
    actual, _ = layer.update(new, new)
    assert pointers == (layer.keys.data_ptr(), layer.values.data_ptr())
    assert actual.shape[-2] == layer.get_seq_length() == 3
    torch.testing.assert_close(actual, torch.cat([keys[..., :1, :], new], -2))
    assert layer.get_mask_sizes(2) == (5, 0)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("family", ["ar", "dlite_v1", "dlite_v2", "dflash", "dspark"])
def test_real_hybrid_static_batch_matches_ar_and_uses_one_batched_forward(monkeypatch, batch_size, image, family):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    draft = make_draft(target, family)
    example = example_for(target, image)
    with RequestPositionState(target).activate():
        expected = generate(target, None, example, max_new_tokens=6)["token_ids"]
    calls, pointers = [], []
    forward = target.forward
    def observed(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        calls.append(tuple(ids.shape))
        output = forward(*args, **kwargs)
        layer = output.past_key_values.layers[-1]
        pointers.append((layer.keys.data_ptr(), layer.values.data_ptr()))
        return output
    monkeypatch.setattr(target, "forward", observed)
    with RequestPositionState(target).activate():
        result = generate_static_batch(target, draft, example, batch_size=batch_size, max_new_tokens=6)
    assert result["token_ids"] == expected
    assert all(shape[0] == batch_size for shape in calls)
    assert len(set(pointers)) == 1, "KV storage must not be copied/reallocated per round"
    assert result["output_tokens"] == batch_size*len(expected)
    assert result["replicas_identical"] and result["decode_seconds"] > 0
    if draft is not None:
        assert sum(result["accept_lengths"]) == len(expected)
        assert result["recovery_replays"] > 0
    if image:
        replicated = replicate_inputs(example, batch_size)
        assert replicated["image_grid_thw"].shape[0] == batch_size
        assert replicated["pixel_values"].shape[0] == batch_size*example.inputs["pixel_values"].shape[0]


def test_fully_accepted_static_blocks_and_eos(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    draft = make_draft(target, "dlite_v1")
    example = example_for(target, False)
    with RequestPositionState(target).activate():
        expected = generate_static_batch(target, None, example, max_new_tokens=8)["token_ids"]
    cursor = 0
    def perfect_proposal(**kwargs):
        nonlocal cursor
        values = expected[cursor+1:cursor+draft.block_size]
        values += [1]*(draft.block_size-1-len(values))
        cursor += draft.block_size
        return torch.tensor([values]*2), None
    monkeypatch.setattr(draft, "sample_draft_tokens", perfect_proposal)
    with RequestPositionState(target).activate():
        result = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=8)
    assert result["token_ids"] == expected and result["recovery_replays"] == 0
    target.config.text_config.eos_token_id = expected[0]
    cursor = 0
    with RequestPositionState(target).activate():
        stopped = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=8)
    assert stopped["token_ids"] == expected[:1] and stopped["output_tokens"] == 2
    assert sum(stopped["accept_lengths"]) == 1


def test_static_throughput_excludes_prefill_and_counts_replicas():
    from benchmark_vlm_static_batch import summarize_batches
    result = dict(output_tokens=80, prefill_seconds=10, decode_seconds=2, total_seconds=12,
                  accept_lengths=[4, 6], recovery_replays=1, target_decode_calls=3, prefill_tokens=200)
    summary = summarize_batches([result]*2, 8)
    assert summary["requests"] == 16 and summary["unique_prompts"] == 2
    assert summary["output_tokens_per_second"] == 40
    assert summary["end_to_end_tokens_per_second"] == 160/24
    assert summary["mean_committed_length"] == 5


@pytest.mark.parametrize("family", ["dlite_v1", "dspark"])
def test_static_benchmark_cli_exports_real_batch_metrics(monkeypatch, tmp_path, family):
    import sys
    import benchmark_vlm_static_batch as cli
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path/"draft"
    make_draft(target, family).save_pretrained(checkpoint)
    manifest = tmp_path/"train.jsonl"
    manifest.write_text(json.dumps(dict(id="image", question="describe", answer="never feed this answer"))+"\n")
    class PromptOnlyProcessor(Processor):
        def apply_chat_template(self, messages, **kwargs):
            assert all(message["role"] == "user" for message in messages)
            return super().apply_chat_template(messages, **kwargs)
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, PromptOnlyProcessor()))
    monkeypatch.setattr(sys, "argv", ["benchmark", "--target-model", "tiny", "--checkpoint", family, str(checkpoint),
        "--data", str(manifest), "--data-split", "train", "--output-dir", str(tmp_path/"results"),
        "--batch-sizes", "1", "8", "--max-samples", "1", "--max-new-tokens", "3", "--device", "cpu"])
    cli.main()
    rows = json.loads((tmp_path/"results/summary.json").read_text())
    assert len(rows) == 4 and all(row["dataset_split"] == "train" for row in rows)
    assert all(row["requests"] == row["batch_size"] for row in rows)
    assert all(row["output_tokens_per_second"] == row["output_tokens"]/row["decode_seconds"] for row in rows)
    assert all(row["valid_token_match"] for row in rows)
    assert (tmp_path/f"results/{family}_b8.jsonl").is_file()


def test_three_static_batch_tables_use_batch_axis_and_decode_ratios(tmp_path):
    from report_text_vlm_speedup import comparison_rows, tables
    for stage, split in (("text", "smoke"), ("multimodal_train", "train"), ("multimodal_val", "validation")):
        directory = tmp_path/stage
        directory.mkdir()
        rows = []
        for batch_size in (1, 8):
            base = dict(model="ar", batch_size=batch_size, concurrency=batch_size, dataset_split=split,
                dataset_path=f"/data/{stage}.jsonl", output_tokens_per_second=10*batch_size,
                output_tokens=100*batch_size, wall_seconds=12, prefill_seconds=2, decode_seconds=10,
                mean_latency_seconds=12, mean_committed_length=None, matched=batch_size, requests=batch_size,
                valid_token_match=True, scheduler="static_replicated_batch", benchmark_mode="static_replicated_batch",
                timing_scope="decode_only_after_prefill", unique_prompts=1)
            rows += [base, dict(base, model="dspark", output_tokens_per_second=20*batch_size, decode_seconds=5,
                                wall_seconds=7, mean_committed_length=3)]
        (directory/"summary.json").write_text(json.dumps(rows))
    combined = comparison_rows(tmp_path)
    report = tables(combined)
    assert report.count("### ") == 3
    assert "B=8 speedup / 接受长度" in report and "C=8" not in report
    assert "decode-only" in report
    assert "| dspark | 2.000× / 3.000 | 2.000× / 3.000 |" in report
    assert all(row["batch_size"] in (1, 8) for row in combined)
