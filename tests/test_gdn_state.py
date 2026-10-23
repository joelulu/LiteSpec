"""Every-prefix hybrid-state recovery and multi-round text/image decode."""
import pytest
import torch

from litespec.vlm.gdn_state import GDNStateTransaction, reference_recurrence, recurrence
from litespec.vlm.static_batch import generate_static_batch, make_hybrid_cache
from litespec.vlm.benchmark import RequestPositionState
from litespec.modeling.draft.dlite import _crop_target_cache
from test_vlm import tiny_qwen35
from test_vlm_static_batch import example_for, make_draft


@pytest.mark.parametrize("mode", ["per_token", "compact_replay"])
@pytest.mark.parametrize("kept", [1, 2, 4])
@pytest.mark.parametrize("image", [False, True])
def test_all_prefixes_restore_recurrent_conv_kv_and_next_logits(mode, kept, image):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    target = tiny_qwen35().requires_grad_(False)
    example = example_for(target, image)
    prefix = torch.tensor([[5, 6, 7, 8]])
    def prefill():
        cache = make_hybrid_cache(target, 64)
        target(**example.inputs, past_key_values=cache, use_cache=True)
        return cache
    with torch.no_grad(), RequestPositionState(target).activate():
        oracle = prefill()
        expected = target(input_ids=prefix[:, :kept], past_key_values=oracle, use_cache=True)
        # Reset request-local MRoPE before the second prefill.
    original_forwards = [m.__dict__.get("forward") for m in target.modules()]
    with torch.no_grad(), RequestPositionState(target).activate(), GDNStateTransaction(target, mode) as tx:
        cache = prefill()
        with tx.capture(cache):
            output = target(input_ids=prefix, past_key_values=cache, use_cache=True)
        assert tx.peak_record_bytes > 0
        tx.commit(cache, kept, 4)
        _crop_target_cache(cache, current_length=example.prefill_length+4, tokens_to_remove=4-kept)
        for a, b in zip(cache.layers, oracle.layers):
            if hasattr(a, "recurrent_states"):
                torch.testing.assert_close(a.recurrent_states[0], b.recurrent_states[0], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(a.conv_states[0], b.conv_states[0], atol=2e-5, rtol=2e-5)
            else:
                assert a.get_seq_length() == b.get_seq_length()
                n = a.get_seq_length()
                torch.testing.assert_close(a.keys[..., :n, :], b.keys[..., :n, :], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(output.logits[:, kept-1], expected.logits[:, -1], atol=2e-5, rtol=2e-5)
        assert not tx.records
    assert original_forwards == [m.__dict__.get("forward") for m in target.modules()]


@pytest.mark.parametrize("family", ["dflash", "dlite_v2"])
@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_three_modes_match_on_real_tiny_hybrid_target(family, image, batch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    draft = make_draft(target, family)
    example = example_for(target, image)
    rows = []
    for mode in ("full_target_replay", "per_token", "compact_replay"):
        with RequestPositionState(target).activate():
            rows.append(generate_static_batch(target, draft, example, batch_size=batch,
                max_new_tokens=9, state_mode=mode, profile=True))
    assert all(r["token_ids"] == rows[0]["token_ids"] for r in rows)
    assert all(r["accept_lengths"] == rows[0]["accept_lengths"] for r in rows)
    assert rows[0]["recovery_replays"] > 0
    for row in rows[1:]:
        assert row["recovery_replays"] == 0
        assert row["target_decode_calls"] == len(row["accept_lengths"])
        assert "target_replay" not in row["profile"]["stage_seconds"]
        assert row["peak_transaction_record_bytes"] > 0
    assert rows[1]["peak_transaction_record_bytes"] > rows[2]["peak_transaction_record_bytes"]


def test_adapter_restores_methods_after_exception():
    target = tiny_qwen35()
    before = [(m, m.__dict__.get("forward")) for m in target.modules()]
    with pytest.raises(RuntimeError, match="interrupted"):
        with GDNStateTransaction(target, "compact_replay"):
            raise RuntimeError("interrupted")
    assert all(m.__dict__.get("forward") is old for m, old in before)


@pytest.mark.parametrize("mode", ["per_token", "compact_replay"])
def test_full_acceptance_truncated_tail_and_eos(monkeypatch, mode):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    draft = make_draft(target, "dlite_v2")
    example = example_for(target, False)
    with RequestPositionState(target).activate():
        expected = generate_static_batch(target, None, example, max_new_tokens=7)["token_ids"]
    cursor = 0
    def perfect(**kwargs):
        nonlocal cursor
        values = expected[cursor+1:cursor+draft.block_size]
        values += [1]*(draft.block_size-1-len(values))
        cursor += draft.block_size
        return torch.tensor([values]*2), None
    monkeypatch.setattr(draft, "sample_draft_tokens", perfect)
    with RequestPositionState(target).activate():
        row = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=7, state_mode=mode)
    assert row["token_ids"] == expected and row["recovery_replays"] == 0
    assert row["accept_lengths"] == [4, 3]
    target.config.text_config.eos_token_id = expected[0]
    for cap in (1, 7):
        cursor = 0
        with RequestPositionState(target).activate():
            row = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=cap, state_mode=mode)
        assert row["token_ids"] == expected[:1] and row["output_tokens"] == 2


def test_three_mode_cli_and_combined_report(monkeypatch, tmp_path):
    import json
    import sys
    import benchmark_vlm_static_batch as cli
    from report_qwen35_state_modes import report
    from test_vlm_aligned import Processor
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path/"draft"
    make_draft(target, "dflash").save_pretrained(checkpoint)
    data = tmp_path/"prompts.jsonl"
    data.write_text(json.dumps(dict(id="test", question="hello"))+"\n")
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw:(target, Processor()))
    fixed_example = example_for(target, True)
    # The synthetic Processor generates fresh random pixels on each call.
    # Cross-mode comparison must use the same image tensor as well as the ID.
    monkeypatch.setattr(cli, "prepare_example", lambda *a, **kw:fixed_example)
    for mode in ("full_target_replay", "per_token", "compact_replay"):
        monkeypatch.setattr(sys, "argv", ["benchmark", "--target-model", "tiny", "--checkpoint", "dflash", str(checkpoint),
            "--data", str(data), "--output-dir", str(tmp_path/mode/"text"), "--state-mode", mode,
            "--batch-sizes", "1", "--max-samples", "1", "--max-new-tokens", "5", "--device", "cpu",
            "--profile-decode", "--profile-samples", "1"])
        cli.main()
    rows = report(tmp_path)
    assert len(rows) == 6
    assert all(r["token_exact_vs_full_target_replay"] and r["token_exact_vs_per_token"] for r in rows)
    assert all(r["profiled_tokens_match_normal"] for r in rows)
    for row in rows:
        if row["model"] != "ar" and row["state_mode"] != "full_target_replay":
            assert row["target_calls_per_round"] == 1 and row["full_target_replay_ms"] == 0
    assert (tmp_path/"state_modes.csv").is_file()
    assert "circular output-only" in (tmp_path/"state_modes.md").read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fused_state_kernel_against_torch_and_every_prefix(dtype):
    torch.manual_seed(42)
    shape = (2, 8, 4)
    q, k = [torch.randn(*shape, 128, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(*shape, 128, device="cuda", dtype=dtype)
    g = -torch.rand(*shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(*shape, device="cuda", dtype=dtype)
    initial = torch.randn(2, 4, 128, 128, device="cuda", dtype=torch.float32)
    expected, final, states = reference_recurrence(q, k, v, g, beta, initial, save_states=True)
    actual, got_final, got_states = recurrence(q, k, v, g, beta, initial, save_states=True)
    torch.testing.assert_close(actual, expected, atol=.02 if dtype == torch.bfloat16 else 2e-5, rtol=.02 if dtype == torch.bfloat16 else 2e-5)
    torch.testing.assert_close(got_final, final, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(got_states, states, atol=2e-5, rtol=2e-5)
    for kept in range(1, shape[1]+1):
        _, committed, _ = recurrence(None, k, v, g, beta, initial, steps=kept)
        torch.testing.assert_close(committed, got_states[:, kept-1], atol=2e-5, rtol=2e-5)
