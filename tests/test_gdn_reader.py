"""Reader integration: real image target, causal anchors, four arms and exports."""
import copy
from types import SimpleNamespace

import pytest
import torch

from litespec.vlm import baselines, train_aligned as aligned
from litespec.vlm.gdn_capture import GDNAnchorCapture, reference_anchor_states
from litespec.vlm.gdn_reader import GDNStateReader, cache_reader_states
from litespec.vlm.runtime import configure_image_token_budget, generate
from litespec.vlm.static_batch import generate_static_batch
from litespec.training.optimizer import register_optimizer_parameters
from test_vlm_aligned import CONFIG, Processor, fixture_model
from test_vlm_static_batch import example_for


def config_for(target, **changes):
    text = target.config.text_config
    value = dict(layer_ids=[0, 2], heads=text.linear_num_value_heads,
                 key_dim=text.linear_key_head_dim, value_dim=text.linear_value_head_dim,
                 rank=2, bottleneck=8, chunk_anchors=2, injection_layer=0)
    value.update(changes)
    return value


def draft_for(target, family, enabled):
    cfg = aligned.load_config(CONFIG if family == "dlite" else CONFIG.with_name("sft_dflash.yaml"))
    cfg.update(algorithm=family, num_draft_layers=1, chs_num_layers=2, sequential_rank=8,
               target_layer_ids=[0, 3], num_anchors=4, block_size=4, gdn_enabled=enabled,
               gdn_experiment=True, gdn_layer_ids=[0, 2], gdn_rank=2,
               gdn_bottleneck=8, gdn_chunk_anchors=2)
    torch.manual_seed(32)
    draft = aligned.make_draft(target, Processor(), cfg) if family == "dlite" else baselines.make_draft(target, Processor(), cfg)
    return draft, cfg


def prefix_inputs(inputs, batch, length):
    return {k: (v[batch:batch+1, :length] if k in {"input_ids", "attention_mask", "mm_token_type_ids"} else v)
            for k, v in inputs.items()}


def test_sparse_capture_matches_native_prefix_cache_and_leaves_target_unchanged():
    target, _, _, example = fixture_model()
    ids = example.inputs["input_ids"]
    anchors, keep = torch.tensor([[8, 10, 14, 0]]), torch.tensor([[True, True, True, False]])
    original = target(**example.inputs, use_cache=False, output_hidden_states=True)
    with GDNAnchorCapture(target, [0, 2], anchors, keep) as capture:
        observed = target(**example.inputs, use_cache=False, output_hidden_states=True)
        states = capture.states
    torch.testing.assert_close(original.logits, observed.logits, rtol=0, atol=0)
    for i, a in enumerate(anchors[0, :3]):
        cache = target(**prefix_inputs(example.inputs, 0, int(a)), use_cache=True).past_key_values
        for layer, state in zip([0, 2], states):
            native = cache.layers[layer].recurrent_states[0]
            torch.testing.assert_close(state[:, i], native, atol=3e-5, rtol=3e-4)
            assert not state.requires_grad
    assert all(not s[:, -1].count_nonzero() for s in states)
    changed = {**example.inputs, "input_ids": ids.clone()}
    changed["input_ids"][:, 10:] = 13
    with GDNAnchorCapture(target, [0, 2], anchors, keep) as capture:
        target(**changed, use_cache=False)
        future_changed = capture.states
    for left, right in zip(states, future_changed):
        torch.testing.assert_close(left[:, :2], right[:, :2], atol=1e-6, rtol=1e-5)
    assert not capture.handles
    assert all("GDNAnchorCapture" not in getattr(h, "__qualname__", "") for m in target.modules() for h in m._forward_hooks.values())
    with pytest.raises(RuntimeError, match="abort"):
        with GDNAnchorCapture(target, [0, 2], anchors, keep):
            raise RuntimeError("abort")
    assert not capture.handles
    assert all("GDNAnchorCapture" not in getattr(h, "__qualname__", "") for m in target.modules() for h in m._forward_hooks.values())


def test_right_padded_batch_uses_own_state_and_sorted_anchors():
    target, _, _, first = fixture_model()
    second = copy.deepcopy(first)
    for key in ("input_ids", "mm_token_type_ids"):
        second.inputs[key] = second.inputs[key][:, :-3]
    second.inputs["input_ids"][:, 8:] = 12
    batch = aligned.collate_examples([first, second])
    anchors = torch.tensor([[8, 12, 15], [8, 11, 0]])
    keep = torch.tensor([[True, True, True], [True, True, False]])
    with GDNAnchorCapture(target, [0, 2], anchors, keep) as capture:
        target(**batch.inputs, use_cache=False)
        states = capture.states
    for row, sample in enumerate([first, second]):
        for slot in range(int(keep[row].sum())):
            cache = target(**prefix_inputs(sample.inputs, 0, int(anchors[row, slot])), use_cache=True).past_key_values
            for layer, state in zip([0, 2], states):
                torch.testing.assert_close(state[row, slot], cache.layers[layer].recurrent_states[0][0], atol=3e-5, rtol=3e-4)


def test_reader_nonsquare_layout_matches_ready_formula_and_block_isolation():
    torch.manual_seed(6)
    cfg = dict(layer_ids=[0, 2], heads=3, key_dim=5, value_dim=7, rank=2,
               bottleneck=4, chunk_anchors=1, injection_layer=0)
    reader = GDNStateReader(16, cfg).eval()
    torch.nn.init.normal_(reader.up.weight, std=.1)
    hidden = torch.randn(2, 12, 16)
    states = tuple(torch.randn(2, 3, 3, 5, 7) for _ in range(2))
    observed = reader(hidden, states)
    # ReaDy formula independently evaluated for one block at a time.
    for a in range(3):
        h = hidden[:, a*4:(a+1)*4]
        state = torch.stack([s[:, a].transpose(-1, -2) for s in states], 1)
        rms = (state.square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
        normalized = state/rms
        q = reader.query(torch.nn.functional.rms_norm(h, (16,))).reshape(2,4,2,3,2)
        compressed = torch.einsum("lhvr,blhvk->blhrk", reader.value_down, normalized)
        key = torch.einsum("bnlhr,lhrk->bnlhk", q, reader.query_up)
        read = torch.einsum("blhrk,bnlhk->bnlhr", compressed, key)/5**.5
        scale = rms[...,0,0].log()[:,None,:,:,None].expand(2,4,2,3,1)
        features = torch.cat((read,scale),-1).flatten(2)
        features = torch.nn.functional.rms_norm(features, (features.shape[-1],))
        expected = h + reader.up(torch.nn.functional.silu(reader.down(features)))
        torch.testing.assert_close(observed[:,a*4:(a+1)*4], expected)
    altered = tuple(s.clone() for s in states)
    altered[0][:,1] += 10
    updated = reader(hidden, altered)
    torch.testing.assert_close(updated[:,:4], observed[:,:4])
    torch.testing.assert_close(updated[:,8:], observed[:,8:])
    assert not torch.allclose(updated[:,4:8], observed[:,4:8])


@pytest.mark.parametrize("family", ["dlite", "dflash"])
def test_paired_initialization_training_gradients_and_export(family, tmp_path):
    target, _, _, example = fixture_model()
    control, cfg = draft_for(target, family, False)
    control_rng = torch.get_rng_state()
    enhanced, cfg_plus = draft_for(target, family, True)
    torch.testing.assert_close(control_rng, torch.get_rng_state(), rtol=0, atol=0)
    for name, param in control.state_dict().items():
        torch.testing.assert_close(param, enhanced.state_dict()[name], rtol=0, atol=0)
    online = aligned.make_online(target, enhanced, cfg_plus)
    base_online = aligned.make_online(target, control, cfg)
    torch.manual_seed(11)
    base_args = aligned.prepare_forward(target, base_online, example)
    torch.manual_seed(11)
    plus_args = aligned.prepare_forward(target, online, example)
    if family == "dlite":
        torch.testing.assert_close(base_args["prepared_batch"].anchor_positions, plus_args["prepared_batch"].anchor_positions)
    else:
        torch.testing.assert_close(base_args["anchor_positions"], plus_args["anchor_positions"])
    torch.testing.assert_close(online(**plus_args).loss, base_online(**base_args).loss, atol=1e-7, rtol=1e-6)
    parameters, specs = register_optimizer_parameters(enhanced)
    assert all(s.optimizer == "adamw" for s in specs if "gdn_reader" in s.name)
    optimizer = torch.optim.AdamW(parameters, lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        online(**plus_args).loss.backward()
        optimizer.step()
    reader = enhanced.layers[0].gdn_reader
    assert reader.up.weight.grad.abs().sum() > 0
    assert reader.value_down.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    enhanced.eval()
    expected = online(**plus_args).loss
    enhanced.save_pretrained(tmp_path)
    loaded = baselines.load_draft(tmp_path, attn_implementation="sdpa")
    loaded.eval()
    reloaded_online = aligned.make_online(target, loaded, cfg_plus)
    torch.testing.assert_close(reloaded_online(**plus_args).loss, expected)
    assert loaded.config.gdn_reader_config == enhanced.config.gdn_reader_config


@pytest.mark.parametrize("family", ["dlite", "dflash"])
@pytest.mark.parametrize("mode", ["per_token", "compact_replay"])
def test_generation_reads_only_committed_state_after_rejection(family, mode):
    target, _, _, _ = fixture_model()
    draft, _ = draft_for(target, family, True)
    draft.eval()
    torch.nn.init.normal_(draft.layers[0].gdn_reader.up.weight, std=.1)
    example = example_for(target, True)
    seen = []
    def record(module, args):
        seen.append(tuple(s.clone() for s in args[1]))
    hook = draft.layers[0].gdn_reader.register_forward_pre_hook(record)
    if family == "dlite":
        draft.sample_draft_tokens = lambda **kw: (torch.zeros(2, 3, dtype=torch.long), None)
    else:
        draft._propose_draft_tokens = lambda *a, **kw: (torch.zeros(2, 3, dtype=torch.long), None, None)
    actual = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=7, state_mode=mode)
    hook.remove()
    ar = generate_static_batch(target, None, example, batch_size=2, max_new_tokens=7)
    assert actual["token_ids"] == ar["token_ids"]
    # The first read must be precisely the image-prefill state, before pending.
    cache = target(**example.inputs, use_cache=True).past_key_values
    expected = cache_reader_states(draft, cache)
    assert len(seen) > 1
    for state, native in zip(seen[0], expected):
        torch.testing.assert_close(state, native.expand(2, *native.shape[1:]), atol=1e-6, rtol=1e-5)
    # Each later read agrees with sequential execution of the committed prefix.
    # Native chunk prefill vs sequential may differ slightly; tolerances are explicit.
    committed = 0
    for round_states, kept in zip(seen, actual["accept_lengths"]):
        if committed:
            cache = target(input_ids=torch.tensor([actual["token_ids"][committed-previous:committed]]),
                           past_key_values=cache, use_cache=True).past_key_values
        for state, native in zip(round_states, cache_reader_states(draft, cache)):
            torch.testing.assert_close(state[:1], native, atol=5e-5, rtol=5e-4)
        previous, committed = kept, committed+kept


def test_image_cap_and_bad_configuration():
    target, _, _, _ = fixture_model()
    p = SimpleNamespace(image_processor=SimpleNamespace(size=dict(shortest_edge=100, longest_edge=10000)))
    configure_image_token_budget(p, target.config, 8)
    assert p.image_processor.size == dict(shortest_edge=32, longest_edge=32)
    with pytest.raises(ValueError):
        configure_image_token_budget(p, target.config, 0)


def test_cuda_preflight_image_pipeline_cpu_reference():
    from check_gdn_reader_cuda import check_image_pipeline
    rows = check_image_pipeline(device="cpu", dtype=torch.float32)
    assert len(rows) == 2 and all(row["passed"] for row in rows)


@pytest.mark.parametrize("family", ["dlite", "dflash"])
def test_aligned_runner_exports_trained_reader_with_original_optimizer(monkeypatch, tmp_path, family):
    import json
    import yaml
    from test_vlm import tiny_qwen35
    monkeypatch.setattr(aligned, "load_target", lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(dict(id=str(i), question="describe", answer="a square"))+"\n" for i in range(4)))
    cfg = dict(upstream_config="scripts/config/dlite_sft.yaml" if family == "dlite" else "scripts/config/dflash.yaml",
        algorithm=family,target_model="tiny",train_data=str(manifest),output_dir=str(tmp_path/"output"),device="cpu",
        report_to="none",log_interval=1,num_epochs=1,batch_size=2,accumulation_steps=2,
        num_draft_layers=1,chs_num_layers=2,sequential_rank=8,target_layer_ids=[0,3],block_size=4,num_anchors=3,
        gdn_enabled=True,gdn_experiment=True,gdn_layer_ids=[0,2],gdn_rank=2,gdn_bottleneck=8,gdn_chunk_anchors=2)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    aligned.run(["--config",str(path),"--save-training-state"])
    checkpoint = tmp_path/"output/last"
    loaded = baselines.load_draft(checkpoint,attn_implementation="sdpa")
    assert loaded.layers[0].gdn_reader.up.weight.count_nonzero()
    saved = torch.load(checkpoint/"training_state_rank_00000.pt",weights_only=False)
    assert saved["cursor"]["optimizer_step"] == 1
    assert saved["config"]["gdn_enabled"] is True


@pytest.mark.parametrize("family", ["dlite", "dflash"])
def test_single_request_runtime_supports_reader_checkpoint(family):
    target, _, _, _ = fixture_model()
    draft, _ = draft_for(target, family, True)
    draft.eval()
    torch.nn.init.normal_(draft.layers[0].gdn_reader.up.weight, std=.1)
    example = example_for(target, True)
    if family == "dlite":
        draft.sample_draft_tokens = lambda **kw: (torch.zeros(1,3,dtype=torch.long), None)
    else:
        draft._propose_draft_tokens = lambda *a,**kw:(torch.zeros(1,3,dtype=torch.long),None,None)
    actual = generate(target, draft, example, max_new_tokens=7, verification="sequential")
    reference = generate(target, None, example, max_new_tokens=7, verification="sequential")
    assert actual["token_ids"] == reference["token_ids"]
