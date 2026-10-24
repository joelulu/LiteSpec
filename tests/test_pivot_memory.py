from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.pivot_memory import PivotMemoryReader, PromptMemory
from litespec.training.optimizer import register_optimizer_parameters
from litespec.vlm.memory import build_prompt_memory, prompt_mask_from_example
from litespec.vlm import train_aligned as aligned
from litespec.vlm.benchmark import RequestPositionState
from litespec.vlm.static_batch import generate_static_batch
from litespec.vlm.context import ContextPolicy
from litespec.vlm.runtime import draft_block
from test_vlm import draft_config
from test_vlm_aligned import fixture_model
from test_vlm_static_batch import example_for


def memory_draft(mode):
    cfg = draft_config()
    cfg.dlite_config.update(memory_mode=mode, memory_max_tokens=3,
        memory_rank=8, memory_layer_ids=[0, 3])
    return DLiteDraftModel(cfg)


@pytest.mark.parametrize("mode", ["vision", "text"])
def test_memory_never_reads_answers_or_padding(mode):
    d = memory_draft(mode)
    ids = torch.tensor([[1, 9, 9, 2, 3, 4], [9, 2, 3, 4, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]).bool()
    states = tuple(torch.randn(2, 6, 32) for _ in range(5))
    config = SimpleNamespace(image_token_id=9)
    expected = build_prompt_memory(states, d, ids, mask, config)
    changed = tuple(t.masked_fill(~mask[..., None], float("nan")) for t in states)
    actual = build_prompt_memory(changed, d, ids, mask, config)
    torch.testing.assert_close(actual.values, expected.values)
    assert actual.valid.sum(1).tolist() == ([2, 1] if mode == "vision" else [2, 1])


def test_projected_bank_equals_raw_and_empty_modality_is_exact_zero():
    torch.manual_seed(5)
    reader = PivotMemoryReader(32, 8)
    bank = PromptMemory(torch.randn(2, 3, 32), torch.tensor([[1, 1, 0], [0, 0, 0]]).bool())
    pivot = torch.randn(2, 4, 2, 32)
    raw = reader(pivot, bank)
    torch.testing.assert_close(raw, reader(pivot, reader.project(bank)))
    assert torch.count_nonzero(raw[1]) == 0
    assert torch.isfinite(raw).all()
    raw.sum().backward()
    assert reader.key.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["vision", "text"])
def test_training_loss_backward_checkpoint_and_shared_initialization(tmp_path, mode):
    target, _, _, example = fixture_model()
    torch.manual_seed(17)
    baseline = DLiteDraftModel(draft_config())
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    draft = memory_draft(mode)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, draft.state_dict()[name], rtol=0, atol=0)
    cfg = dict(block_size=4, num_anchors=4, flex_attention_backend="TRITON",
        final_ce_weight=.1, l1_loss_weight=1., base_lm_ce_weight=0.)
    online = aligned.make_online(target, draft, cfg)
    batch = aligned.collate_examples([example, replace(example, sample_id="second")])
    inputs = aligned.prepare_forward(target, online, batch)
    assert not prompt_mask_from_example(batch)[:, example.prefill_length:].any()
    output = online(**inputs)
    output.loss.backward()
    assert draft.memory_reader.output.weight.grad.abs().sum() > 0
    assert draft.memory_reader.query.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    register_optimizer_parameters(draft)  # Every new parameter has an optimizer owner.
    draft.save_pretrained(tmp_path)
    loaded = DLiteDraftModel.from_pretrained(tmp_path, attn_implementation="sdpa")
    assert loaded.config.dlite_config["memory_mode"] == mode
    for name, tensor in draft.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("condition", ["correct", "wrong_image", "disabled"])
def test_static_inference_projects_once_and_preserves_primary_prefill(monkeypatch, condition):
    target, _, _, _ = fixture_model()
    draft = memory_draft("vision").eval()
    example = example_for(target, True)
    donor = replace(example, inputs={**example.inputs,
        "pixel_values": example.inputs["pixel_values"] + 2}, sample_id="donor")
    projects = []
    project = draft.memory_reader.project
    def counted(bank):
        projects.append(bank.values.clone())
        return project(bank)
    monkeypatch.setattr(draft.memory_reader, "project", counted)
    primary_ids = example.inputs["input_ids"].clone()
    with torch.no_grad(), RequestPositionState(target).activate():
        result = generate_static_batch(target, draft, example, batch_size=2, max_new_tokens=6,
            memory_example=donor if condition == "wrong_image" else None,
            disable_memory=condition == "disabled", state_mode="per_token")
    assert len(projects) == (0 if condition == "disabled" else 1)
    assert result["replicas_identical"]
    assert result["memory_condition"] == condition
    assert result["memory_tokens"] == (0 if condition == "disabled" else 3)
    torch.testing.assert_close(example.inputs["input_ids"], primary_ids)


def test_missing_memory_fails_instead_of_silently_disabling():
    reader = PivotMemoryReader(32, 8)
    with pytest.raises(ValueError, match="explicit prompt memory"):
        reader(torch.randn(1, 1, 2, 32), None)


def test_native_forward_memory_matches_cached_runtime_injection():
    d = memory_draft('vision').eval()
    embedding = torch.nn.Embedding(128, 32)
    pivot = torch.randn(2, 1, 2, 32)
    bank = PromptMemory(torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool))
    anchors = torch.tensor([[4], [5]])
    ids = torch.full((2, 4), 127)
    ids[:, 0] = anchors[:, 0]
    query = d.build_inference_query_embeddings(embedding, ids)
    offset = 0 if d.architecture_version == 'dlite_v1' else 1
    pos = torch.arange(offset, offset+4)[None].expand(2, -1)
    context = torch.zeros(2, 2, dtype=torch.long)
    direct = d(position_ids=pos, noise_embedding=query, target_hidden=pivot,
        history_hidden=query.new_empty(2, 1, 0, 32), prompt_memory=bank,
        rotary_position_ids=torch.cat([context, pos], -1), is_causal=False)
    runtime, _ = draft_block(d, embedding, anchor_ids=anchors, pivot=pivot, history=None,
        anchor=8, prefill_length=8, vision_mask=torch.zeros(8, dtype=torch.bool),
        policy=ContextPolicy.no_history(), prompt_memory=d.memory_reader.project(bank))
    torch.testing.assert_close(d._prediction_hidden(direct), runtime)


def test_real_benchmark_cli_wrong_image_and_ar_reuse(monkeypatch, tmp_path):
    import json
    import sys
    import benchmark_vlm_static_batch as cli
    from test_vlm_aligned import Processor
    target, _, _, _ = fixture_model()
    checkpoint = tmp_path/'draft'
    memory_draft('vision').save_pretrained(checkpoint)
    data = tmp_path/'data.jsonl'
    data.write_text('\n'.join(json.dumps(dict(id=str(i), question='same question', image=f'{i}.png')) for i in range(2)))
    monkeypatch.setattr(cli, 'load_target', lambda *a, **kw: (target, Processor()))
    def prepare(processor, config, record, **kwargs):
        ex = example_for(target, True)
        return replace(ex, sample_id=record['id'])
    monkeypatch.setattr(cli, 'prepare_example', prepare)
    common = ['benchmark', '--target-model', 'tiny', '--data', str(data), '--batch-sizes', '1',
        '--max-samples', '2', '--max-new-tokens', '5', '--device', 'cpu', '--cpu-inputs', '--state-mode', 'per_token']
    monkeypatch.setattr(sys, 'argv', common+['--output-dir', str(tmp_path/'normal'), '--checkpoint', 'vision', str(checkpoint)])
    cli.main()
    monkeypatch.setattr(sys, 'argv', common+['--output-dir', str(tmp_path/'wrong'), '--checkpoint', 'vision_wrong', str(checkpoint),
        '--memory-condition', 'wrong_image', '--ar-reference', str(tmp_path/'normal')])
    cli.main()
    rows = json.loads((tmp_path/'wrong/summary.json').read_text())
    assert rows[1]['mean_memory_tokens'] == 3
    assert rows[1]['donor_prefill_seconds'] > 0
    settings = json.loads((tmp_path/'wrong/settings.json').read_text())
    assert all(k != v['donor_id'] for k, v in settings['wrong_image_mapping'].items())


@pytest.mark.parametrize('mode', ['none', 'vision', 'text'])
def test_training_entrypoint_one_update_exports_memory_checkpoint(monkeypatch, tmp_path, mode):
    import json
    import yaml
    from test_vlm import tiny_qwen35
    from test_vlm_aligned import Processor
    monkeypatch.setattr(aligned, 'load_target', lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()))
    data = tmp_path/'train.jsonl'
    data.write_text('\n'.join(json.dumps(dict(id=str(i), question='describe', answer='target answer')) for i in range(4)))
    cfg = dict(upstream_config='scripts/config/dlite_sft.yaml', target_model='tiny', train_data=str(data),
        output_dir=str(tmp_path/'run'), device='cpu', model_role='student', report_to='none', log_interval=1,
        num_draft_layers=1, chs_num_layers=2, sequential_rank=8, block_size=4, num_anchors=2,
        batch_size=2, accumulation_steps=1, num_epochs=1, memory_mode=mode, memory_max_tokens=3, memory_rank=8)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    aligned.run(['--config', str(path), '--max-steps', '1'])
    loaded = DLiteDraftModel.from_pretrained(tmp_path/'run/last', attn_implementation='sdpa')
    assert (loaded.memory_reader is not None) == (mode != 'none')
    effective = json.loads((tmp_path/'run/effective_config.json').read_text())
    assert effective['eligible_samples'] == 4


def test_evaluation_plan_runs_three_models_and_wrong_image_only_for_vision(monkeypatch, tmp_path):
    import json
    import sys
    import evaluate_memory_ablation as evaluation
    group = tmp_path/'group'
    for mode in ('none', 'vision', 'text'):
        path = group/mode/'last'
        path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps(dict(dlite_config=dict(memory_mode=mode),
            vlm_config=dict(image_max_tokens=1024))))
    target = tmp_path/'target'
    target.mkdir()
    (target/'config.json').write_text(json.dumps(dict(vision_config=dict(patch_size=16, spatial_merge_size=2))))
    monkeypatch.setenv('TARGET_MODEL', str(target))
    monkeypatch.setenv('EVAL_DISABLED', '0')
    monkeypatch.setenv('IMAGE_MAX_TOKENS', '1024')
    monkeypatch.setattr(evaluation, 'select_data', lambda args: {
        case:dict(serialized='{}\n', samples=2) for case in ('gsm8k', 'charxiv_val')})
    commands = []
    def fake_run(cmd, env, directory):
        commands.append(cmd)
        directory.mkdir()
        labels = ['ar']+[cmd[i+1] for i, arg in enumerate(cmd) if arg == '--checkpoint']
        rows = [dict(model=label, dataset_name=cmd[cmd.index('--dataset-name')+1], batch_size=1,
            mean_committed_length=None if label == 'ar' else 2., raw_throughput_speedup_vs_ar=1.,
            output_tokens_per_second=10., mean_memory_tokens=3.) for label in labels]
        (directory/'summary.json').write_text(json.dumps(rows))
    monkeypatch.setattr(evaluation, 'run_logged', fake_run)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--run-group', str(group)])
    evaluation.main()
    assert len(commands) == 3  # GSM8K normal; CharXiv normal + wrong-image.
    assert commands[-1][commands[-1].index('--memory-condition')+1] == 'wrong_image'
    assert '--ar-reference' in commands[-1]
    assert (group/'evaluation/memory_tables.md').is_file()
