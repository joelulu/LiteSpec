import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from litespec.vlm import train_aligned as aligned
from litespec.vlm import baselines
from litespec.vlm.runtime import PreparedExample
from test_vlm import tiny_qwen35, tiny_image_inputs
from test_vlm_aligned import Processor, ROOT


def examples():
    result = []
    for i in range(4):
        inputs = tiny_image_inputs()
        prefix = inputs['input_ids'].shape[1]
        answer = torch.tensor([[4+i]* (12+i)])
        inputs['input_ids'] = torch.cat([inputs['input_ids'], answer], 1)
        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(answer)], 1)
        result.append(PreparedExample(inputs, prefix, torch.zeros(prefix, dtype=torch.bool), str(i)))
    return result


@pytest.mark.parametrize('algorithm', ['dlite', 'dflash', 'dspark'])
def test_four_different_images_use_one_target_and_draft_batch(monkeypatch, algorithm):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    items = examples()
    batch = aligned.collate_examples(items)
    mask = aligned.loss_mask(batch)
    assert batch.inputs['input_ids'].shape == (4, 23)
    assert batch.inputs['pixel_values'].shape[0] == 16
    assert batch.inputs['image_grid_thw'].shape == (4, 3)
    assert mask.sum(-1).tolist() == [12, 13, 14, 15]
    assert (mask[:, :8] == 0).all()
    torch.testing.assert_close(batch.inputs['pixel_values'], torch.cat([item.inputs['pixel_values'] for item in items]))
    name = 'sft_student_aligned.yaml' if algorithm == 'dlite' else f'sft_{algorithm}.yaml'
    cfg = aligned.load_config(ROOT/'scripts/config/vlm'/name)
    cfg.update(num_draft_layers=1, chs_num_layers=2, sequential_rank=8, block_size=4, num_anchors=3)
    draft = aligned.make_draft(target, Processor(), cfg) if algorithm == 'dlite' else baselines.make_draft(target, Processor(), cfg)
    online = aligned.make_online(target, draft, cfg)
    seen = []
    handle = target.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs['input_ids'].shape[0]), with_kwargs=True)
    prepared = aligned.prepare_forward(target, online, batch)
    handle.remove()
    assert seen == [4], 'the target must process all four images in one forward'
    seen_draft = []
    handle = draft.register_forward_pre_hook(lambda module, args, kwargs: seen_draft.append(kwargs['noise_embedding'].shape[0]), with_kwargs=True)
    output = online(**prepared)
    handle.remove()
    # DLite flattens batch * sampled anchors into independent draft blocks.
    expected = 4 * prepared['prepared_batch'].anchor_positions.size(1) if algorithm == 'dlite' else 4
    assert seen_draft == [expected]
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in draft.parameters())
    if algorithm == 'dlite':
        layout = prepared['prepared_batch']
        lengths = batch.inputs['attention_mask'].sum(-1)
        assert (layout.anchor_positions + 4 <= lengths[:, None]).all()
        with torch.no_grad():
            full = target(**batch.inputs, use_cache=False).logits
            for row in range(4):
                single = target(**items[row].inputs, use_cache=False).logits
                torch.testing.assert_close(full[row, :single.size(1)], single[0], rtol=1e-4, atol=1e-5)
                positions = layout.anchor_positions[row, :, None] + torch.arange(3)
                torch.testing.assert_close(prepared['target_prefill_logits'][row], full[row, positions], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('algorithm', ['dlite', 'dflash'])
def test_batch_four_runner_epoch_length_accumulation_and_export(monkeypatch, tmp_path, algorithm):
    torch.set_num_threads(1)
    monkeypatch.setattr(aligned, 'load_target', lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()))
    manifest = tmp_path/'train.jsonl'
    manifest.write_text('\n'.join(json.dumps(dict(id=str(i), question='describe', answer='square')) for i in range(10)))
    cfg = dict(upstream_config=f'scripts/config/{"dlite_sft" if algorithm=="dlite" else "dflash"}.yaml',
        algorithm=algorithm, target_model='tiny', train_data=str(manifest), output_dir=str(tmp_path/'out'),
        device='cpu', model_role='student', report_to='none', log_interval=1, save_interval=4000,
        num_draft_layers=1, chs_num_layers=2, sequential_rank=8, block_size=4, num_anchors=3,
        batch_size=4, accumulation_steps=3, num_epochs=2)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    aligned.run(['--config', str(path)])
    effective = json.loads((tmp_path/'out/effective_config.json').read_text())
    saved = json.loads((tmp_path/'out/last/checkpoint.json').read_text())
    assert effective['batch_size'] == 4
    assert effective['total_optimizer_steps'] == 2
    assert saved['cursor']['global_step'] == 4  # floor(10/4) batches * 2 epochs
    assert saved['cursor']['optimizer_step'] == 2
    assert (tmp_path/'out/last/model.safetensors').is_file()
    def forbidden(*args, **kwargs):
        raise AssertionError('a compatible completed preflight must be reused')
    monkeypatch.setattr(aligned, 'validate_records', forbidden)
    cfg['batch_size'] = 2
    path.write_text(yaml.safe_dump(cfg))
    aligned.run(['--config', str(path), '--output-dir', str(tmp_path/'reused'),
                 '--reuse-validation-from', str(tmp_path/'out')])
    reused = json.loads((tmp_path/'reused/effective_config.json').read_text())
    assert reused['eligible_indices_sha256'] == effective['eligible_indices_sha256']
    assert reused['batch_size'] == 2 and reused['total_optimizer_steps'] == 4


def validation_fixture(tmp_path):
    records = [dict(id=str(i)) for i in range(5)]
    cfg = dict(data_sha256='source-hash', train_data='/data/train.jsonl', target_model='/models/qwen35',
               max_length=4096, block_size=16, debug_max_samples=0, batch_size=4)
    valid = [0, 2, 4]
    previous = dict(cfg, eligible_samples=3,
        eligible_indices_sha256=hashlib.sha256(json.dumps(valid).encode()).hexdigest())
    (tmp_path/'effective_config.json').write_text(json.dumps(previous))
    (tmp_path/'skipped_training.jsonl').write_text(''.join(json.dumps(dict(index=i, id=str(i), reason='short'))+'\n' for i in [1, 3]))
    return records, cfg


def test_reuse_validation_skips_encoding_and_allows_batch_or_algorithm_change(monkeypatch, tmp_path):
    records, cfg = validation_fixture(tmp_path)
    cfg.update(batch_size=2, algorithm='dflash')
    def forbidden(*args, **kwargs):
        raise AssertionError('reuse must not read or tokenize images')
    monkeypatch.setattr(aligned, 'prepare_example', forbidden)
    result = aligned.preflight(records, None, None, cfg, tmp_path, reuse_from=tmp_path)
    assert result['valid'] == [0, 2, 4]
    assert [row['index'] for row in result['skipped']] == [1, 3]


@pytest.mark.parametrize('field,value', [('data_sha256', 'different'), ('block_size', 8),
    ('max_length', 8192), ('target_model', '/another/model'), ('train_data', '/another/data'), ('debug_max_samples', 3)])
def test_validation_reuse_rejects_incompatible_inputs(tmp_path, field, value):
    records, cfg = validation_fixture(tmp_path)
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        aligned.reuse_validation(records, cfg, tmp_path)


def test_validation_reuse_rejects_incomplete_skip_file(tmp_path):
    records, cfg = validation_fixture(tmp_path)
    (tmp_path/'skipped_training.jsonl').write_text('')
    with pytest.raises(ValueError, match='incomplete'):
        aligned.reuse_validation(records, cfg, tmp_path)
