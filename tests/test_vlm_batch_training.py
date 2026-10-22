import json
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
