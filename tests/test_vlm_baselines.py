import json
from pathlib import Path

import pytest
import torch
import yaml

from litespec.vlm import baselines, train_aligned
from litespec.vlm.runtime import PreparedExample, generate
from litespec.vlm.context import visual_positions
from test_vlm import tiny_qwen35, tiny_image_inputs
from test_vlm_aligned import Processor

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", ["dflash", "dspark"])
def test_baseline_image_train_export_and_cached_decode(family, monkeypatch, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    target = tiny_qwen35().requires_grad_(False)
    cfg = train_aligned.load_config(ROOT / f"scripts/config/vlm/sft_{family}.yaml")
    cfg.update(num_draft_layers=1, block_size=4, num_anchors=2, markov_rank=8)
    draft = baselines.make_draft(target, Processor(), cfg)
    online = baselines.make_online(target, draft, cfg)
    inputs = Processor().apply_chat_template([], add_generation_prompt=False)
    example = PreparedExample(inputs, 8, torch.zeros(8, dtype=torch.bool), "train")
    output = online(**train_aligned.prepare_forward(target, online, example))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in draft.parameters())
    assert all(p.grad is None for p in target.parameters())
    draft.eval()
    draft.save_pretrained(tmp_path / "checkpoint")
    reloaded = baselines.load_draft(tmp_path / "checkpoint", attn_implementation="sdpa").eval()
    assert type(reloaded) is type(draft)
    image = tiny_image_inputs()
    prompt = PreparedExample(image, 8, visual_positions(image["input_ids"][0], target.config), "eval")
    for mode in ("block", "sequential"):
        ar = generate(target, None, prompt, max_new_tokens=7, verification=mode)
        spec = generate(target, reloaded, prompt, max_new_tokens=7, verification=mode)
        assert ar["token_ids"] == spec["token_ids"]
        assert all(row["physical_history"] >= 8 for row in spec["context_counts"])
    # Exercise the actual common optimizer/runner/export, not just forward.
    monkeypatch.setattr(train_aligned, "load_target", lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(dict(id="a", question="describe", answer="square")) + "\n")
    cfg.update(upstream_config=f"scripts/config/{family}.yaml", device="cpu", num_epochs=1,
               train_data=str(manifest), output_dir=str(tmp_path / "run"))
    # load_config's derived identity fields are recomputed.
    cfg.pop("upstream_config_sha256")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    train_aligned.run(["--config", str(config), "--max-steps", "1"])
    assert (tmp_path / "run/last/model.safetensors").exists()
