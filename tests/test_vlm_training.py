import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm.context import ContextPolicy
from litespec.vlm.runtime import PreparedExample, draft_block, visual_positions
from litespec.vlm.train import train_block
from test_vlm import draft_config, tiny_image_inputs, tiny_qwen35


def test_history_mask_and_compact_match_outputs_and_gradients():
    torch.manual_seed(13)
    model = DLiteDraftModel(draft_config("swa_teacher"))
    embedding = torch.nn.Embedding(128, 32).requires_grad_(False)
    pivot = torch.randn(1, 1, 2, 32)
    history = torch.randn(1, 10, 3, 32)
    vision = torch.tensor([False, True, True, True, False])
    outputs, gradients, counts = [], [], []
    for mode in ("mask", "compact"):
        model.zero_grad(set_to_none=True)
        output, count = draft_block(
            model, embedding, anchor_ids=torch.tensor([5]), pivot=pivot,
            history=history, anchor=10, prefill_length=5, vision_mask=vision,
            policy=ContextPolicy("drop", "drop", "keep_last", 2, mode),
        )
        output.square().mean().backward()
        outputs.append(output.detach())
        gradients.append(model.layers[0].self_attn.k_proj.weight.grad.clone())
        counts.append(count)
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(gradients[0], gradients[1], atol=2e-6, rtol=2e-5)
    assert counts[0]["physical_history"] == 9
    assert counts[1]["physical_history"] == counts[0]["visible_history"] == 2


@pytest.mark.parametrize("role", ["student", "swa_teacher"])
def test_real_image_training_backward_and_optimizer_restore(role, tmp_path):
    torch.manual_seed(21)
    torch.set_num_threads(1)
    target = tiny_qwen35().requires_grad_(False)
    draft = DLiteDraftModel(draft_config(role))
    inputs = tiny_image_inputs()
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])], dim=1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 8, dtype=torch.long)], dim=1)
    example = PreparedExample(inputs, 8, visual_positions(inputs["input_ids"][0, :8], target.config), "training")
    policy = ContextPolicy.no_history() if role == "student" else ContextPolicy()
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = train_block(target, draft, example, [8, 10], policy=policy)
        assert torch.isfinite(loss)
        loss.backward()
        assert draft.sequential_head.output_proj.weight.grad.abs().sum() > 0
        if role != "student":
            assert draft.history_fuse.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in target.parameters())
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    draft.save_pretrained(tmp_path)
    restored = DLiteDraftModel.from_pretrained(tmp_path, attn_implementation="sdpa")
    for key, tensor in draft.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key])
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(torch.load(tmp_path / "optimizer.pt", weights_only=True))
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_student_checkpoint_loads_in_inference_repo(tmp_path):
    repo = os.environ.get("DLITE_REPO_PATH")
    if not repo:
        pytest.skip("set DLITE_REPO_PATH for cross-repository checkpoint validation")
    draft = DLiteDraftModel(draft_config())
    draft.save_pretrained(tmp_path)
    code = "from dlite.modeling import DLiteDraftModel; import sys; m=DLiteDraftModel.from_pretrained(sys.argv[1]); assert m.is_student"
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True,
                   cwd=repo, env={**os.environ, "PYTHONPATH": str(Path(repo).resolve())})


def test_short_answer_partial_block_is_supervised():
    target = tiny_qwen35().requires_grad_(False)
    draft = DLiteDraftModel(draft_config())
    inputs = tiny_image_inputs()
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 120]])], dim=1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 2, dtype=torch.long)], dim=1)
    example = PreparedExample(inputs, 8, visual_positions(inputs["input_ids"][0, :8], target.config), "short")
    loss = train_block(target, draft, example, [8], policy=ContextPolicy.no_history())
    assert torch.isfinite(loss)
    loss.backward()
    assert draft.sequential_head.output_proj.weight.grad.abs().sum() > 0


def test_cli_interrupted_training_resumes_record_and_optimizer(monkeypatch, tmp_path):
    import json
    import yaml
    from litespec.vlm import train

    target = tiny_qwen35().requires_grad_(False)

    class Processor:
        tokenizer = type("Tokenizer", (), {"get_vocab": lambda self: {str(i): i for i in range(121)}})()

        def apply_chat_template(self, messages, **kwargs):
            inputs = tiny_image_inputs()
            if not kwargs["add_generation_prompt"]:
                inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 120]])], dim=1)
                inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 3, dtype=torch.long)], dim=1)
            return inputs

    manifest = tmp_path / "train.jsonl"
    manifest.write_text('\n'.join(json.dumps({"id": str(i), "question": "describe", "answer": "a square"}) for i in range(2)))
    output = tmp_path / "output"
    config = dict(target_model="tiny", train_data=str(manifest), output_dir=str(output),
                  device="cpu", max_steps=2, save_steps=1, block_size=4,
                  num_draft_layers=1, chs_num_layers=2, sequential_rank=8,
                  context_policy=ContextPolicy.no_history().to_dict())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(train, "load_target", lambda *a, **kw: (target, Processor()))
    original = train.train_block
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "train_block", interrupt)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.main()
    saved = torch.load(output / "last/vlm_training_state.pt", weights_only=False)
    assert saved["state"] == {"step": 1, "epoch": 0, "record": 1}
    monkeypatch.setattr(train, "train_block", original)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--resume", str(output / "last")])
    train.main()
    saved = torch.load(output / "last/vlm_training_state.pt", weights_only=False)
    assert saved["state"] == {"step": 2, "epoch": 0, "record": 2}
