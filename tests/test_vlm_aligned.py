import json
from pathlib import Path

import pytest
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
import yaml

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm import train_aligned as aligned
from litespec.vlm.runtime import PreparedExample, visual_positions
from test_vlm import draft_config, tiny_image_inputs, tiny_qwen35

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/config/vlm/sft_student_aligned.yaml"


def test_default_recipe_inherits_every_upstream_training_hyperparameter():
    base = yaml.safe_load((ROOT / "scripts/config/dlite_sft.yaml").read_text())["arguments"]
    cfg = aligned.load_config(CONFIG)
    for key, value in base.items():
        if key not in {"target_model_backend", "report_to", "save_interval", "log_interval"}:
            assert cfg[key] == value, key
    assert cfg["num_epochs"] == 6
    assert cfg["num_draft_layers"] == 5
    assert cfg["sequential_rank"] == 320
    assert "max_steps" not in cfg


def fixture_model():
    torch.manual_seed(21)
    torch.set_num_threads(1)
    target = tiny_qwen35().requires_grad_(False)
    draft = DLiteDraftModel(draft_config())
    cfg = aligned.load_config(CONFIG)
    cfg.update(block_size=4, num_anchors=4)
    online = aligned.make_online(target, draft, cfg)
    inputs = tiny_image_inputs()
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 120]])], 1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 9, dtype=torch.long)], 1)
    example = PreparedExample(inputs, 8, visual_positions(inputs["input_ids"][0, :8], target.config), "upstream")
    return target, draft, online, example


def test_image_target_gather_and_upstream_weighted_greedy_loss():
    target, draft, online, example = fixture_model()
    assert aligned.has_anchor(example, 4)
    kwargs = aligned.prepare_forward(target, online, example)
    batch = kwargs["prepared_batch"]
    assert batch.anchor_positions.min() >= example.prefill_length
    with torch.no_grad():
        full = target(**example.inputs, use_cache=False).logits
        positions = batch.anchor_positions.unsqueeze(-1) + torch.arange(3)
        reference = full[:, positions[0]]
    torch.testing.assert_close(reference, kwargs["target_prefill_logits"], rtol=1e-5, atol=1e-6)
    hidden = online.forward_backbone(batch, seq_len=kwargs["seq_len"])
    serial = draft.sequential_head.forward_teacher_forcing(hidden_states=hidden, prev_token_ids=batch.prev_token_ids)
    logits = draft.sequential_head.project_logits(serial).float()
    weights = online._position_weights(batch.raw_weight_mask, 4)
    teacher = kwargs["target_prefill_logits"].float()
    ce = F.cross_entropy(logits.reshape(-1, 128), teacher.argmax(-1).reshape(-1), reduction="none").reshape(weights.shape)
    l1 = (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
    expected = ((0.1 * ce + l1) * weights).sum() / weights.sum()
    output = online(**kwargs)
    torch.testing.assert_close(output.loss, expected)
    output.loss.backward()
    assert draft.sequential_head.output_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


class Processor:
    tokenizer = type("Tokenizer", (), {"get_vocab": lambda self: {str(i): i for i in range(121)}})()

    def apply_chat_template(self, messages, **kwargs):
        inputs = tiny_image_inputs()
        if not kwargs["add_generation_prompt"]:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 120]])], 1)
            inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 9, dtype=torch.long)], 1)
        return inputs


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_six_epoch_muon_runner_accumulation_tail_and_exact_resume(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(aligned, "load_target", lambda *a, **kw: (tiny_qwen35().requires_grad_(False), Processor()))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(dict(id=str(i), question="describe", answer="a square")) for i in range(5)))
    configs = {}
    for name in ("full", "resumed"):
        cfg = dict(upstream_config="scripts/config/dlite_sft.yaml", target_model="tiny", train_data=str(manifest),
                   output_dir=str(tmp_path / name), device="cpu", model_role="student",
                   report_to="none", log_interval=1, save_interval=4000,
                   num_draft_layers=1, chs_num_layers=2, sequential_rank=8, block_size=4,
                   num_anchors=3)
        path = tmp_path / (name + ".yaml")
        path.write_text(yaml.safe_dump(cfg))
        configs[name] = path
    aligned.run(["--config", str(configs["full"]), "--save-training-state"])
    export = tmp_path / "weights_only"
    aligned.run(["--config", str(configs["full"]), "--output-dir", str(export), "--max-steps", "1"])
    assert (export / "last/model.safetensors").is_file()
    assert not list((export / "last").glob("training_state_rank_*.pt"))
    assert json.loads((export / "last/checkpoint.json").read_text())["resumable"] is False
    with pytest.raises(ValueError, match="inference weights only"):
        aligned.run(["--config", str(configs["full"]), "--resume", str(export / "last")])
    original_log = aligned.log

    def interrupt(session, window, grad_norm):
        original_log(session, window, grad_norm)
        if session.state.optimizer_step == 2:
            raise RuntimeError("simulated interruption at an optimizer boundary")

    monkeypatch.setattr(aligned, "log", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        aligned.run(["--config", str(configs["resumed"]), "--save-training-state"])
    monkeypatch.setattr(aligned, "log", original_log)
    last = tmp_path / "resumed/last"
    aligned.run(["--config", str(configs["resumed"]), "--resume", str(last), "--save-training-state"])
    full = tmp_path / "full/last"
    assert_tree_equal(load_file(full / "model.safetensors"), load_file(last / "model.safetensors"))
    a = torch.load(full / "training_state_rank_00000.pt", weights_only=False)
    b = torch.load(last / "training_state_rank_00000.pt", weights_only=False)
    assert a["cursor"]["global_step"] == b["cursor"]["global_step"] == 30
    assert a["cursor"]["optimizer_step"] == b["cursor"]["optimizer_step"] == 8
    assert_tree_equal(a["optimizer"], b["optimizer"])
    assert all(p.dtype == torch.float32 for p in b["optimizer"]["fp32_master_params"])
    assert_tree_equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"]


def test_short_answers_are_reported_instead_of_changing_upstream_anchor_semantics():
    target, draft, online, example = fixture_model()
    example.prefill_length = example.inputs["input_ids"].size(1) - 2
    assert not aligned.has_anchor(example, 4)


def test_sequential_verification_matches_ar_on_real_image_target():
    from litespec.vlm.runtime import generate
    target, draft, _, example = fixture_model()
    inputs = {k: v[:, :8] if k in ("input_ids", "mm_token_type_ids") else v for k, v in example.inputs.items()}
    example.inputs = inputs
    draft.eval()
    draft.sample_draft_tokens = lambda **kw: (torch.zeros(1, 3, dtype=torch.long), None)
    ar = generate(target, None, example, max_new_tokens=12, verification="sequential")
    spec = generate(target, draft, example, max_new_tokens=12, verification="sequential")
    assert ar["token_ids"] == spec["token_ids"]
    assert spec["recovery_seconds"] == 0
    assert sum(spec["accept_lengths"]) == len(spec["token_ids"])
