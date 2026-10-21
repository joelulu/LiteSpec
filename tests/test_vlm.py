from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.vlm.runtime import HistoryBank, PreparedExample, draft_block, generate, prepare_example
from litespec.vlm.context import ContextPolicy, visual_positions


def draft_config(role="student"):
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, layer_types=["full_attention"])
    config.num_target_layers = 4
    config.block_size = 4
    config.dlite_config = dict(model_role=role, chs_num_layers=2,
                               target_layer_ids=[0, 3], sequential_head="rnn",
                               sequential_rank=8, mask_token_id=127)
    if role == "swa_teacher":
        config.dlite_config.update(swa_window_size=64, history_layer_ids=[0, 2, 3])
    config._attn_implementation = "sdpa"
    return config


def test_context_sources_windows_and_no_future():
    vision = torch.tensor([False, True, True, True, False])
    positions = torch.arange(12)
    policy = ContextPolicy("drop", "drop", "keep_last", 2)
    # a=9, pivot=8; last two eligible history positions are 6,7.
    keep = policy.keep(positions, anchor=9, prefill_length=5, vision_mask=vision)
    assert positions[keep].tolist() == [6, 7]
    policy = ContextPolicy("keep", "drop", "drop_recent", 2)
    keep = policy.keep(positions, anchor=9, prefill_length=5, vision_mask=vision)
    assert positions[keep].tolist() == [0, 4, 5]
    assert not ContextPolicy.no_history().keep(positions, anchor=9, prefill_length=5, vision_mask=vision).any()


def test_visual_delimiters_belong_to_visual_interface():
    config = SimpleNamespace(image_token_id=8, vision_start_token_id=7, vision_end_token_id=9)
    assert visual_positions(torch.tensor([1, 7, 8, 8, 9, 2]), config).tolist() == [False, True, True, True, True, False]


def test_invalid_policies_and_student_history_rejected():
    with pytest.raises(ValueError):
        ContextPolicy(decode_mode="keep_last", window_size=0)
    with pytest.raises(ValueError):
        ContextPolicy(decode_mode="all", window_size=4)
    draft = DLiteDraftModel(draft_config())
    with pytest.raises(ValueError, match="no history"):
        draft_block(draft, torch.nn.Embedding(128, 32), anchor_ids=torch.tensor([1]),
                    pivot=torch.randn(1, 1, 2, 32), history=None, anchor=5,
                    prefill_length=5, vision_mask=torch.zeros(5, dtype=torch.bool), policy=ContextPolicy())


def test_compaction_retains_temporarily_masked_recent_tokens():
    bank = HistoryBank(torch.randn(1, 12, 3, 8), torch.arange(12))
    policy = ContextPolicy("drop", "drop", "drop_recent", 2, "compact")
    bank.prune(policy, anchor=12, prefill_length=5, vision_mask=torch.zeros(5, dtype=torch.bool))
    assert bank.positions.tolist() == list(range(5, 12))
    keep = policy.keep(bank.positions, anchor=14, prefill_length=5, vision_mask=torch.zeros(5, dtype=torch.bool))
    assert bank.positions[keep].tolist() == list(range(5, 11))


class FakeTarget(torch.nn.Module):
    """Deterministic next-token rule with a mutable recurrent-style cache."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 32)
        self.config = SimpleNamespace(eos_token_id=120)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, past_key_values=None, **kwargs):
        cache = past_key_values if past_key_values is not None else {"tokens": []}
        cache["tokens"].extend(input_ids[0].tolist())
        # Cumulative rule makes rejected candidates poison later predictions
        # unless the generator restores and replays exactly the accepted prefix.
        prefix = sum(cache["tokens"][:-input_ids.size(1)])
        predictions = []
        for token in input_ids[0].tolist():
            prefix += token
            predictions.append((prefix % 60) + 1)
        logits = torch.full((1, input_ids.size(1), 128), -100.0)
        logits.scatter_(-1, torch.tensor(predictions).view(1, -1, 1), 100.0)
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=logits, past_key_values=cache,
                               hidden_states=tuple(hidden for _ in range(5)))


@pytest.mark.parametrize("perfect", [False, True])
def test_verification_acceptance_and_rejection_preserve_ar(perfect):
    target = FakeTarget()
    draft = DLiteDraftModel(draft_config()).eval()
    example = PreparedExample({"input_ids": torch.tensor([[1, 2, 3]])}, 3,
                               torch.zeros(3, dtype=torch.bool), "fake")
    if perfect:
        prefix_sum = [6]
        def sample_draft_tokens(*, first_prev_token_ids, **kwargs):
            total = prefix_sum[0] + int(first_prev_token_ids.item())
            tokens = []
            for _ in range(3):
                token = total % 60 + 1
                tokens.append(token)
                total += token
            prefix_sum[0] = total
            return torch.tensor([tokens]), None
    else:
        def sample_draft_tokens(**kwargs):
            return torch.zeros(1, 3, dtype=torch.long), None
    draft.sample_draft_tokens = sample_draft_tokens
    ar = generate(target, None, example, max_new_tokens=10)
    spec = generate(target, draft, example, max_new_tokens=10)
    assert ar["token_ids"] == spec["token_ids"]
    assert max(spec["accept_lengths"]) == (4 if perfect else 1)
    assert all(c["physical_history"] == 0 for c in spec["context_counts"])


def tiny_qwen35():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, linear_num_key_heads=4, linear_num_value_heads=4,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         layer_types=["linear_attention"] * 3 + ["full_attention"],
                         eos_token_id=120, pad_token_id=0,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "partial_rotary_factor": 1.0,
                                          "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=1,
                           out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[]),
        image_token_id=91, vision_start_token_id=90, vision_end_token_id=92,
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


def test_rejection_uses_replayed_hidden_not_old_chunk(monkeypatch):
    from litespec.vlm import runtime
    class ChunkSensitiveTarget(FakeTarget):
        def forward(self, input_ids, **kwargs):
            result = super().forward(input_ids, **kwargs)
            if input_ids.size(1) > 1:
                result.hidden_states = tuple(x + 100 for x in result.hidden_states)
            return result
    target = ChunkSensitiveTarget()
    draft = DLiteDraftModel(draft_config()).eval()
    draft.sample_draft_tokens = lambda **kwargs: (torch.zeros(1, 3, dtype=torch.long), None)
    captured = []
    original = runtime.draft_block
    def capture(*args, **kwargs):
        captured.append(kwargs["pivot"].clone())
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, "draft_block", capture)
    example = PreparedExample({"input_ids": torch.tensor([[1, 2, 3]])}, 3,
                              torch.zeros(3, dtype=torch.bool), "replay")
    spec = generate(target, draft, example, max_new_tokens=4)
    expected = target.embedding(torch.tensor([[spec["token_ids"][0]]])).unsqueeze(2).expand(-1, -1, 2, -1)
    torch.testing.assert_close(captured[1], expected)


def tiny_image_inputs():
    return dict(input_ids=torch.tensor([[1, 90, 91, 91, 91, 91, 92, 2]]),
                pixel_values=torch.randn(4, 24), image_grid_thw=torch.tensor([[1, 2, 2]]),
                mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]))


def test_real_tiny_qwen35_image_prefill_and_hybrid_rollback():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    target = tiny_qwen35()
    draft = DLiteDraftModel(draft_config()).eval()
    inputs = tiny_image_inputs()
    example = PreparedExample(inputs, 8, visual_positions(inputs["input_ids"][0], target.config), "tiny")
    # Force rejections to exercise actual GDN convolution/recurrent recovery.
    draft.sample_draft_tokens = lambda **kw: (torch.zeros(1, 3, dtype=torch.long), None)
    ar = generate(target, None, example, max_new_tokens=6)
    spec = generate(target, draft, example, max_new_tokens=6)
    assert ar["token_ids"] == spec["token_ids"]
    assert spec["recovery_seconds"] > 0
    with torch.no_grad():
        original = target(**inputs).logits
        changed = target(**{**inputs, "pixel_values": inputs["pixel_values"] + 5}).logits
    assert not torch.allclose(original[:, -1], changed[:, -1], atol=1e-7)


def test_processor_supervision_boundary_and_no_truncation(tmp_path):
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            ids = [1, 2, 3] if kwargs["add_generation_prompt"] else [1, 2, 3, 4, 5]
            return {"input_ids": torch.tensor([ids])}
    record = {"question": "describe", "answer": "a small image"}
    example = prepare_example(Processor(), SimpleNamespace(), record, device="cpu", training=True)
    assert example.prefill_length == 3
    with pytest.raises(ValueError, match="not truncating"):
        prepare_example(Processor(), SimpleNamespace(), record, device="cpu", training=True, max_length=4)
