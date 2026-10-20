import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml
from safetensors.torch import load_file

from litespec.vlm.context import ContextPolicy
from litespec.vlm.train import epoch_indices


@pytest.mark.parametrize("size", [1, 7, 8, 19, 20000])
def test_eight_rank_shards_cover_data_with_only_tail_padding(size):
    shards = [epoch_indices(size, rank, 8, seed=42, epoch=0, shuffle=True) for rank in range(8)]
    assert len({len(shard) for shard in shards}) == 1
    flat = [index for shard in shards for index in shard]
    assert set(flat) == set(range(size))
    assert len(flat) - size == (-size) % 8
    assert shards[0] == epoch_indices(size, 0, 8, seed=42, epoch=0, shuffle=True)


def test_torchrun_image_training_and_exact_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("\n".join(json.dumps({"id": str(i), "question": "describe", "answer": "image"}) for i in range(9)))
    paths = {}
    for name in ("full", "resumed"):
        output = tmp_path / name
        cfg = dict(target_model="tiny", train_data=str(manifest), output_dir=str(output),
                   device="cpu", max_steps=3, epochs=2, save_steps=1, block_size=4,
                   num_draft_layers=1, chs_num_layers=2, sequential_rank=8, shuffle=True,
                   context_policy=ContextPolicy.no_history().to_dict())
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump(cfg))
        paths[name] = (config, output)

    def launch(mode, name):
        config, output = paths[name]
        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--rdzv-backend=c10d",
             "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1", "--nproc-per-node=2",
             "tests/vlm_distributed_worker.py", mode, str(config), str(output)],
            cwd=root, env={**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "1",
                           "GLOO_SOCKET_IFNAME": "lo"},
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode and "gloo/transport/tcp/device.cc" in result.stderr and "Operation not permitted" in result.stderr:
            pytest.skip("execution sandbox blocks Gloo TCP device creation; run on the training host")
        assert result.returncode == 0, result.stdout + result.stderr
        return output / "last"

    full = launch("full", "full")
    interrupted = launch("interrupt", "resumed")
    saved = torch.load(interrupted / "vlm_training_state.pt", weights_only=False)
    assert saved["state"]["step"] == 1
    assert len(saved["rng_by_rank"]) == 2
    resumed = launch("resume", "resumed")
    a, b = load_file(full / "model.safetensors"), load_file(resumed / "model.safetensors")
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    left = torch.load(full / "vlm_training_state.pt", weights_only=False)
    right = torch.load(resumed / "vlm_training_state.pt", weights_only=False)
    assert left["state"] == right["state"] == {"step": 3, "epoch": 0, "record": 3}
    for first, second in zip(left["rng_by_rank"], right["rng_by_rank"]):
        assert first["python_rng"] == second["python_rng"]
        assert torch.equal(first["torch_rng"], second["torch_rng"])
    for param_id, state in left["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["optimizer"]["state"][param_id][key], rtol=0, atol=0)


def test_objective_wrapper_preserves_image_loss_and_gradients():
    from copy import deepcopy
    from test_vlm import draft_config, tiny_image_inputs, tiny_qwen35
    from litespec.modeling.draft.dlite import DLiteDraftModel
    from litespec.vlm.runtime import PreparedExample, visual_positions
    from litespec.vlm.train import TrainingObjective, train_block

    torch.manual_seed(21)
    torch.set_num_threads(1)
    target = tiny_qwen35().requires_grad_(False)
    original = DLiteDraftModel(draft_config())
    wrapped = deepcopy(original)
    inputs = tiny_image_inputs()
    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[4, 5, 6, 7, 120]])], dim=1)
    inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros(1, 5, dtype=torch.long)], dim=1)
    example = PreparedExample(inputs, 8, visual_positions(inputs["input_ids"][0, :8], target.config), "wrapper")
    kwargs = dict(policy=ContextPolicy.no_history(), l1_weight=0.1)
    a = train_block(target, original, example, [8, 10], **kwargs)
    b = TrainingObjective(wrapped)(target, example, [8, 10], **kwargs)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    b.backward()
    for left, right in zip(original.parameters(), wrapped.parameters()):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert all(p.grad is None for p in target.parameters())
