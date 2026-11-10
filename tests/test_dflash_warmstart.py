"""Checkpoint metadata/launcher checks, plus a real tiny-model check when installed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dflash_warmstart", ROOT / "litespec/vlm/dflash_warmstart.py")
warm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm)


class WarmStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "source"
        self.path.mkdir()
        self.target = self.root / "target"
        self.target.mkdir()
        self.parent = dict(
            architectures=["DFlashDraftModel"], block_size=8, num_hidden_layers=5,
            hidden_size=32, vocab_size=128, num_target_layers=32,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            dflash_config=dict(target_layer_ids=[1, 8, 15, 22, 29], mask_token_id=127),
            vlm_config=dict(target_model=str(self.target), draft_context="full"),
        )
        self.write_parent()
        self.cfg = dict(algorithm="dflash", draft_context="pivot1", block_size=8,
                        num_draft_layers=5, target_model=str(self.target), context_policy={})

    def write_parent(self):
        (self.path / "config.json").write_text(json.dumps(self.parent))

    def test_both_modes_record_same_source_layers_and_config_hash(self):
        configs = []
        for mode in ("full", "pivot1"):
            cfg = dict(self.cfg, draft_context=mode)
            warm.warm_start_config(cfg, self.path)
            self.assertEqual(cfg["target_layer_ids"], [1, 8, 15, 22, 29])
            self.assertEqual(cfg["init_checkpoint"], str(self.path))
            configs.append(cfg)
        self.assertEqual(configs[0]["init_config_sha256"], configs[1]["init_config_sha256"])

    def test_rejects_structure_layer_order_and_unsupported_modes(self):
        for cfg, message in (
            (dict(self.cfg, block_size=7), "block_size"),
            (dict(self.cfg, num_draft_layers=6), "num_draft_layers"),
            (dict(self.cfg, target_layer_ids=[29, 22, 15, 8, 1]), "target_layer_ids"),
            (dict(self.cfg, draft_context="pivot5"), "Full/Pivot1"),
            (dict(self.cfg, pivot_two_rounds=True), "two-round"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                warm.warm_start_config(cfg, self.path)

    def test_rejects_wrong_family_or_nonfull_source(self):
        self.parent["architectures"] = ["DLiteDraftModel"]
        self.write_parent()
        with self.assertRaisesRegex(ValueError, "DFlashDraftModel"):
            warm.warm_start_config(self.cfg, self.path)
        self.parent["architectures"] = ["DFlashDraftModel"]
        self.parent["vlm_config"]["draft_context"] = "pivot1"
        self.write_parent()
        with self.assertRaisesRegex(ValueError, "original Full"):
            warm.warm_start_config(self.cfg, self.path)

    def test_rejects_invalid_mask_and_source_layer_indices(self):
        self.parent["dflash_config"]["mask_token_id"] = 128
        self.write_parent()
        with self.assertRaisesRegex(ValueError, "MASK"):
            warm.warm_start_config(self.cfg, self.path)
        self.parent["dflash_config"]["mask_token_id"] = 127
        self.parent["dflash_config"]["target_layer_ids"] = [32]
        self.write_parent()
        with self.assertRaisesRegex(ValueError, "outside Target"):
            warm.warm_start_config(self.cfg, self.path)

    def fake_loader(self, info):
        captured = {}

        def load(path, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(context_window=1, to=lambda **kw: captured), info

        modules = {
            "transformers": SimpleNamespace(AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(**self.parent))),
            "litespec.modeling.draft.dflash": SimpleNamespace(DFlashDraftModel=SimpleNamespace(from_pretrained=load)),
        }
        text = SimpleNamespace(**{k: self.parent[k] for k in
                                 ("hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim")},
                               num_hidden_layers=32)
        target = SimpleNamespace(config=SimpleNamespace(text_config=text), device="cpu", dtype="float32")
        return modules, target, captured

    def test_loader_changes_visibility_and_records_provenance(self):
        modules, target, captured = self.fake_loader({})
        with patch.dict(sys.modules, modules):
            warm.load_warm_start(self.path, target, self.cfg)
        self.assertEqual(captured["config"].vlm_config["draft_context"], "pivot1")
        self.assertEqual(captured["config"].vlm_config["warm_start"], str(self.path))
        self.assertTrue(captured["output_loading_info"])

    def test_incomplete_weights_fail_and_target_mismatch_fails(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            modules, target, _ = self.fake_loader({key: ["weight"]})
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "every parameter"):
                warm.load_warm_start(self.path, target, dict(self.cfg))
        modules, target, _ = self.fake_loader({})
        target.config.text_config.vocab_size += 1
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "vocab_size"):
            warm.load_warm_start(self.path, target, dict(self.cfg))


class PresetTests(unittest.TestCase):
    def test_recorded_and_warmstart_presets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "worktree"
            for name in ("train_dflash_context_pair.sh", "train_dflash_context_10w_pair.sh", "train_dflash_context_20k_warmstart_pair.sh"):
                dest = repo / "tests" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text((ROOT / "tests" / name).read_text())
            launcher = repo / "scripts/dlite/run_training_vlm.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_text('''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$CUDA_VISIBLE_DEVICES" > "$OUTPUT_DIR/gpus.txt"
mkdir -p "$OUTPUT_DIR/last"
printf '{}\\n' > "$OUTPUT_DIR/last/config.json"
''')
            python = root / "python"
            python.write_text("#!" + sys.executable + "\n" + '''
import json, os, pathlib, sys
assert sys.argv[1] == "tests/dflash_context_config.py"
pathlib.Path(sys.argv[3]).write_text(json.dumps(dict(mode=sys.argv[2], epochs=os.environ['EPOCHS'],
    micro=os.environ['MICRO_BATCH_SIZE'], accumulation=os.environ['ACCUMULATION_STEPS'],
    data=os.environ['TRAIN_DATA'], maximum=os.environ['MAX_SAMPLES'], init=os.environ.get('INIT_CHECKPOINT'))))
''')
            python.chmod(0o755)
            for dataset in ("finevision_qwen35_10w_img1024_relaxed", "finevision_qwen35_gdn_valid20k_img1024_b8_s42"):
                data = root / "data" / dataset / "train.jsonl"
                data.parent.mkdir(parents=True)
                data.write_text('{}\n')
            for cache in ("preflight_10w_img1024_parallel_20261110_092423_ccec48bb",
                          "dflash_context_20k_1ep_20261110_041837_1299222/full"):
                directory = root / "runs" / cache
                directory.mkdir(parents=True)
                (directory / "effective_config.json").write_text('{}')
                (directory / "skipped_training.jsonl").touch()
            source = root / "runs/vlm_dlite_dflash_3ep_mb2_b8_20261022_060311_3121049/dflash/last"
            source.mkdir(parents=True)
            (source / "config.json").write_text('{}')

            for preset in ("10w", "20k_warmstart"):
                output = root / preset
                env = dict(os.environ, DLITE_WORKSPACE=str(root), PYTHON_BIN=str(python), RUN_GROUP=str(output))
                for key in ("INIT_CHECKPOINT", "VALIDATION_FROM", "FULL_GPUS", "PIVOT_GPUS"):
                    env.pop(key, None)
                result = subprocess.run(["bash", str(repo / "tests" / f"train_dflash_context_{preset}_pair.sh")],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                configs = [json.loads((output / mode / "config.yaml").read_text()) for mode in ("full", "pivot1")]
                self.assertEqual(configs[0]["init"], configs[1]["init"])
                self.assertEqual(configs[0]["init"], None if preset == "10w" else str(source))
                for config in configs:
                    self.assertEqual((config["epochs"], config["micro"], config["accumulation"]), ("2", "4", "16"))
                    self.assertEqual(config["maximum"], "0" if preset == "10w" else "20000")
                self.assertEqual((output / "full/gpus.txt").read_text().strip(), "0,1,2,3")
                self.assertEqual((output / "pivot1/gpus.txt").read_text().strip(), "4,5,6,7")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "requires torch/transformers")
class RealWeightTests(unittest.TestCase):
    def test_full_and_pivot1_load_identical_real_parameters(self):
        import torch
        from transformers import Qwen3Config
        from litespec.modeling.draft.dflash import DFlashDraftModel
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=['full_attention'])
            config.architectures = ['DFlashDraftModel']
            config.block_size, config.num_target_layers = 4, 4
            config.dflash_config = dict(target_layer_ids=[0], mask_token_id=127)
            config.vlm_config = dict(target_model=str(root / 'target'), draft_context='full')
            original = DFlashDraftModel(config).eval()
            original.save_pretrained(root / 'source')
            text = SimpleNamespace(hidden_size=32, vocab_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8)
            target = SimpleNamespace(config=SimpleNamespace(text_config=text), device='cpu', dtype=torch.float32)
            for mode in ('full', 'pivot1'):
                cfg = dict(algorithm='dflash', draft_context=mode, block_size=4, num_draft_layers=1,
                           target_model=str(root / 'target'), context_policy={})
                loaded = warm.load_warm_start(root / 'source', target, cfg)
                self.assertEqual(loaded.context_window, None if mode == 'full' else 1)
                self.assertEqual(set(loaded.state_dict()), set(original.state_dict()))
                for key, tensor in original.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
