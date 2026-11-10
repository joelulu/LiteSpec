"""Check actual epoch export/config code without importing GPU dependencies."""
import argparse
import ast
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace):
    source = ast.parse((ROOT / path).read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@dataclass
class Cursor:
    optimizer_step: int = 1
    global_step: int = 16
    epoch: int = 0


class EpochSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exports = []

        def export(path, state_dict):
            self.exports.append(Path(path))
            (Path(path) / "model.safetensors").write_text(json.dumps(state_dict))

        self.session = SimpleNamespace(
            args=SimpleNamespace(output_dir=str(self.root)),
            vlm_config=dict(final_only=False, save_every_epoch=True),
            state=Cursor(), phases=[SimpleNamespace(final_checkpoint="last")],
            save_training_state=False, fsdp_model=None,
            draft=SimpleNamespace(state_dict=lambda: {"weight": 1}, save_pretrained=export),
        )
        self.ns = functions("litespec/vlm/train_aligned.py", {"checkpoint_epoch_link", "save"}, dict(
            Path=Path, os=os, uuid=uuid, json=json, asdict=asdict,
            rank=lambda: 0, world=lambda: 1,
            dist=SimpleNamespace(is_initialized=lambda: False), FSDP=type("FSDP", (), {}),
        ))

    def test_epoch_aliases_and_final_export_do_not_duplicate_weights(self):
        save = self.ns["save"]
        save(self.session, "epoch_1")
        first = self.root / "step_0000001"
        self.assertEqual((self.root / "epoch_1").resolve(), first)
        self.assertEqual((self.root / "last").resolve(), first)
        save(self.session, "epoch_2")
        self.assertEqual(len(self.exports), 1)
        self.assertEqual((self.root / "epoch_2").resolve(), first)
        self.session.state.optimizer_step = 2
        save(self.session, "epoch_2")
        second = self.root / "step_0000002"
        self.assertEqual((self.root / "epoch_1").resolve(), first)
        self.assertEqual((self.root / "epoch_2").resolve(), second)
        self.assertEqual((self.root / "last").resolve(), second)
        save(self.session, "last")
        self.assertEqual(len(self.exports), 2)
        self.assertFalse(list(self.root.glob(".checkpoint-*")))

    def test_final_only_default_is_preserved(self):
        self.session.vlm_config = dict(final_only=True)
        self.ns["save"](self.session, "epoch_1")
        self.assertEqual(self.exports, [])
        self.ns["save"](self.session, "last")
        self.assertTrue((self.root / "last/model.safetensors").is_file())
        self.assertFalse((self.root / "last").is_symlink())
        self.assertFalse((self.root / "epoch_1").exists())

    def test_logging_exports_each_completed_epoch_without_repeating(self):
        saved = []
        namespace = functions("litespec/vlm/train_aligned.py", {"log"}, dict(
            rank=lambda: 1, save=lambda session, name: saved.append(name),
        ))
        self.session.state.global_step = 20
        self.session.steps_per_epoch = 10
        self.session.last_exported_epoch = 0
        self.session.progress = SimpleNamespace(update=lambda step: {})
        self.session.tracker = SimpleNamespace(log=lambda *args, **kw: None)
        self.session.optimizer = SimpleNamespace(get_learning_rates=lambda: {})
        window = SimpleNamespace(metrics=SimpleNamespace(reduce=lambda: {"objective/loss": 1.0}))
        namespace["log"](self.session, window, 0.0)
        namespace["log"](self.session, window, 0.0)
        self.assertEqual(saved, ["epoch_1", "epoch_2"])
        self.assertEqual(self.session.last_exported_epoch, 2)


class ConfigTests(unittest.TestCase):
    def build(self):
        namespace = functions("tests/dflash_context_config.py", {"build"}, dict(
            os=os, CONTEXTS=("full", "pivot1"),
            Path=lambda path: SimpleNamespace(read_text=lambda: ""),
            yaml=SimpleNamespace(safe_load=lambda text: {}),
        ))
        return namespace["build"]

    def test_config_switch_for_both_contexts(self):
        for flag in ("0", "1"):
            with patch.dict(os.environ, {"SAVE_EVERY_EPOCH": flag, "EPOCHS": "2"}, clear=True):
                for mode in ("full", "pivot1"):
                    cfg = self.build()(mode)
                    self.assertEqual(cfg["save_every_epoch"], flag == "1")
                    self.assertEqual(cfg["final_only"], flag == "0")
                    self.assertEqual(cfg["num_epochs"], 2)
                    self.assertNotIn("init_checkpoint", cfg)

    def test_invalid_environment_flag_fails(self):
        with patch.dict(os.environ, {"SAVE_EVERY_EPOCH": "typo"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SAVE_EVERY_EPOCH"):
                self.build()("full")

    def test_runner_respects_epoch_flag_before_loading_target(self):
        class TargetReached(Exception):
            pass

        for flag in (False, True, "false"):
            with tempfile.TemporaryDirectory() as temp:
                cfg = dict(dflash_context_experiment=True, save_every_epoch=flag, final_only=True,
                           seed=42, device="cpu", target_model=temp, output_dir=str(Path(temp) / "out"),
                           train_data=str(Path(temp) / "data.jsonl"))
                namespace = functions("litespec/vlm/train_aligned.py", {"run"}, dict(
                    argparse=argparse, load_config=lambda path: cfg, Path=Path, os=os,
                    random=random, np=SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)),
                    torch=SimpleNamespace(manual_seed=lambda seed: None, float32="float32"),
                    dist=SimpleNamespace(is_initialized=lambda: False), world=lambda: 1,
                    load_target=lambda *args, **kw: (_ for _ in ()).throw(TargetReached()),
                ))
                with patch.dict(os.environ, {}, clear=True):
                    if isinstance(flag, bool):
                        with self.assertRaises(TargetReached):
                            namespace["run"](["--config", "unused"])
                        self.assertEqual(cfg["final_only"], not flag)
                    else:
                        with self.assertRaisesRegex(ValueError, "save_every_epoch"):
                            namespace["run"](["--config", "unused"])


if __name__ == "__main__":
    unittest.main()
