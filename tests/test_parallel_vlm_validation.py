"""Stdlib orchestration tests; real eligibility functions, fake image processing.

No GPU/model dependencies are required. Server-side image processing is covered
by the same validate_record function used by the serial training runner.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_vlm_validation", ROOT / "tests/prepare_vlm_validation.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class Progress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def update(self, count):
        pass


def real_training_rules(prepare):
    """Execute the repository's actual rules without importing CUDA dependencies."""
    source = ast.parse((ROOT / "litespec/vlm/train_aligned.py").read_text())
    names = {"validate_record", "validate_records", "reuse_validation"}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(
        prepare_example=prepare, has_anchor=lambda example, block: example.anchor,
        loss_mask=lambda example: example, Path=Path, json=json, hashlib=hashlib,
        __name__="litespec.vlm.train_aligned", __package__="litespec.vlm",
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_training_rules", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.data = self.root / "train.jsonl"
        self.records = [dict(id=str(i), kind=("valid", "bad", "short", "empty_error")[i % 4]) for i in range(40)]
        self.data.write_text("".join(json.dumps(row) + "\n" for row in self.records))
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(dict(max_length=2048, block_size=8, image_max_tokens=1024)))
        self.calls = []
        self.owners = set()
        self.mutex = threading.Lock()

        def prepare(processor, config, record, **kwargs):
            self.assertEqual(processor.owner, threading.get_ident())
            with self.mutex:
                self.calls.append(record["id"])
                self.owners.add(processor.owner)
            time.sleep(0.001 * (int(record["id"]) % 3))
            if record["kind"] == "bad":
                raise ValueError("too long")
            if record["kind"] == "empty_error":
                raise ValueError("")
            if record["kind"] == "io_error":
                raise OSError("image storage unavailable")
            return SimpleNamespace(anchor=record["kind"] == "valid", index=int(record["id"]))

        train = real_training_rules(prepare)
        train.load_config = lambda path: json.loads(Path(path).read_text())
        train.read_records = lambda path, limit: [json.loads(line) for line in Path(path).read_text().splitlines()][:limit]
        self.deps = SimpleNamespace(
            train=train, tqdm=Progress,
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace()),
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(owner=threading.get_ident())),
            configure=lambda processor, config, tokens: dict(longest_edge=tokens),
        )
        self.args = SimpleNamespace(
            config=str(self.config), train_data=str(self.data), target_model=str(self.model),
            max_samples=0, cache_root=str(self.root / "cache"), search_root=[],
            reuse_validation_from=None, force=False, workers=4, chunk_size=2,
        )
        self.patcher = patch.object(cli, "load_dependencies", return_value=self.deps)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_parallel_matches_real_serial_filter_and_order(self):
        directory = cli.prepare(self.args)
        saved = json.loads((directory / "effective_config.json").read_text())
        with patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=Progress)}):
            serial = self.deps.train.validate_records(
                self.records, SimpleNamespace(owner=threading.get_ident()), None, saved, self.root,
            )
        actual = self.deps.train.reuse_validation(self.records, saved, directory)
        self.assertEqual(actual, serial)
        self.assertGreater(len(self.owners), 1)
        self.assertTrue((directory / "reuse_validation.env").is_file())
        self.assertFalse(list(Path(self.args.cache_root).glob(".building-*")))

    def test_completed_cache_reused_without_processing_images(self):
        first = cli.prepare(self.args)
        self.calls.clear()
        self.assertEqual(cli.prepare(self.args), first)
        self.assertEqual(self.calls, [])

    def test_concurrent_jobs_build_only_once(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: cli.prepare(self.args), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.calls), len(self.records))
        self.assertEqual(len(list(Path(self.args.cache_root).rglob("effective_config.json"))), 1)

    def test_explicit_mismatch_fails_and_automatic_search_builds_new_cache(self):
        first = cli.prepare(self.args)
        config = json.loads(self.config.read_text())
        config["image_max_tokens"] = 512
        self.config.write_text(json.dumps(config))
        self.args.reuse_validation_from = str(first)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "image_max_tokens"):
            cli.prepare(self.args)
        self.assertEqual(self.calls, [])
        self.args.reuse_validation_from = None
        self.assertNotEqual(cli.prepare(self.args), first)

    def test_corrupt_cache_rejected_and_rebuilt(self):
        first = cli.prepare(self.args)
        with (first / "skipped_training.jsonl").open("a") as handle:
            handle.write((first / "skipped_training.jsonl").read_text().splitlines()[0] + "\n")
        self.assertNotEqual(cli.prepare(self.args), first)

    def test_unexpected_error_does_not_publish_cache(self):
        self.records[0]["kind"] = "io_error"
        self.data.write_text("".join(json.dumps(row) + "\n" for row in self.records))
        with self.assertRaisesRegex(OSError, "image storage"):
            cli.prepare(self.args)
        self.assertFalse(list(Path(self.args.cache_root).rglob("effective_config.json")))

    def test_force_rechecks_and_sample_limit_does_not_reuse_full_cache(self):
        first = cli.prepare(self.args)
        self.args.force = True
        self.calls.clear()
        second = cli.prepare(self.args)
        self.assertNotEqual(first, second)
        self.assertEqual(len(self.calls), len(self.records))
        self.args.force = False
        self.calls.clear()
        self.assertEqual(cli.prepare(self.args), second)
        self.assertEqual(self.calls, [])
        self.args.max_samples = 5
        third = cli.prepare(self.args)
        saved = json.loads((third / "effective_config.json").read_text())
        self.assertEqual(saved["debug_max_samples"], 5)
        self.assertNotIn(third, [first, second])

    def test_two_round_rule_matches_serial_and_has_its_own_cache_key(self):
        first = cli.prepare(self.args)
        config = json.loads(self.config.read_text())
        config["pivot_two_rounds"] = True
        self.config.write_text(json.dumps(config))
        module = SimpleNamespace(eligible_anchors=lambda ex, block: SimpleNamespace(any=lambda: ex.anchor and ex.index % 8 == 0))
        with patch.dict(sys.modules, {"litespec.vlm.pivot_training": module, "tqdm": SimpleNamespace(tqdm=Progress)}):
            second = cli.prepare(self.args)
            saved = json.loads((second / "effective_config.json").read_text())
            serial = self.deps.train.validate_records(
                self.records, SimpleNamespace(owner=threading.get_ident()), None, saved, self.root,
            )
        self.assertNotEqual(first, second)
        self.assertEqual(self.deps.train.reuse_validation(self.records, saved, second), serial)
        self.assertEqual(serial["valid"], [0, 8, 16, 24, 32])

    def test_cli_writes_result_path(self):
        output = self.root / "result.txt"
        directory = cli.main([
            "--config", str(self.config), "--train-data", str(self.data),
            "--target-model", str(self.model), "--cache-root", self.args.cache_root,
            "--workers", "2", "--chunk-size", "3", "--result-file", str(output),
        ])
        self.assertEqual(output.read_text().strip(), str(directory))

    def test_help_does_not_import_cuda_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests/prepare_vlm_validation.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--workers", result.stdout)


class LauncherTests(unittest.TestCase):
    def test_preprocessing_runs_before_torchrun_and_passes_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("scripts/dlite/run_training_vlm.sh", "tests/prepare_vlm_validation.sh"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text((ROOT / name).read_text())
            python = root / "python"
            python.write_text("#!" + sys.executable + "\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == "-":
    sys.stdin.read()
elif args[0] == "tests/prepare_vlm_validation.py":
    with open(os.environ["CALL_LOG"], "a") as h: h.write("prepare\\n")
    if os.environ.get("FAIL_PREPARE") == "1": raise SystemExit(3)
    pathlib.Path(args[args.index("--result-file") + 1]).write_text("/checked/cache\\n")
elif args[0] == "-m":
    with open(os.environ["CALL_LOG"], "a") as h: h.write("train\\n")
    pathlib.Path(os.environ["ARGS_LOG"]).write_text(json.dumps(args))
else:
    raise SystemExit("Unexpected Python invocation: " + repr(args))
''')
            python.chmod(0o755)
            data = root / "train.jsonl"
            data.write_text('{}\n')
            config = root / "config.yaml"
            config.write_text('{}\n')
            model = root / "model"
            model.mkdir()
            env = dict(os.environ, PYTHON_BIN=str(python), TRAIN_DATA=str(data), CONFIG=str(config),
                       TARGET_MODEL=str(model), OUTPUT_DIR=str(root / "output"), PREPARE_VALIDATION="1",
                       NPROC_PER_NODE="4", CUDA_VISIBLE_DEVICES="0,1,2,3", CALL_LOG=str(root / "calls"),
                       ARGS_LOG=str(root / "args"))
            env.pop("REUSE_VALIDATION_FROM", None)
            result = subprocess.run(["bash", str(root / "scripts/dlite/run_training_vlm.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "calls").read_text(), "prepare\ntrain\n")
            args = json.loads((root / "args").read_text())
            self.assertEqual(args[args.index("--reuse-validation-from") + 1], "/checked/cache")
            (root / "calls").unlink()
            (root / "args").unlink()
            env["FAIL_PREPARE"] = "1"
            failed = subprocess.run(["bash", str(root / "scripts/dlite/run_training_vlm.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual((root / "calls").read_text(), "prepare\n")
            self.assertFalse((root / "args").exists())


if __name__ == "__main__":
    unittest.main()
