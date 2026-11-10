"""Stdlib checks for frozen selections, GPU isolation and combined reporting."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_dflash_all", ROOT / "tests/evaluate_dflash_all.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def row(model, samples):
    return dict(model=model, batch_size=1, requests=samples, matched=samples - 1,
                mean_committed_length=None if model == "ar" else 3.0,
                output_tokens_per_second=80.0, raw_throughput_speedup_vs_ar=2.0,
                decode_tok_s=70.0, raw_decode_ratio_vs_ar=1.75, peak_allocated_gib=9.0)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        group = self.root / "group"
        for name in ("full", "pivot1"):
            directory = group / name / "step_0000443"
            directory.mkdir(parents=True)
            (directory / "config.json").write_text(json.dumps(dict(
                architectures=["DFlashDraftModel"], vlm_config=dict(draft_context=name))))
            (directory / "model.safetensors").touch()
            (group / name / "last").symlink_to(directory.name, target_is_directory=True)
        target = self.root / "target"
        target.mkdir()
        (target / "config.json").write_text(json.dumps(dict(vision_config=dict(patch_size=16, spatial_merge_size=2))))
        (target / "model.safetensors").touch()
        self.args = SimpleNamespace(
            run_group=group, output_dir=self.root / "eval", workspace=self.root,
            target_model=target, data_dir=self.root / "data", image_root=None, text_data=None,
            video_data_root=self.root / "videos", prompt_field="reference",
            samples=32, video_samples=16, seed=42, gpus=list(map(str, range(8))),
            batch_size=1, max_new_tokens=256, warmup=1, frames=16,
            image_max_tokens=1024, max_pixels_per_frame=200704,
        )
        self.calls = []

        def images(args):
            self.assertEqual(os.environ["PROMPT_FIELD"], "reference")
            name = "gsm8k" if args.tests == ["text-only"] else args.tests[0]
            self.calls.append(name)
            records = [dict(id=f"{name}:{i}", question="q", source_row=i) for i in range(args.samples)]
            return {name: dict(source="original/" + name + ".jsonl", samples=len(records),
                               serialized="".join(json.dumps(r) + "\n" for r in records))}

        def videos(root, datasets, samples, seed):
            name = datasets[0]
            self.calls.append(name)
            return {name: [dict(id=f"{name}:{i}", video=f"/local/{i}.mp4", question="describe") for i in range(samples)]}

        self.modules = {"run_vlm_matrix": SimpleNamespace(select_data=images),
                        "video_benchmark_data": SimpleNamespace(select_data=videos)}

    def test_eight_frozen_jobs_and_counts_ignore_inherited_prompt_field(self):
        with patch.dict(os.environ, {"PROMPT_FIELD": "question"}), patch.dict(sys.modules, self.modules):
            root, jobs = cli.prepare(self.args)
        self.assertEqual(len(jobs), 8)
        self.assertEqual([j["gpu"] for j in jobs], list(map(str, range(8))))
        self.assertEqual([j["samples"] for j in jobs], [32] * 6 + [16] * 2)
        self.assertEqual(len(self.calls), 8)
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertNotIn("last", manifest["checkpoints"]["full"]["path"])
        for job in jobs:
            self.assertEqual(len(job["input_sha256"]), 64)
            command = job["command"]
            labels = [command[i + 1] for i, value in enumerate(command) if value == "--checkpoint"]
            self.assertEqual(labels, ["full", "pivot1"])
            if job["kind"] == "video":
                data = json.loads(Path(job["snapshot"]).read_text())
                self.assertEqual(set(data), {job["dataset"]})
                self.assertEqual(len(data[job["dataset"]]), 16)
                self.assertEqual(command[command.index("--frames") + 1], "16")
                self.assertNotIn("--state-mode", command)
            else:
                self.assertEqual(len(Path(job["snapshot"]).read_text().splitlines()), 32)
                self.assertEqual(command[command.index("--state-mode") + 1], "per_token")

    def test_preparation_reports_all_missing_sources_before_model_processes(self):
        def failed(args):
            raise FileNotFoundError(args.tests[0])
        modules = dict(self.modules, run_vlm_matrix=SimpleNamespace(select_data=failed))
        with patch.dict(sys.modules, modules), patch.object(cli.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Dataset preparation failed") as caught:
                cli.prepare(self.args)
        self.assertIn("gsm8k", str(caught.exception))
        self.assertIn("hr_bench_8k", str(caught.exception))
        popen.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_wrong_checkpoint_context_fails_before_selecting_data(self):
        path = self.args.run_group / "pivot1/last/config.json"
        path.write_text(json.dumps(dict(architectures=["DFlashDraftModel"], vlm_config=dict(draft_context="full"))))
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, "wrong architecture/context"):
            cli.prepare(self.args)
        self.assertEqual(self.calls, [])

    def test_help_needs_no_torch(self):
        result = subprocess.run([sys.executable, str(ROOT / "tests/evaluate_dflash_all.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--video-samples", result.stdout)

    def test_five_checkpoints_four_gpu_queues_and_32_video_samples(self):
        labels = ['pivot1', 'pivot5', 'pivot12', 'pivot5_B', 'pivot5_C']
        self.args.checkpoint = []
        self.args.gpus = ['4','5','6','7']
        self.args.video_samples = 32
        for label in labels:
            directory = self.root / 'models' / label
            directory.mkdir(parents=True)
            (directory / 'config.json').write_text(json.dumps(dict(
                architectures=['DFlashDraftModel'],
                vlm_config=dict(draft_context='pivot5' if label.endswith(('_B','_C')) else label),
                dflash_config=dict(context_projection='per_layer' if label.endswith(('_B','_C')) else 'shared',
                                   position_layer_bias=label == 'pivot5_C'))))
            (directory / 'model.safetensors').touch()
            self.args.checkpoint.append((label, directory))
        with patch.dict(sys.modules, self.modules):
            root, jobs = cli.prepare(self.args)
        self.assertEqual([j['gpu'] for j in jobs], ['4','5','6','7']*2)
        self.assertTrue(all(j['models'] == ['ar', *labels] for j in jobs))
        self.assertTrue(all(j['samples'] == 32 for j in jobs))
        for job in jobs:
            command = job['command']
            self.assertEqual([command[i+1] for i, a in enumerate(command) if a == '--checkpoint'], labels)

    def test_wrong_c_adapter_is_rejected(self):
        self.args.checkpoint = [('pivot5_C', self.args.run_group / 'pivot1/last')]
        path = self.args.run_group / 'pivot1/last/config.json'
        path.write_text(json.dumps(dict(architectures=['DFlashDraftModel'], vlm_config=dict(draft_context='pivot5'))))
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, 'wrong layer adapter'):
            cli.prepare(self.args)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "logs").mkdir()
        (self.root / "results").mkdir()

    def job(self, name, kind, gpu, fail=False, incomplete=False):
        samples = 16 if kind == "video" else 32
        output = self.root / "results" / name
        rows = [row(label, samples) for label in (["ar"] if incomplete else ["ar", "full", "pivot1"])]
        code = "import json,os,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'gpu.txt').write_text(os.environ['CUDA_VISIBLE_DEVICES']); (p/'summary.json').write_text(sys.argv[2]); sys.exit(int(sys.argv[3]))"
        return dict(dataset=name, kind=kind, gpu=str(gpu), samples=samples, output=str(output),
                    log=str(self.root / "logs" / (name + ".log")),
                    command=[sys.executable, "-c", code, str(output), json.dumps(rows), "3" if fail else "0"])

    def test_one_gpu_per_process_and_image_video_summary_mapping(self):
        jobs = [self.job("images", "image", 2), self.job("movies", "video", 7)]
        self.assertEqual(cli.execute(self.root, jobs, 1), 0)
        for job in jobs:
            self.assertEqual((Path(job["output"]) / "gpu.txt").read_text(), job["gpu"])
        rows = json.loads((self.root / "comparison.json").read_text())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["tok_s"], 80.0)
        self.assertEqual(rows[-1]["tok_s"], 70.0)
        self.assertLess(rows[0]["matched"], rows[0]["requests"])
        self.assertTrue((self.root / "comparison.csv").is_file())

    def test_failed_job_does_not_stop_other_dataset(self):
        jobs = [self.job("bad", "image", 0, fail=True), self.job("good", "video", 1)]
        self.assertEqual(cli.execute(self.root, jobs, 1), 1)
        statuses = json.loads((self.root / "status.json").read_text())
        self.assertEqual(statuses["bad"]["status"], "failed")
        self.assertEqual(statuses["good"]["status"], "success")
        self.assertEqual(len(json.loads((self.root / "comparison.json").read_text())), 3)

    def test_incomplete_summary_is_not_reported_as_success(self):
        self.assertEqual(cli.execute(self.root, [self.job("partial", "image", 0, incomplete=True)], 1), 1)
        status = json.loads((self.root / "status.json").read_text())["partial"]
        self.assertEqual(status["status"], "invalid_summary")
        self.assertEqual(json.loads((self.root / "comparison.json").read_text()), [])

    def test_same_gpu_jobs_never_overlap_and_ar_plus_five_rows_are_checked(self):
        labels = ['ar', 'pivot1', 'pivot5', 'pivot12', 'pivot5_B', 'pivot5_C']
        jobs = [self.job('dataset'+str(i), 'image', i % 2) for i in range(8)]
        # A nonblocking per-GPU OS lock fails immediately if the scheduler
        # accidentally launches two datasets on the same physical selector.
        code = ("import fcntl,json,os,pathlib,sys,time; "
                "lock=open(sys.argv[3]+'/gpu'+os.environ['CUDA_VISIBLE_DEVICES']+'.lock','w'); "
                "fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); time.sleep(.03); "
                "p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'summary.json').write_text(sys.argv[2])")
        for job in jobs:
            job['models'] = labels
            job['command'] = [sys.executable, '-c', code, job['output'],
                              json.dumps([row(label, 32) for label in labels]), str(self.root)]
        self.assertEqual(cli.execute(self.root, jobs, 1), 0)
        self.assertEqual(len(json.loads((self.root/'comparison.json').read_text())), 8*6)

    def test_interrupt_cleans_up_only_owned_child_process(self):
        pid_file = self.root / "child.pid"
        job = self.job("interrupt", "image", 0)
        job["command"] = [sys.executable, "-c", "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)", str(pid_file)]
        controller = self.root / "controller.py"
        controller.write_text(
            "import importlib.util,json,pathlib,sys\n"
            f"s=importlib.util.spec_from_file_location('cli', {str(ROOT / 'tests/evaluate_dflash_all.py')!r})\n"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            f"m.execute(pathlib.Path({str(self.root)!r}), json.loads({json.dumps([job])!r}), 1)\n"
        )
        process = subprocess.Popen([sys.executable, str(controller)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while not pid_file.is_file() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pid_file.is_file())
            child_pid = int(pid_file.read_text())
            os.kill(process.pid, signal.SIGINT)
            self.assertNotEqual(process.wait(timeout=8), 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if pid_file.is_file():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
