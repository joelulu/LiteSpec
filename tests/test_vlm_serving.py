"""CPU HTTP contract tests. No SGLang, CUDA, model or dataset downloads."""

import argparse
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.vlm.serving_data import (
    LengthValidator, chat_payload, first_image_record, normalize_server,
    output_lock, prepare_manifest, read_jsonl, regenerate, request_answer,
)


class Validator:
    def prompt(self, record):
        if record["question"] == "oversized":
            raise ValueError("input too long")
        return [1, 2, 3]

    def answer(self, record, prefix):
        assert prefix == [1, 2, 3]
        assert record["answer"] == "A picture."


@contextmanager
def server(*, fail_once=None, model="qwen35-vlm"):
    state = {"requests": [], "active": 0, "peak": 0, "failed": False}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/v1/models"
            self.send_json(200, {"data": [{"id": "qwen35-vlm"}]})

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["requests"].append(body)
                if fail_once and not state["failed"]:
                    state["failed"] = True
                    self.send_json(fail_once, {"error": "test failure"})
                    return
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.03)
            try:
                self.send_json(200, {"model": model, "choices": [{
                    "message": {"content": "A picture."}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3}})
            finally:
                with lock:
                    state["active"] -= 1

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"test image bytes")

    def tearDown(self):
        self.temp.cleanup()

    def raw(self, question="What?", images=None):
        return {"id": "original", "source": "FineVision", "images": images or [str(self.image)],
                "conversations": [{"role": "user", "content": "<image>\n" + question},
                                  {"role": "assistant", "content": "old answer"},
                                  {"role": "user", "content": "ignored later turn"}]}

    def manifest(self, count=6):
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(self.raw(f"question {i}")) + "\n" for i in range(count)))
        prepare_manifest(path, self.root / "manifest", 0)
        return self.root / "manifest/prompts.jsonl"

    def args(self, manifest, addresses):
        return argparse.Namespace(
            manifest=str(manifest), output_dir=str(self.root / "answers"),
            target_model="local-Qwen3.5-4B", model="qwen35-vlm",
            server_address=addresses, concurrency=2, max_tokens=32,
            max_length=4096, timeout=5, retries=0, api_key_env="TEST_UNUSED_KEY")

    def test_adapter_first_turn_stable_ids_and_image_split(self):
        a = first_image_record(self.raw(), self.root)
        b = first_image_record(self.raw("Another?"), self.root)
        self.assertEqual(a["question"], "What?")
        self.assertEqual(a["id"], hashlib.sha256((str(self.image) + "\nWhat?").encode()).hexdigest())
        self.assertEqual(a["split"], b["split"])
        self.assertNotIn("answer", a)
        with self.assertRaisesRegex(ValueError, "not_single_image"):
            first_image_record(self.raw(images=[str(self.image)] * 2), self.root)

    def test_manifest_first_five_inspection_dedup_resume_identity(self):
        source = self.root / "source.jsonl"
        rows = [self.raw(), self.raw()] + [self.raw(str(i)) for i in range(4)]
        source.write_text("".join(json.dumps(r) + "\n" for r in rows))
        directory = self.root / "manifest"
        result = prepare_manifest(source, directory, 0)
        self.assertEqual(result["counts"]["duplicate"], 1)
        self.assertEqual(len(list(read_jsonl(directory / "prompts.jsonl"))), 5)
        self.assertEqual(len(json.loads((directory / "inspect_first5.json").read_text())), 5)
        self.assertEqual(result, prepare_manifest(source, directory, 0))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            prepare_manifest(source, directory, 3)

    def test_manifest_limit_counts_candidates_before_dedup(self):
        source = self.root / "source.jsonl"
        source.write_text("".join(json.dumps(self.raw(str(i))) + "\n" for i in range(6)))
        result = prepare_manifest(source, self.root / "manifest", 2)
        self.assertEqual(sum(result["counts"].get(x, 0) for x in ("train", "val")), 2)

    def test_image_payload_contains_bytes_not_placeholder(self):
        record = first_image_record(self.raw(), self.root)
        payload = chat_payload(record, "qwen35-vlm", 32)
        content = payload["messages"][0]["content"]
        self.assertEqual(base64.b64decode(content[0]["image_url"]["url"].split(",")[1]), self.image.read_bytes())
        self.assertEqual(content[1], {"type": "text", "text": "What?"})
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])

    def test_multiserver_concurrency_and_exact_resume(self):
        manifest = self.manifest(12)
        with server() as (one, state1), server() as (two, state2):
            args = self.args(manifest, [one, two])
            summary = regenerate(args, Validator())
            self.assertEqual(summary["succeeded"], 12)
            self.assertEqual(summary["completion_tokens"], 36)
            self.assertTrue(1 < state1["peak"] <= 2)
            self.assertTrue(1 < state2["peak"] <= 2)
            again = regenerate(args, Validator())
            self.assertEqual(again["submitted"], 0)
            self.assertEqual(len(state1["requests"]) + len(state2["requests"]), 12)
        records = [row for p in (self.root / "answers").glob("*.jsonl")
                   if p.name in {"train.jsonl", "val.jsonl"} for _, row in read_jsonl(p)]
        self.assertEqual(len({r["id"] for r in records}), 12)
        self.assertTrue(all(r["prefill_tokens"] == 3 for r in records))

    def test_failed_records_are_retried_next_run(self):
        manifest = self.manifest(1)
        with server(fail_once=400) as (address, state):
            args = self.args(manifest, [address])
            first = regenerate(args, Validator())
            self.assertEqual(first["failed"], 1)
            second = regenerate(args, Validator())
            self.assertEqual(second["succeeded"], 1)
            self.assertEqual(len(state["requests"]), 2)

    def test_transient_http_failure_retries(self):
        record = first_image_record(self.raw(), self.root)
        with server(fail_once=503) as (address, state), patch("litespec.vlm.serving_data.time.sleep"):
            result = request_answer(record, address, model="qwen35-vlm", max_tokens=32,
                                    timeout=5, retries=1)
            self.assertEqual(result["generation"]["attempts"], 2)
            self.assertEqual(len(state["requests"]), 2)

    def test_changed_generation_settings_rejected(self):
        manifest = self.manifest(1)
        with server() as (address, _):
            args = self.args(manifest, [address])
            regenerate(args, Validator())
            args.max_tokens += 1
            with self.assertRaisesRegex(ValueError, "identity changed"):
                regenerate(args, Validator())

    def test_skip_length_before_request_and_resume(self):
        path = self.root / "source.jsonl"
        path.write_text(json.dumps(self.raw("oversized")) + "\n")
        prepare_manifest(path, self.root / "manifest", 0)
        with server() as (address, state):
            args = self.args(self.root / "manifest/prompts.jsonl", [address])
            result = regenerate(args, Validator())
            self.assertEqual(result["skipped"], 1)
            self.assertEqual(state["requests"], [])
            self.assertEqual(regenerate(args, Validator())["submitted"], 0)

    def test_response_model_mismatch_fails(self):
        with server(model="different-checkpoint") as (address, _):
            result = regenerate(self.args(self.manifest(1), [address]), Validator())
            self.assertEqual(result["failed"], 1)

    def test_lock_and_url_validation(self):
        with output_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "another writer"):
                with output_lock(self.root):
                    pass
        self.assertEqual(normalize_server("localhost:30000"), "http://localhost:30000/v1")
        self.assertEqual(normalize_server("http://localhost:30000/v1/"), "http://localhost:30000/v1")
        with self.assertRaises(ValueError):
            normalize_server("http://name:secret@localhost:30000")

    def test_validator_prefix_and_short_answer(self):
        validator = object.__new__(LengthValidator)
        validator.max_length, validator.max_new_tokens = 100, 10
        validator.encode = lambda record, answer=None: [1, 2, 3] if answer is None else [1, 2, 3, 4, 5]
        prefix = validator.prompt({})
        validator.answer({"answer": "good"}, prefix)
        validator.encode = lambda *args, **kwargs: [9, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "prefix-stable"):
            validator.answer({"answer": "bad"}, prefix)
        validator.encode = lambda *args, **kwargs: [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "too short"):
            validator.answer({"answer": "short"}, prefix)


if __name__ == "__main__":
    unittest.main()
