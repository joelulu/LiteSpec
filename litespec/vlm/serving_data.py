"""Single-image prompt preparation and bounded, resumable serving generation.

No GPU model is loaded here. Transformers/Pillow are needed only by the CPU
length validator; the HTTP client and manifest adapter use the standard library.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, build_opener, ProxyHandler


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{number}: {exc}") from exc


def append_json(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def output_lock(directory):
    """A single client owns the output; GPUs are accessed through servers."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another writer owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def first_image_record(row, source_dir):
    """Confirmed FineVision schema: conversations[0].content, first turn only."""
    images, turns = row.get("images", []), row.get("conversations", [])
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("not_single_image")
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], dict):
        raise ValueError("missing_first_turn")
    if turns[0].get("role") != "user":
        raise ValueError("first_turn_not_user")
    question = turns[0].get("content")
    if not isinstance(question, str) or question.count("<image>") > 1:
        raise ValueError("unsupported_question")
    question = question.replace("<image>", "").strip()
    if not question or not isinstance(images[0], str):
        raise ValueError("empty_question_or_invalid_image_path")
    image = Path(images[0]).expanduser()
    image = (image if image.is_absolute() else source_dir / image).resolve()
    if not image.is_file():
        raise ValueError("missing_image")
    key = hashlib.sha256((str(image) + "\n" + question).encode()).hexdigest()
    bucket = int(hashlib.sha256(str(image).encode()).hexdigest(), 16) % 100
    return {"id": key, "source_id": row.get("sample_id", row.get("id")),
            "data_source": row.get("source"), "image": str(image),
            "question": question, "split": "val" if bucket < 5 else "train"}


def prepare_manifest(source, output_dir, max_samples=10000):
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative (0 = all)")
    source, directory = Path(source).expanduser().resolve(), Path(output_dir).resolve()
    # Inspect five records before applying the confirmed mapping. Do not guess
    # alternative conversation fields or convert multimodal text placeholders.
    preview = []
    for number, row in read_jsonl(source):
        turns = row.get("conversations", [])
        preview.append({"line": number, "keys": list(row),
                        "image_count": len(row.get("images", [])),
                        "roles": [x.get("role") for x in turns if isinstance(x, dict)]})
        if len(preview) == 5:
            break
    if not preview:
        raise ValueError("empty source")
    settings = {"source": str(source), "source_sha256": file_digest(source),
                "max_samples": max_samples, "prompt_path": "conversations[0].content",
                "turn_mode": "first", "split": "image_path_sha256_mod100_val5",
                "schema_version": 2}
    with output_lock(directory):
        meta, manifest = directory / "manifest_config.json", directory / "prompts.jsonl"
        if meta.exists():
            old = json.loads(meta.read_text())
            if old["settings"] != settings or file_digest(manifest) != old["manifest_sha256"]:
                raise ValueError("manifest input/settings changed; choose a new manifest directory")
            return old
        if manifest.exists():
            raise ValueError("untracked prompts.jsonl exists; choose a new manifest directory")
        (directory / "inspect_first5.json").write_text(json.dumps(preview, indent=2))
        counts, seen, eligible = Counter(), set(), 0
        temporary = directory / "prompts.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as handle, (
            directory / "skipped.jsonl"
        ).open("w", encoding="utf-8") as skipped:
            for number, row in read_jsonl(source):
                if max_samples and eligible >= max_samples:
                    break
                counts["scanned"] += 1
                try:
                    record = first_image_record(row, source.parent)
                except ValueError as exc:
                    counts[str(exc)] += 1
                    append_json(skipped, {"line": number, "reason": str(exc)})
                    continue
                if record["id"] in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(record["id"])
                eligible += 1
                append_json(handle, record)
                counts[record["split"]] += 1
        if not seen:
            raise ValueError("no eligible single-image prompts")
        if file_digest(source) != settings["source_sha256"]:
            raise ValueError("source changed during conversion")
        os.replace(temporary, manifest)
        result = {"settings": settings, "counts": dict(counts),
                  "manifest_sha256": file_digest(manifest)}
        meta.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result


def normalize_server(address):
    address = address.rstrip("/")
    if "://" not in address:
        address = "http://" + address
    parts = urlsplit(address)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.query or parts.fragment:
        raise ValueError("server must be an HTTP(S) URL without credentials/query/fragment")
    if parts.path not in {"", "/v1"}:
        raise ValueError("server URL path must be empty or /v1")
    return address if parts.path == "/v1" else address + "/v1"


def http_json(url, *, body=None, timeout=600, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers)
    opener = build_opener(ProxyHandler({})) if urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"} else None
    with (opener.open(request, timeout=timeout) if opener else urlopen(request, timeout=timeout)) as response:
        return json.load(response)


def chat_payload(record, model, max_tokens):
    path = Path(record["image"])
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError(f"unsupported image format: {path.suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        {"type": "text", "text": record["question"]},
    ]}], "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}


class ShortAnswer(ValueError):
    pass


class LengthValidator:
    """CPU-only, native HF processor matching the training input contract."""

    def __init__(self, target_model, max_length, max_new_tokens, min_answer_tokens=2):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(target_model)
        self.max_length, self.max_new_tokens = max_length, max_new_tokens
        self.min_answer_tokens = min_answer_tokens

    def encode(self, record, *, answer=None):
        from PIL import Image

        with Image.open(record["image"]) as image:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": record["question"]}]}]
            if answer is not None:
                messages.append({"role": "assistant", "content": answer})
            encoded = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=answer is None,
                enable_thinking=False, return_dict=True, return_tensors="pt")
        return encoded["input_ids"][0].tolist()

    def prompt(self, record):
        prefix = self.encode(record)
        if len(prefix) + self.max_new_tokens + 32 > self.max_length:
            raise ValueError("input too long to reserve answer budget")
        return prefix

    def answer(self, record, prefix):
        tokens = self.encode(record, answer=record["answer"])
        if len(tokens) > self.max_length:
            raise ValueError("training sequence exceeds max_length")
        prefix_length = len(prefix)
        if tokens[:prefix_length] != prefix:
            raise ValueError("chat template is not prefix-stable")
        if len(tokens) < prefix_length + getattr(self, "min_answer_tokens", 2):
            raise ShortAnswer("answer too short for the requested supervision budget")


def request_answer(record, server, *, model, max_tokens, timeout, retries, api_key=None):
    started = time.monotonic()
    payload = chat_payload(record, model, max_tokens)
    for attempt in range(retries + 1):
        try:
            response = http_json(server + "/chat/completions", body=payload,
                                 timeout=timeout, api_key=api_key)
            choice = response["choices"][0]
            text = choice["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty assistant response")
            if choice.get("message", {}).get("reasoning_content"):
                raise ValueError("server returned reasoning although thinking was disabled")
            if response.get("model") != model:
                raise ValueError(f"response model differs from requested model: {response.get('model')!r}")
            return {"answer": text, "generation": {
                "backend": "sglang", "model": response["model"],
                "server": server, "finish_reason": choice.get("finish_reason"),
                "completion_tokens": response.get("usage", {}).get("completion_tokens"),
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens"),
                "request_seconds": time.monotonic() - started, "attempts": attempt + 1}}
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            transient = not isinstance(exc, HTTPError) or exc.code in {408, 429, 500, 502, 503, 504}
            if not transient or attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def regenerate(args, validator=None):
    from tqdm import tqdm

    if args.concurrency < 1 or args.retries < 0 or args.timeout <= 0:
        raise ValueError("invalid concurrency, retries or timeout")
    if args.max_tokens < 1 or args.max_length <= args.max_tokens:
        raise ValueError("invalid token/length budget")
    min_answer_tokens = getattr(args, "min_answer_tokens", 2)
    if not 1 <= min_answer_tokens <= args.max_tokens:
        raise ValueError("min_answer_tokens must lie between 1 and max_tokens")
    servers = list(dict.fromkeys(normalize_server(x) for x in args.server_address))
    manifest, directory = Path(args.manifest).resolve(), Path(args.output_dir).resolve()
    records = [row for _, row in read_jsonl(manifest)]
    ids = set()
    for row in records:
        if row["id"] in ids or row.get("split") not in {"train", "val"}:
            raise ValueError("duplicate manifest id or invalid split")
        ids.add(row["id"])
    if not records:
        raise ValueError("empty manifest")
    api_key = os.environ.get(args.api_key_env)
    with output_lock(directory):
        settings = {"manifest_sha256": file_digest(manifest), "target_model": args.target_model,
                    "backend": "sglang", "model": args.model, "temperature": 0.0,
                    "enable_thinking": False, "max_tokens": args.max_tokens,
                    "max_length": args.max_length, "schema_version": 1}
        if min_answer_tokens != 2:
            settings["min_answer_tokens"] = min_answer_tokens
        image_budget = None
        if getattr(args, "image_max_tokens", 0):
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(args.target_model).image_processor
            factor = int(processor.patch_size) * int(processor.merge_size)
            native_min = getattr(processor, "min_pixels", processor.size["shortest_edge"])
            native_max = getattr(processor, "max_pixels", processor.size["longest_edge"])
            image_budget = dict(factor=factor, min_pixels=int(native_min),
                                max_pixels=int(args.image_max_tokens) * factor * factor)
            if not native_min <= image_budget["max_pixels"] <= native_max:
                raise ValueError("image token cap must lie within the native processor pixel range")
            settings.update(schema_version=2, image_budget=image_budget)
            settings["split"] = "image_content_sha256_mod100_val5"
        meta = directory / "generation_config.json"
        if meta.exists():
            if json.loads(meta.read_text()) != settings:
                raise ValueError("generation identity changed; use a new output directory")
        else:
            if any((directory / name).exists() for name in ("train.jsonl", "val.jsonl", "skipped.jsonl")):
                raise ValueError("untracked output files exist; use a new output directory")
            meta.write_text(json.dumps(settings, indent=2))
        done = set()
        for name in ("train.jsonl", "val.jsonl", "skipped.jsonl"):
            if (directory / name).exists():
                for _, row in read_jsonl(directory / name):
                    if row["id"] not in ids or row["id"] in done:
                        raise ValueError("unknown/duplicate id in saved outputs")
                    done.add(row["id"])
        todo = [r for r in records if r["id"] not in done]
        if not todo:
            return {"already_done": len(done), "submitted": 0, "failed": 0}
        for server in servers:
            available = http_json(server + "/models", timeout=min(args.timeout, 30), api_key=api_key)
            if args.model not in {x["id"] for x in available.get("data", [])}:
                raise ValueError(f"model {args.model!r} is not served at {server}")
        validator = validator or LengthValidator(args.target_model, args.max_length, args.max_tokens, min_answer_tokens)
        counts = Counter(already_done=len(done))
        started = time.monotonic()
        workers = args.concurrency * len(servers)
        pending = {}
        with (directory / "train.jsonl").open("a", encoding="utf-8") as train, (
            directory / "val.jsonl"
        ).open("a", encoding="utf-8") as val, (
            directory / "errors.jsonl"
        ).open("a", encoding="utf-8") as errors, (
            directory / "skipped.jsonl"
        ).open("a", encoding="utf-8") as skipped, ExitStack() as pools, tqdm(
            total=len(records), initial=len(done), desc="SGLang generation", unit="sample", dynamic_ncols=True
        ) as progress:
            executors = [pools.enter_context(ThreadPoolExecutor(max_workers=args.concurrency)) for _ in servers]
            def finish(futures):
                for future in futures:
                    record, prefix = pending.pop(future)
                    try:
                        result = {**record, **future.result(), "answer_target": args.target_model,
                                  "prefill_tokens": len(prefix)}
                        reported = result["generation"].get("prompt_tokens")
                        if reported is not None and reported != len(prefix):
                            raise ValueError(f"server/CPU prompt token mismatch: {reported} != {len(prefix)}")
                        validator.answer(result, prefix)
                    except ShortAnswer as exc:
                        counts["skipped"] += 1
                        append_json(skipped, {"id": record["id"], "reason": str(exc)})
                    except Exception as exc:
                        counts["failed"] += 1
                        append_json(errors, {"id": record["id"], "error_type": type(exc).__name__,
                                             "error": str(exc), "time": time.time()})
                    else:
                        append_json(val if record["split"] == "val" else train, result)
                        counts["succeeded"] += 1
                        counts["completion_tokens"] += result["generation"]["completion_tokens"] or 0
                    progress.update(1)
                    progress.set_postfix(ok=counts["succeeded"], fail=counts["failed"],
                                         skip=counts["skipped"], refresh=False)

            for record in todo:
                while len(pending) >= workers:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    finish(finished)
                # Drain completed requests before CPU validation of the next input.
                finish({f for f in pending if f.done()})
                try:
                    if image_budget:
                        from .image_budget import materialize_image
                        record = materialize_image(record, directory / "images", **image_budget)
                    prefix = validator.prompt(record)
                except (ValueError, OSError) as exc:
                    append_json(skipped, {"id": record["id"], "reason": str(exc)})
                    counts["skipped"] += 1
                    progress.update(1)
                    continue
                server_index = counts["submitted"] % len(servers)
                server = servers[server_index]
                future = executors[server_index].submit(request_answer, record, server, model=args.model,
                                         max_tokens=args.max_tokens, timeout=args.timeout,
                                         retries=args.retries, api_key=api_key)
                pending[future] = record, prefix
                counts["submitted"] += 1
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                finish(finished)
        seconds = time.monotonic() - started
        summary = {**dict(counts), "seconds": seconds,
                   "samples_per_second": counts["succeeded"] / seconds,
                   "completion_tokens_per_second": counts["completion_tokens"] / seconds}
        (directory / "last_summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def manifest_main():
    parser = argparse.ArgumentParser(description="Prepare FineVision single-image first-turn prompts on CPU")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()
    print("Prompt path: conversations[0].content; turn mode: first; single image only", flush=True)
    print(json.dumps(prepare_manifest(args.input, args.output_dir, args.max_samples), indent=2))


def serving_main():
    parser = argparse.ArgumentParser(description="Regenerate image answers through SGLang/OpenAI vision API")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", required=True, help="local HF processor/model identity, no GPU weights loaded")
    parser.add_argument("--model", required=True, help="server's --served-model-name")
    parser.add_argument("--server-address", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=8, help="maximum workers per server")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--image-max-tokens", type=int, default=0,
                        help="cap merged visual tokens via persistent resized PNGs; 0 keeps native images")
    parser.add_argument("--min-answer-tokens", type=int, default=2,
                        help="skip answers with fewer supervised tokens (includes chat end marker)")
    args = parser.parse_args()
    summary = regenerate(args)
    print(json.dumps(summary, indent=2))
    if summary.get("failed", 0):
        raise SystemExit(1)
