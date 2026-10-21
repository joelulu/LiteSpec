import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

from PIL import Image
import pytest

from litespec.vlm.data_pipeline import stop_owned, wait_ready
from litespec.vlm.image_budget import bounded_size, materialize_image


def test_resize_budget_is_materialized_and_source_untouched(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (1900, 1100), "red").save(image)
    original = image.read_bytes()
    record = dict(id="x", image=str(image), question="what", split="train")
    result = materialize_image(record, tmp_path / "images", factor=32, min_pixels=65536, max_pixels=262144)
    with Image.open(result["image"]) as transformed:
        w, h = transformed.size
        assert w % 32 == h % 32 == 0
        assert 65536 <= w * h <= 262144
    assert image.read_bytes() == original
    assert materialize_image(record, tmp_path / "images", factor=32, min_pixels=65536, max_pixels=262144) == result
    duplicate = tmp_path / "duplicate.jpg"
    duplicate.write_bytes(original)
    other = materialize_image({**record, "image": str(duplicate)}, tmp_path / "images", factor=32,
                              min_pixels=65536, max_pixels=262144)
    assert other["split"] == result["split"]
    assert other["image"] == result["image"]
    with pytest.raises(ValueError):
        bounded_size(1, 10000, factor=32, min_pixels=65536, max_pixels=262144)


def test_cleanup_terminates_owned_session_without_touching_caller():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    try:
        stop_owned([process], grace=.1)
        assert process.poll() is not None
        os.kill(os.getpid(), 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_wait_fails_fast_on_crashed_server():
    process = Mock(returncode=2)
    process.poll.return_value = 2
    with pytest.raises(RuntimeError, match="exited"):
        wait_ready([process], ["http://127.0.0.1:30000"], "test", 5)


def test_all_eight_servers_must_pass_health_and_model(monkeypatch):
    import urllib.request
    from litespec.vlm import data_pipeline as pipeline
    from unittest.mock import MagicMock
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.status = 200
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: opener)
    models = Mock(return_value={"data": [{"id": "expected"}]})
    monkeypatch.setattr(pipeline, "http_json", models)
    processes = [Mock() for _ in range(8)]
    for process in processes:
        process.poll.return_value = None
    wait_ready(processes, [f"http://127.0.0.1:{30000+i}" for i in range(8)], "expected", 3)
    assert opener.open.call_count == models.call_count == 8


def test_startup_failure_cleans_all_owned_processes(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from litespec.vlm import data_pipeline as pipeline
    source, target = tmp_path / "source.jsonl", tmp_path / "target"
    source.write_text('{}\n')
    target.mkdir()
    for key, value in dict(SOURCE_DATA=str(source), TARGET_MODEL=str(target),
                           PREPARE_OUTPUT=str(tmp_path / "output")).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(pipeline.subprocess, "run", Mock())
    process = Mock(pid=1234)
    popen = Mock(return_value=process)
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline.socket, "socket", MagicMock())
    monkeypatch.setattr(pipeline.signal, "signal", Mock())
    def fail(*args):
        raise TimeoutError("test startup timeout")
    monkeypatch.setattr(pipeline, "wait_ready", fail)
    cleanup = Mock()
    monkeypatch.setattr(pipeline, "stop_owned", cleanup)
    with pytest.raises(TimeoutError):
        pipeline.main()
    assert popen.call_count == 8  # generation was never started
    assert cleanup.call_args.args[0] == [process] * 8
