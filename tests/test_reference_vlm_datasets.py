import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from reference_vlm_data import DATASET_FILES, dataset_path, get_prompt, load_reference_records, resolve_image_path
from reference_vlm_image_budget import configure_image_budget
from report_original_vlm_datasets import collect, render


def dataset_fixture(root, count=8):
    images = root / "imgs"
    images.mkdir()
    Image.new("RGB", (64, 32), "red").save(images / "sample.png")
    records = [dict(id="repeated-source-id", image="imgs/sample.png",
                    conversations=[dict(role="user", content=f"question {i}")],
                    answer="never used for prompting") for i in range(count)]
    records[0]["prompt"] = " explicit prompt\n"
    path = root / "textvqa.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return path, images


@pytest.mark.parametrize("name", list(DATASET_FILES))
def test_actual_evaluation_filenames_preferred_over_legacy(tmp_path, name):
    preferred = tmp_path / DATASET_FILES[name]
    preferred.write_text('{}\n')
    if name in ('flickr30k', 'textvqa'):
        (tmp_path / f'{name}.jsonl').write_text('{}\n')
        (tmp_path / f'{name}_train.jsonl').write_text('{}\n')
    assert dataset_path(tmp_path, name) == preferred
    if name in ('flickr30k', 'textvqa'):
        preferred.unlink()
        legacy = tmp_path / f'{name}.jsonl'
        assert dataset_path(tmp_path, name) == legacy
        legacy.unlink()
        with pytest.raises(FileNotFoundError, match='available JSONL files'):
            dataset_path(tmp_path, name)  # Never silently select the training file.


@pytest.mark.parametrize("value", ["sample.png", "imgs/sample.png", r"imgs\/sample.png"])
def test_image_resolution_handles_reference_prefix(tmp_path, value):
    assert resolve_image_path(value, tmp_path / "imgs") == tmp_path / "imgs/sample.png"
    absolute = str(tmp_path / "elsewhere.png")
    assert resolve_image_path(absolute, tmp_path / "imgs") == Path(absolute)
    assert resolve_image_path("imgs/sample.png", tmp_path) == tmp_path / "imgs/sample.png"


def test_reference_prompt_priority_and_explicit_override():
    row = dict(prompt=" keep exact whitespace\n", question="alternative",
               conversations=[dict(content="first"), dict(content="later")])
    assert get_prompt(row) == (" keep exact whitespace\n", "prompt")
    assert get_prompt(row, "first-content") == ("first", "first-content")
    assert get_prompt(row, "question") == ("alternative", "question")
    assert get_prompt(dict(row, prompt="")) == ("first", "first-content")
    with pytest.raises(ValueError, match="missing string"):
        get_prompt(dict(question="not implicitly selected"))
    with pytest.raises(ValueError, match="missing string"):
        get_prompt(dict(prompt=["invalid"]))


def test_seeded_pairs_inspection_and_no_source_mutation(tmp_path):
    path, images = dataset_fixture(tmp_path)
    original = path.read_bytes()
    vision, info = load_reference_records(path, 3, 42, images)
    text, _ = load_reference_records(path, 3, 42, images, False)
    assert [r["id"] for r in vision] == [r["id"] for r in text]
    assert [r["question"] for r in vision] == [r["question"] for r in text]
    assert len({r["id"] for r in vision}) == 3
    assert all(r["source_id"] == "repeated-source-id" for r in vision)
    assert [r["source_row"] for r in info["first_five"]] == list(range(5))
    assert info["first_five"][0]["prompt_preview"] == " explicit prompt\n"
    assert all(r["original_image_size"] == [64, 32] for r in vision)
    assert all("image" not in r and "answer" not in r for r in text)
    assert path.read_bytes() == original
    assert len(load_reference_records(path, 0)[0]) == 8
    assert len(load_reference_records(path, 100)[0]) == 8
    (images / "sample.png").unlink()
    assert len(load_reference_records(path, 3, require_image=False)[0]) == 3
    with pytest.raises(FileNotFoundError, match="image not found"):
        load_reference_records(path)


def test_inspection_cli_loads_no_models(tmp_path):
    path, images = dataset_fixture(tmp_path)
    out = tmp_path / "inspection.json"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("reference_vlm_data.py")),
        "--data-dir", str(path.parent), "--image-root", str(images), "--datasets", "textvqa",
        "--samples", "3", "--modes", "text", "vision", "--output", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())[0]["selected_records"] == 3


@pytest.mark.parametrize("models", ['ar dlite_original', 'ar dflash_3ep dspark_3ep dlite_3ep dlite_6ep'])
@pytest.mark.parametrize("inspect_only", [False, True])
def test_wrapper_preflight_and_argument_forwarding(tmp_path, inspect_only, models):
    from test_vlm_benchmark_selection import shell_fixture
    repo, env = shell_fixture(tmp_path)
    for name in ("compare_original_vlm_datasets.sh", "reference_vlm_data.py", "report_original_vlm_datasets.py"):
        shutil.copy(Path(__file__).with_name(name), repo / "tests" / name)
    # Extend the fake benchmark's summary with the new report fields; it still
    # records actual shell arguments while real Python runs the data preflight.
    runner = Path(env["PYTHON_BIN"])
    runner.write_text(runner.read_text().replace(
        "output_tokens_per_second=10, output_tokens=10, wall_seconds=1,",
        "output_tokens_per_second=10, output_tokens=10, wall_seconds=1, "
        "unique_prompts=1, mean_input_tokens=50, mean_image_tokens=4, "
        "decode_tokens_per_second=10, end_to_end_tokens_per_second=5, "
        "prefill_seconds=1, decode_seconds=1, peak_allocated_gib=10,"))
    data = tmp_path / "input data"
    data.mkdir()
    path, images = dataset_fixture(data)
    env.update(DATA_DIR=str(data), DATASETS="textvqa", MODES="text vision",
               SAMPLES_PER_DATASET="3", MAX_NEW_TOKENS="512", IMAGE_MAX_PIXELS="2048",
               INSPECT_ONLY="1" if inspect_only else "0", MODELS=models)
    env.pop("IMAGE_ROOT", None)
    run = subprocess.run(["bash", str(repo / "tests/compare_original_vlm_datasets.sh")],
                         env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (Path(env["RESULT_DIR"]) / "inspection.json").is_file()
    if inspect_only:
        assert not Path(env["CAPTURE"]).exists()
        return
    calls = [json.loads(line) for line in Path(env["CAPTURE"]).read_text().splitlines()]
    assert len(calls) == 2
    for mode, call in zip(("text", "vision"), calls):
        option = lambda key: call[call.index(key)+1]
        assert option("--input-mode") == mode and "--reference-input" in call
        assert option("--image-root") == str(images)
        assert option("--data") == str(path)
        assert option("--max-samples") == "3" and option("--max-new-tokens") == "512"
        assert option("--image-max-pixels") == "2048"
        assert [call[i+1] for i, value in enumerate(call) if value == "--checkpoint"] == models.split()[1:]
    report = (Path(env["RESULT_DIR"]) / "comparison.md").read_text()
    assert "同题去图" in report and "图文输入" in report


def test_pixel_override_changes_active_processor_size():
    image = SimpleNamespace(size=dict(shortest_edge=1024, longest_edge=1048576))
    processor = SimpleNamespace(image_processor=image)
    assert configure_image_budget(processor, max_pixels=262144) == dict(shortest_edge=1024, longest_edge=262144)
    assert image.size["longest_edge"] == 262144
    with pytest.raises(ValueError, match="positive"):
        configure_image_budget(processor, min_pixels=262145)
    assert image.size["shortest_edge"] == 1024, "failed overrides must be atomic"
    with pytest.raises(ValueError, match="unsupported"):
        configure_image_budget(SimpleNamespace(image_processor=SimpleNamespace(size={"height": 224})))


def test_pixel_override_changes_real_qwen_processed_grid():
    from transformers import Qwen2VLImageProcessor
    image = Qwen2VLImageProcessor()
    picture = Image.new("RGB", (1024, 1024))
    before = image(images=[picture], return_tensors="pt")["image_grid_thw"][0].tolist()
    configure_image_budget(SimpleNamespace(image_processor=image), max_pixels=262144)
    after = image(images=[picture], return_tensors="pt")["image_grid_thw"][0].tolist()
    assert after[1] * after[2] < before[1] * before[2]
    assert after[1] * after[2] * image.patch_size**2 <= 262144


def test_report_separates_decode_and_end_to_end_ratios(tmp_path):
    for mode in ("text", "vision"):
        folder = tmp_path / "textvqa" / mode
        folder.mkdir(parents=True)
        ar = dict(model="ar", batch_size=8, unique_prompts=2, requests=16,
            output_tokens=160, mean_input_tokens=500, mean_image_tokens=400 if mode == "vision" else 0,
            decode_tokens_per_second=10, end_to_end_tokens_per_second=5,
            prefill_seconds=16, decode_seconds=16, matched=16, valid_token_match=True,
            mean_committed_length=None, peak_allocated_gib=10)
        draft = dict(ar, model="dlite_original", decode_tokens_per_second=20,
            end_to_end_tokens_per_second=6, mean_committed_length=4, matched=8, valid_token_match=False)
        (folder / "summary.json").write_text(json.dumps([ar, draft]))
    rows = collect(tmp_path, ["textvqa"], ["text", "vision"])
    assert all(r["decode_speedup"] == 2 and r["end_to_end_speedup"] == 1.2 for r in rows)
    assert all(r["unique_prompts"] == 2 and r["sequences"] == 16 for r in rows)
    report = render(rows, ["text", "vision"])
    assert "同题去图" in report and "图文输入" in report
    assert report.count("2.000×") == 2


@pytest.mark.parametrize("mode", ["text", "vision"])
def test_reference_input_real_hybrid_cli(monkeypatch, tmp_path, mode):
    import torch
    import benchmark_vlm_static_batch as cli
    from test_vlm import tiny_qwen35, tiny_image_inputs
    from test_vlm_static_batch import make_draft

    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    checkpoint = tmp_path / "draft"
    make_draft(target, "dlite_v1").save_pretrained(checkpoint)
    source = tmp_path / "data"
    source.mkdir()
    path, images = dataset_fixture(source)
    seen = []

    class Processor:
        image_processor = SimpleNamespace(size=dict(shortest_edge=4, longest_edge=4096))
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "decoded answer")

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1 and messages[0]["role"] == "user"
            content = messages[0]["content"]
            assert content[-1]["type"] == "text" and "never used" not in content[-1]["text"]
            assert kwargs["enable_thinking"] is False
            seen.append(content[-1]["text"])
            if mode == "vision":
                assert content[0]["image"].size == (64, 32)
                return tiny_image_inputs()
            assert len(content) == 1
            return dict(input_ids=torch.tensor([[1, 2, 3]]))

    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, Processor()))
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--target-model", "tiny",
        "--checkpoint", "dlite_original", str(checkpoint), "--data", str(path),
        "--dataset-name", "textvqa", "--reference-input", "--input-mode", mode,
        "--image-root", str(images), "--image-max-pixels", "2048", "--seed", "42",
        "--output-dir", str(output), "--batch-sizes", "1", "2", "--max-samples", "2",
        "--max-new-tokens", "3", "--warmup", "0", "--device", "cpu"])
    cli.main()
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 4 and all(r["valid_token_match"] for r in rows)
    assert all(r["input_mode"] == mode and r["dataset_name"] == "textvqa" for r in rows)
    settings = json.loads((output / "settings.json").read_text())
    assert settings["effective_image_processor_size"]["longest_edge"] == 2048
    expected, _ = load_reference_records(path, 2, 42, images, mode == "vision")
    assert seen == [r["question"] for r in expected]
    metadata = json.loads((output / "inputs.json").read_text())
    if mode == "vision":
        assert metadata[0]["original_image_size"] == [64, 32]
        assert metadata[0]["image_grid_thw"] is not None and metadata[0]["image_tokens"] > 0
        assert metadata[0]["processed_image_hw"] is not None
    else:
        assert metadata[0]["image_tokens"] == 0 and metadata[0]["image_path"] is None
    predictions = [json.loads(s) for s in (output / "dlite_original_b2.jsonl").read_text().splitlines()]
    assert all(r["decoded_text"] == "decoded answer" for r in predictions)
