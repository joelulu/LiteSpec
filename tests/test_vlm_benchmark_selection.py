import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vlm_benchmark_data import read_eval_records

ROOT = Path(__file__).resolve().parents[1]


def test_turns_reader_preserves_exact_prompt_and_does_not_rewrite(tmp_path):
    prompts = [f"Question {i}: Janet’s ducks lay 16 eggs per day.\nPlease reason step by step, and put your final answer within \\boxed{{}}."
               for i in range(5)]
    path = tmp_path/"gsm8k.jsonl"
    path.write_text("".join(json.dumps(dict(turns=[prompt]), ensure_ascii=False)+"\n" for prompt in prompts))
    original = path.read_bytes()
    rows = list(read_eval_records(path))
    assert [row["question"] for row in rows] == prompts
    assert [row["id"] for row in rows] == [str(i) for i in range(5)]
    assert len(list(read_eval_records(path, 2))) == 2
    assert path.read_bytes() == original


@pytest.mark.parametrize("record", [dict(turns=[]), dict(turns=["one", "two"]), dict(turns=[""]),
                                     dict(turns=["one"], question="two")])
def test_turns_reader_rejects_ambiguous_or_multiturn_input(tmp_path, record):
    path = tmp_path/"input.jsonl"
    path.write_text(json.dumps(record)+"\n")
    with pytest.raises(ValueError):
        list(read_eval_records(path))


def shell_fixture(tmp_path):
    repo = tmp_path/"workspace/LiteSpec"
    scripts = repo/"tests"
    scripts.mkdir(parents=True)
    for name in ("compare_text_vlm_16.sh", "compare_vlm_16.sh", "compare_static_batch_text_vlm.sh",
                 "report_text_vlm_speedup.py", "vlm_benchmark_data.py"):
        shutil.copy(ROOT/"tests"/name, scripts/name)
    (repo/"litespec").symlink_to(ROOT/"litespec", target_is_directory=True)
    runner = tmp_path/"python"
    runner.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
args = sys.argv[1:]
if args and args[0] in ('tests/benchmark_vlm_checkpoints.py', 'tests/benchmark_vlm_static_batch.py'):
    with open(os.environ['CAPTURE'], 'a') as stream:
        stream.write(json.dumps(args)+'\\n')
    def option(name):
        return args[args.index(name)+1]
    root = pathlib.Path(option('--output-dir'))
    root.mkdir(parents=True, exist_ok=False)
    names = ['ar']+[args[i+1] for i, value in enumerate(args) if value == '--checkpoint']
    static = args[0].endswith('static_batch.py')
    flag = '--batch-sizes' if static else '--concurrency'
    values = []
    for value in args[args.index(flag)+1:]:
        if value.startswith('--'): break
        values.append(int(value))
    rows = []
    for name in names:
        for size in values:
            rows.append(dict(model=name, concurrency=size, batch_size=size if static else None,
                output_tokens_per_second=10, output_tokens=10, wall_seconds=1,
                mean_latency_seconds=1, mean_committed_length=None if name=='ar' else 2,
                matched=1, requests=1, valid_token_match=True,
                dataset_split=option('--data-split'), dataset_name=option('--dataset-name'),
                dataset_path=option('--data'),
                scheduler='static_replicated_batch' if static else 'round_robin_unbatched',
                benchmark_mode='static_replicated_batch' if static else 'round_robin_unbatched'))
    (root/'summary.json').write_text(json.dumps(rows))
else:
    os.execv(sys.executable, [sys.executable, *args])
''')
    runner.chmod(0o755)
    env = os.environ.copy()
    for name in ("TEXT_DATA", "TRAIN_EVAL_DATA", "VAL_EVAL_DATA", "EVAL_DATA", "BENCHMARK_MODE",
                 "ONLY_ORIGINAL", "MODELS", "TESTS", "TEXT_SPLIT", "TEXT_DATASET_NAME"):
        env.pop(name, None)
    env.update(HOME=str(tmp_path/"home"), DLITE_WORKSPACE=str(repo.parent), PYTHON_BIN=str(runner),
               EVAL_SAMPLES="1", CONCURRENCIES="1", BATCH_SIZES="1 8", RESULT_DIR=str(tmp_path/"result"),
               CAPTURE=str(tmp_path/"calls.jsonl"))
    return repo, env


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("selection,stage", [("text-only", "text"), ("visual-eval", "multimodal_val"),
                                             ("visual-training", "multimodal_train")])
def test_shell_runs_only_selected_datasets_and_models(tmp_path, static, selection, stage):
    repo, env = shell_fixture(tmp_path)
    if selection == "text-only":
        path = Path(env["HOME"])/"LMM/gaohl/project/from_aliyun/LDM/dllm_dflash/DeepSpec-main-0708-new/eval_datasets/gsm8k.jsonl"
        record = dict(turns=["Janet’s ducks lay 16 eggs per day. How much money?\nAnswer in \\boxed{}."])
        env["MODELS"] = "ar"
        expected_models = []
    else:
        path = repo.parent/"data/finevision_qwen35_sglang_20000"/("val.jsonl" if selection == "visual-eval" else "train.jsonl")
        record = dict(id="image", image="image.png", question="describe", answer="do not feed")
        env["MODELS"] = "dlite_original,dspark_3ep"
        expected_models = ["dlite_original", "dspark_3ep"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record)+"\n")
    # All unselected data and all checkpoint files deliberately do not exist.
    env["TESTS"] = selection
    script = "compare_static_batch_text_vlm.sh" if static else "compare_text_vlm_16.sh"
    result = subprocess.run(["bash", str(repo/"tests"/script)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    calls = [json.loads(line) for line in Path(env["CAPTURE"]).read_text().splitlines()]
    assert len(calls) == 1
    call = calls[0]
    assert [call[i+1] for i, value in enumerate(call) if value == "--checkpoint"] == expected_models
    assert call[call.index("--data")+1] == str(path)
    assert call[0].endswith("static_batch.py") == static
    report = (Path(env["RESULT_DIR"])/"speedup_tables.md").read_text()
    assert report.count("### ") == 1
    assert (Path(env["RESULT_DIR"])/stage/"summary.json").exists()
    if selection == "text-only":
        assert "gsm8k" in report


@pytest.mark.parametrize("static", [False, True])
def test_real_cli_accepts_gsm8k_turns_and_ar_only(monkeypatch, tmp_path, static):
    import torch
    from test_vlm import tiny_qwen35
    import benchmark_vlm_checkpoints
    import benchmark_vlm_static_batch
    cli = benchmark_vlm_static_batch if static else benchmark_vlm_checkpoints
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    prompt = "Janet’s ducks lay 16 eggs per day.\nPlease reason step by step, and put your final answer within \\boxed{}."
    class TextProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            assert kwargs["add_generation_prompt"]
            return dict(input_ids=torch.tensor([[1, 2, 3]]))
    manifest = tmp_path/"gsm8k.jsonl"
    manifest.write_text(json.dumps(dict(turns=[prompt]))+"\n")
    monkeypatch.setattr(cli, "load_target", lambda *a, **kw: (target, TextProcessor()))
    def unexpected_draft(*args, **kwargs):
        raise AssertionError("AR-only must not load a draft")
    monkeypatch.setattr(cli, "load_draft", unexpected_draft)
    argv = ["benchmark", "--target-model", "tiny", "--data", str(manifest), "--dataset-name", "gsm8k",
            "--output-dir", str(tmp_path/"results"), "--max-samples", "1", "--max-new-tokens", "3",
            "--warmup", "0", "--device", "cpu"]
    argv += ["--batch-sizes", "1", "8"] if static else ["--concurrency", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    rows = json.loads((tmp_path/"results/summary.json").read_text())
    assert len(rows) == (2 if static else 1)
    assert all(row["model"] == "ar" and row["dataset_name"] == "gsm8k" for row in rows)
