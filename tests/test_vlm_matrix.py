"""CPU queue tests: isolation, resume, GPU affinity and real AR reuse."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

import run_vlm_matrix as matrix


def test_worker_reuses_same_gpu_ar_and_isolates_oom(monkeypatch, tmp_path):
    matrix.STOP.clear()
    calls = []
    def fake_process(command, log, env):
        calls.append((command, env["CUDA_VISIBLE_DEVICES"]))
        output = Path(command[command.index("--output-dir")+1])
        output.mkdir(parents=True)
        label = command[command.index("--checkpoint")+1] if "--checkpoint" in command else "ar"
        log.write_text("CUDA out of memory" if label == "bad" else "done")
        if label == "bad":
            return "oom", 1
        (output/"summary.json").write_text(json.dumps([dict(model=label)]))
        return "success", 0
    monkeypatch.setattr(matrix, "run_process", fake_process)
    args = SimpleNamespace(output_dir=tmp_path, retry_oom=False, target_model="target", samples=2,
        max_new_tokens=4, seed=42, warmup=1, attention_backend="sdpa", image_options={}, profile_decode=False)
    models = dict(bad=dict(path="bad"), good=dict(path="good"))
    for mode in matrix.MODES:
        matrix.worker(args, "6", ["gsm8k_b16"], mode, models)
    assert len(calls) == 7  # AR once, two drafts x three modes.
    assert all(gpu == "6" for _,gpu in calls)
    assert all("--ar-reference" in command for command,_ in calls[1:])
    assert all("--cpu-inputs" in command for command,_ in calls)
    assert all(json.loads((tmp_path/f"jobs/gsm8k_b16/{m}/good/status.json").read_text())["status"] == "success" for m in matrix.MODES)
    # Completed/OOM cells are skipped; no child process on resume.
    matrix.worker(args, "6", ["gsm8k_b16"], "per_token", models)
    assert len(calls) == 7
    args.retry_oom = True
    matrix.worker(args, "6", ["gsm8k_b16"], "per_token", models)
    assert len(calls) == 8


def test_process_classifier_and_exit_status(tmp_path):
    state, code = matrix.run_process([sys.executable, "-c", "raise RuntimeError('CUDA out of memory')"], tmp_path/"oom.log", {})
    assert state == "oom" and code != 0
    state, code = matrix.run_process([sys.executable, "-c", "raise RuntimeError('bad config')"], tmp_path/"bad.log", {})
    assert state == "failed" and code != 0


def test_seven_checkpoint_resolution_requires_unambiguous_text_run(monkeypatch, tmp_path):
    runs = tmp_path/"runs"
    for i in range(2):
        path = runs/f"dflash_text_20k_3ep_{i}/last"
        path.mkdir(parents=True)
        (path/"config.json").write_text(json.dumps(dict(architectures=["DFlashDraftModel"], block_size=8)))
    monkeypatch.delenv("DFLASH_TEXT", raising=False)
    with pytest.raises(ValueError, match="Set DFLASH_TEXT"):
        matrix.checkpoints(tmp_path, ["dflash_text_20k"])
    monkeypatch.setenv("DFLASH_TEXT", str(runs/"dflash_text_20k_3ep_0/last"))
    assert matrix.checkpoints(tmp_path, ["dflash_text_20k"])["dflash_text_20k"]["block_size"] == 8


def test_real_cli_reuses_ar_and_rejects_changed_gpu(monkeypatch, tmp_path):
    import benchmark_vlm_static_batch as cli
    from test_vlm import tiny_qwen35
    from test_vlm_static_batch import make_draft, example_for
    from test_vlm_aligned import Processor
    torch.set_num_threads(1)
    torch.manual_seed(7)
    target = tiny_qwen35().requires_grad_(False)
    fixed = example_for(target, True)
    checkpoint = tmp_path/"draft"
    make_draft(target, "dflash").save_pretrained(checkpoint)
    data = tmp_path/"data.jsonl"
    data.write_text(json.dumps(dict(id="test", question="hello"))+"\n")
    monkeypatch.setattr(cli, "load_target", lambda *a,**kw:(target,Processor()))
    monkeypatch.setattr(cli, "prepare_example", lambda *a,**kw:fixed)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    common = ["benchmark", "--target-model", "tiny", "--data",str(data),"--batch-sizes","1",
              "--max-samples","1","--max-new-tokens","5","--device","cpu","--cpu-inputs"]
    monkeypatch.setattr(sys, "argv", common+["--output-dir",str(tmp_path/"ar")])
    cli.main()
    monkeypatch.setattr(sys, "argv", common+["--output-dir",str(tmp_path/"draft_result"),
        "--checkpoint","dflash",str(checkpoint),"--state-mode","per_token","--ar-reference",str(tmp_path/"ar")])
    cli.main()
    rows = json.loads((tmp_path/"draft_result/summary.json").read_text())
    original = json.loads((tmp_path/"ar/summary.json").read_text())[0]
    assert rows[0] == original  # Same measured baseline; no second AR run.
    assert len(rows) == 2 and rows[1]["valid_token_match"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(sys, "argv", common+["--output-dir",str(tmp_path/"wrong_gpu"),
        "--checkpoint","dflash",str(checkpoint),"--ar-reference",str(tmp_path/"ar")])
    with pytest.raises(ValueError, match="cuda_visible_devices"):
        cli.main()


def test_queue_main_reports_parallel_cells_and_resumes_without_rerunning(monkeypatch, tmp_path):
    matrix.STOP.clear()
    target, draft = tmp_path/"target", tmp_path/"draft"
    for path in (target, draft):
        path.mkdir()
        (path/"config.json").write_text(json.dumps(dict(architectures=["DFlashDraftModel"], block_size=8)))
    data = tmp_path/"gsm.jsonl"
    data.write_text("".join(json.dumps(dict(turns=[f"question {i}"]))+"\n" for i in range(4)))
    monkeypatch.setenv("TARGET_MODEL", str(target))
    monkeypatch.setenv("DFLASH_VLM", str(draft))
    monkeypatch.setenv("TEXT_DATA", str(data))
    monkeypatch.setattr(matrix.signal, "signal", lambda *a:None)
    monkeypatch.setattr(matrix, "inventory", lambda gpus:[dict(selector=g, uuid="GPU-"+g, name="fake", memory_mib="81920", index=g) for g in gpus])
    calls = []
    def fake(command, log, env):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("done")
        if "--output-dir" not in command:
            return "success", 0
        calls.append((command, env["CUDA_VISIBLE_DEVICES"]))
        output = Path(command[command.index("--output-dir")+1]); output.mkdir(parents=True)
        batch = int(command[command.index("--batch-sizes")+1])
        label = command[command.index("--checkpoint")+1] if "--checkpoint" in command else "ar"
        row = dict(model=label, batch_size=batch, decode_tokens_per_second=20 if label == "ar" else 40,
            mean_committed_length=None if label == "ar" else 2, peak_allocated_gib=10, peak_transaction_record_mib=48)
        (output/"summary.json").write_text(json.dumps([row]))
        records = [json.loads(l) for l in Path(command[command.index("--data")+1]).read_text().splitlines()]
        (output/f"{label}_b{batch}.jsonl").write_text("".join(json.dumps(dict(sample_id=r["id"],
            result=dict(token_ids=[1,2], accept_lengths=[] if label == "ar" else [2])))+"\n" for r in records))
        return "success",0
    monkeypatch.setattr(matrix,"run_process",fake)
    argv = ["queue", "--workspace",str(tmp_path), "--output-dir",str(tmp_path/"result"),
        "--gpus","0,1", "--models","dflash_vlm_100k", "--tests","text-only", "--samples","2",
        "--batch-sizes","1","16", "--state-modes","per_token","compact_replay"]
    monkeypatch.setattr(sys,"argv",argv)
    matrix.main()
    assert len(calls) == 6  # Two AR cells, four draft cells.
    assert {g for _,g in calls} == {"GPU-0","GPU-1"}
    rows = matrix.summarize(tmp_path/"result")
    assert all(r["status"] == "success" and r["speedup"] == 2 and r["token_exact_vs_per_token"] for r in rows)
    monkeypatch.setattr(sys,"argv",argv+["--resume"])
    matrix.main()
    assert len(calls) == 6
    assert "| ar | 1.000×" in (tmp_path/"result/matrix.md").read_text()
    monkeypatch.setattr(sys,"argv",argv+["--resume","--max-new-tokens","17"])
    with pytest.raises(SystemExit):
        matrix.main()
