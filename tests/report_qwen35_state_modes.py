"""Three recovery modes: normal throughput, separate cost profiles, parity."""
import argparse
import csv
import json
from pathlib import Path

MODES = ("full_target_replay", "per_token", "compact_replay")


def report(root):
    root = Path(root)
    rows, cases = [], {}
    settings_by_case = {}
    for mode in MODES:
        if not (root/mode).is_dir():
            raise ValueError(f"missing state-mode results: {mode}")
        for path in sorted((root/mode).rglob("summary.json")):
            case = str(path.parent.relative_to(root/mode))
            settings = json.loads((path.parent/"settings.json").read_text())
            identity = {k:settings.get(k) for k in ("target_model", "data", "sample_ids", "resolved_checkpoints",
                "max_new_tokens", "batch_sizes", "actual_attention_backend", "effective_image_processor_size", "input_mode")}
            if case in settings_by_case and settings_by_case[case] != identity:
                raise ValueError(f"different model/data/generation settings across state modes: {case}")
            settings_by_case[case] = identity
            profiles = json.loads((path.parent/"decode_profile.json").read_text()) if (path.parent/"decode_profile.json").exists() else []
            for summary in json.loads(path.read_text()):
                model, batch = summary["model"], summary["batch_size"]
                records = [json.loads(line) for line in (path.parent/f"{model}_b{batch}.jsonl").read_text().splitlines() if line.strip()]
                # Compare by ID, never silently zip different sample lists.
                tokens = {r["sample_id"]:r["result"]["token_ids"] for r in records}
                lengths = {r["sample_id"]:r["result"]["accept_lengths"] for r in records}
                cases[(case, model, batch, mode)] = (tokens, lengths, summary)
                profile = next((p for p in profiles if p["model"] == model and p["batch_size"] == batch), {})
                stages = profile.get("stage_seconds", {})
                row = dict(case=case, model=model, batch_size=batch, state_mode=mode,
                    gdn_verify_backend=summary["gdn_verify_backend"], samples=summary["unique_prompts"],
                    decode_tokens_per_second=summary["decode_tokens_per_second"],
                    speedup_vs_ar=summary["raw_throughput_speedup_vs_ar"],
                    mean_committed_length=summary["mean_committed_length"],
                    peak_allocated_gib=summary["peak_allocated_gib"],
                    peak_transaction_record_mib=summary["peak_transaction_record_mib"],
                    target_calls_per_round=profile.get("target_calls_per_round"),
                    ms_per_round=profile.get("ms_per_round"),
                    verify_ms=1000*stages.get("target_verify", 0),
                    full_target_replay_ms=1000*stages.get("target_replay", 0),
                    state_commit_ms=1000*(stages.get("state_select", 0)+stages.get("gdn_state_replay", 0)),
                    cache_commit_ms=1000*stages.get("cache_commit", 0),
                    profiled_tokens_match_normal=profile.get("tokens_match_unprofiled"))
                rows.append(row)
    for row in rows:
        case, model, batch, mode = (row[k] for k in ("case", "model", "batch_size", "state_mode"))
        tokens, lengths, summary = cases[(case, model, batch, mode)]
        reference = cases.get((case, model, batch, "full_target_replay"))
        compact_control = cases.get((case, model, batch, "per_token"))
        if any((case, model, batch, m) not in cases for m in MODES):
            raise ValueError(f"incomplete three-mode results: {case}/{model}/B={batch}")
        if reference:
            rt, rl, rs = reference
            row["same_sample_ids"] = tokens.keys() == rt.keys()
            row["token_exact_vs_full_target_replay"] = tokens == rt
            row["accept_lengths_equal_vs_full_target_replay"] = lengths == rl
            row["speedup_vs_full_target_replay"] = summary["decode_tokens_per_second"]/rs["decode_tokens_per_second"]
        else:
            raise ValueError(f"missing full_target_replay baseline: {case}/{model}/B={batch}")
        row["token_exact_vs_per_token"] = tokens == compact_control[0] if compact_control else None
        row["accept_lengths_equal_vs_per_token"] = lengths == compact_control[1] if compact_control else None
        if not row["same_sample_ids"]:
            raise ValueError(f"different samples across state modes: {case}/{model}/B={batch}")
    lines = ["# Qwen3.5 state recovery comparison", "",
        "Same checkpoints and prompts; no retraining. Rates use normal decode-only passes.",
        "Stage times use separate instrumented passes. Prefill excluded; this is HF static batching.",
        "full_target_replay uses native HF verification; the two new modes share a fused recurrent verifier.",
        "Thus old-vs-new includes a verifier-kernel change; per_token-vs-compact_replay isolates the recording/commit strategy.",
        "compact_replay is per-block checkpoint + k/v/g/beta state-only replay, not upstream ReplaySSM's circular output-only kernel.",
        "Cross-mode token equality and acceptance equality are reported, not assumed. Differences confound direct timing comparisons.", ""]
    def f(value):
        return "—" if value is None else f"{value:.3f}"
    for case in sorted({r["case"] for r in rows}):
        lines += [f"## {case}", "",
            "| Model | B | Mode | tok/s | vs AR | vs full replay | Accept | Peak GiB | Records MiB | Tokens = full / per-token |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
        selected = [r for r in rows if r["case"] == case]
        for r in selected:
            lines.append(f"| {r['model']} | {r['batch_size']} | {r['state_mode']} | {f(r['decode_tokens_per_second'])} | "
                f"{f(r['speedup_vs_ar'])} | {f(r['speedup_vs_full_target_replay'])} | {f(r['mean_committed_length'])} | "
                f"{f(r['peak_allocated_gib'])} | {f(r['peak_transaction_record_mib'])} | "
                f"{r['token_exact_vs_full_target_replay']} / {r['token_exact_vs_per_token']} |")
        lines += ["", "Stage totals below cover the profiled sample subset, not the normal throughput pass.", "",
            "| Model | B | Mode | ms/round | Target calls/round | Verify ms | Full replay ms | State select/replay ms | Cache crop ms |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for r in selected:
            lines.append(f"| {r['model']} | {r['batch_size']} | {r['state_mode']} | {f(r['ms_per_round'])} | "
                f"{f(r['target_calls_per_round'])} | {f(r['verify_ms'])} | {f(r['full_target_replay_ms'])} | "
                f"{f(r['state_commit_ms'])} | {f(r['cache_commit_ms'])} |")
    (root/"state_modes.md").write_text("\n".join(lines)+"\n")
    if rows:
        with (root/"state_modes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
