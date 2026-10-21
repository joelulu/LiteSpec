"""Text, multimodal train/validation speedup tables without token-match gating."""
import argparse
import csv
import json
from pathlib import Path


def comparison_rows(root, selected_cases=None):
    combined = []
    cases = [("text", "text", "smoke"), ("multimodal_train", "multimodal", "train"),
             ("multimodal_val", "multimodal", "validation")]
    if not any((root / name).exists() for name in ("multimodal_train", "multimodal_val")):
        # Keep old two-stage results readable; their split was not recorded.
        cases = [("text", "text", "smoke"), ("multimodal", "multimodal", "unspecified")]
    known = dict(text=("text", "smoke"), multimodal_train=("multimodal", "train"),
                 multimodal_val=("multimodal", "validation"), multimodal=("multimodal", "unspecified"))
    if selected_cases is not None:
        cases = [(case, *known[case]) for case in selected_cases]
    else:
        cases = [case for case in cases if (root/case[0]/"summary.json").exists()]
    if not cases:
        raise ValueError("no completed dataset comparisons found")
    for case, modality, split in cases:
        source = json.loads((root / case / "summary.json").read_text())
        settings_path = root / case / "settings.json"
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        bases = {row["concurrency"]: row for row in source if row["model"] == "ar"}
        for row in source:
            base = bases[row["concurrency"]]
            combined.append(dict(case=case, modality=modality,
                dataset_split=row.get("dataset_split", split),
                dataset_path=row.get("dataset_path", settings.get("data", "")),
                dataset_name=row.get("dataset_name", settings.get("dataset_name", "unspecified")),
                model=row["model"], concurrency=row["concurrency"],
                batch_size=row.get("batch_size"),
                output_tokens_per_second=row["output_tokens_per_second"],
                throughput_speedup=row["output_tokens_per_second"] / base["output_tokens_per_second"],
                wall_time_speedup=base["wall_seconds"] / row["wall_seconds"],
                mean_latency_seconds=row["mean_latency_seconds"],
                mean_committed_length=row["mean_committed_length"],
                output_tokens=row["output_tokens"], wall_seconds=row["wall_seconds"],
                matched=row["matched"], requests=row["requests"],
                valid_token_match=row["valid_token_match"],
                scheduler=row["scheduler"],
                benchmark_mode=row.get("benchmark_mode", "round_robin_unbatched"),
                timing_scope=row.get("timing_scope", "prefill_and_decode"),
                prefill_seconds=row.get("prefill_seconds"), decode_seconds=row.get("decode_seconds"),
                end_to_end_tokens_per_second=row.get("end_to_end_tokens_per_second"),
                unique_prompts=row.get("unique_prompts", row["requests"]),
                throughput_is_production_benchmark=False))
    return combined


def tables(rows):
    modes = {row["benchmark_mode"] for row in rows}
    if len(modes) != 1:
        raise ValueError("do not mix static-batch and round-robin results in the same comparison")
    static = modes == {"static_replicated_batch"}
    axis = "B" if static else "C"
    output = [("静态 batch：每个 prompt 复制 B 份，一次 forward 处理整个 batch。speedup 按 decode-only tok/s / 同 B 的 AR decode-only tok/s。"
               if static else "Observed throughput speedup = output tok/s divided by AR output tok/s at the same concurrency."),
              ("Prefill 单独记录；不计入本表 speedup。Token match 不限制速率比。离线静态 batch，非线上 continuous batching。"
               if static else "Token mismatches do not suppress these ratios. HF round-robin, unbatched; not a serving benchmark."),
              "每格为 speedup / 平均接受长度（含保证推进的 token）；AR 无接受长度。", ""]
    titles = dict(text="纯文本", multimodal_train="多模态：训练集", multimodal_val="多模态：验证集",
                  multimodal="多模态（来源未标注）")
    for case in dict.fromkeys(row["case"] for row in rows):
        selected = [row for row in rows if row["case"] == case]
        concurrencies = sorted({row["concurrency"] for row in selected})
        models = list(dict.fromkeys(row["model"] for row in selected))
        indexed = {(row["model"], row["concurrency"]): row for row in selected}
        output += [f"### {titles[case]}", "", f"数据集：`{selected[0]['dataset_name']}`；划分：`{selected[0]['dataset_split']}`；路径：`{selected[0]['dataset_path']}`。", "",
                   "| 模型 | " + " | ".join(f"{axis}={c} speedup / 接受长度" for c in concurrencies) + " |",
                   "|---|" + "---:|" * len(concurrencies)]
        for model in models:
            cells = []
            for c in concurrencies:
                row = indexed.get((model, c))
                if row is None:
                    cells.append("—")
                else:
                    length = row["mean_committed_length"]
                    acceptance = f"{length:.3f}" if length is not None else "—"
                    cells.append(f"{row['throughput_speedup']:.3f}× / {acceptance}")
            output.append(f"| {model} | " + " | ".join(cells) + " |")
        output.append("")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=["text", "multimodal_train", "multimodal_val", "multimodal"])
    args = parser.parse_args()
    rows = comparison_rows(args.root, args.cases)
    with (args.root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "comparison.json").write_text(json.dumps(rows, indent=2))
    report = tables(rows)
    (args.root / "speedup_tables.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
