"""Two observed throughput speedup tables, independent of token-match flags."""
import argparse
import csv
import json
from pathlib import Path


def comparison_rows(root):
    combined = []
    for modality in ("text", "multimodal"):
        source = json.loads((root / modality / "summary.json").read_text())
        bases = {row["concurrency"]: row for row in source if row["model"] == "ar"}
        for row in source:
            base = bases[row["concurrency"]]
            combined.append(dict(modality=modality, model=row["model"], concurrency=row["concurrency"],
                output_tokens_per_second=row["output_tokens_per_second"],
                throughput_speedup=row["output_tokens_per_second"] / base["output_tokens_per_second"],
                wall_time_speedup=base["wall_seconds"] / row["wall_seconds"],
                mean_latency_seconds=row["mean_latency_seconds"],
                mean_committed_length=row["mean_committed_length"],
                output_tokens=row["output_tokens"], wall_seconds=row["wall_seconds"],
                matched=row["matched"], requests=row["requests"],
                valid_token_match=row["valid_token_match"],
                scheduler=row["scheduler"], throughput_is_production_benchmark=False))
    return combined


def tables(rows):
    output = ["Observed throughput speedup = output tok/s divided by AR output tok/s at the same concurrency.",
              "Token mismatches do not suppress these ratios. HF round-robin, unbatched; not a serving benchmark.", ""]
    for modality, title in (("text", "纯文本"), ("multimodal", "多模态")):
        selected = [row for row in rows if row["modality"] == modality]
        concurrencies = sorted({row["concurrency"] for row in selected})
        models = list(dict.fromkeys(row["model"] for row in selected))
        indexed = {(row["model"], row["concurrency"]): row for row in selected}
        output += [f"### {title}", "", "| 模型 | " + " | ".join(f"C={c} speedup" for c in concurrencies) + " |",
                   "|---|" + "---:|" * len(concurrencies)]
        for model in models:
            cells = [f"{indexed[(model, c)]['throughput_speedup']:.3f}×" if (model, c) in indexed else "—"
                     for c in concurrencies]
            output.append(f"| {model} | " + " | ".join(cells) + " |")
        output.append("")
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = comparison_rows(args.root)
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
