"""Compare AR and original DLite across named datasets and input modalities."""
import argparse
import csv
import json
from pathlib import Path


def collect(root, datasets, modes):
    combined = []
    for dataset in datasets:
        for mode in modes:
            rows = json.loads((root/dataset/mode/"summary.json").read_text())
            for batch in sorted({row["batch_size"] for row in rows}):
                pair = {row["model"]: row for row in rows if row["batch_size"] == batch}
                ar = pair["ar"]
                for label in [key for key in pair if key != "ar"] or ["ar"]:
                    draft = pair[label]
                    combined.append(dict(dataset=dataset, mode=mode, model=label, batch_size=batch,
                        unique_prompts=draft["unique_prompts"], sequences=draft["requests"],
                        mean_input_tokens=draft["mean_input_tokens"], mean_image_tokens=draft.get("mean_image_tokens", 0),
                        mean_output_tokens=draft["output_tokens"]/draft["requests"],
                        ar_decode_tokens_per_second=ar["decode_tokens_per_second"],
                        model_decode_tokens_per_second=draft["decode_tokens_per_second"],
                        decode_speedup=draft["decode_tokens_per_second"]/ar["decode_tokens_per_second"],
                        ar_end_to_end_tokens_per_second=ar["end_to_end_tokens_per_second"],
                        model_end_to_end_tokens_per_second=draft["end_to_end_tokens_per_second"],
                        end_to_end_speedup=draft["end_to_end_tokens_per_second"]/ar["end_to_end_tokens_per_second"],
                        mean_committed_length=draft["mean_committed_length"],
                        ar_prefill_seconds=ar["prefill_seconds"], model_prefill_seconds=draft["prefill_seconds"],
                        ar_decode_seconds=ar["decode_seconds"], model_decode_seconds=draft["decode_seconds"],
                        matched=draft["matched"], requests=draft["requests"], valid_token_match=draft["valid_token_match"],
                        ar_peak_allocated_gib=ar["peak_allocated_gib"], model_peak_allocated_gib=draft["peak_allocated_gib"]))
    return combined


def render(rows, modes):
    text = ["AR 与所选 draft；相同问题/样本，静态复制 batch；主 speedup 为 decode-only 吞吐比。",
            "接受长度含保证推进的 token。保留 match 但不限制速率比。没有计算数据集任务准确率。", ""]
    for mode in modes:
        text += ["### " + ("图文输入" if mode == "vision" else "同题去图对照"), "",
            "| 数据集 | 模型 | B | 独立样本 | 输入 tokens | 图像 tokens | 输出/序列 | AR tok/s | Draft tok/s | speedup | 接受长度 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["mode"] != mode:
                continue
            length = row["mean_committed_length"]
            length = f"{length:.3f}" if length is not None else "—"
            text.append(f"| {row['dataset']} | {row['model']} | {row['batch_size']} | {row['unique_prompts']} | {row['mean_input_tokens']:.1f} | "
                f"{row['mean_image_tokens']:.1f} | {row['mean_output_tokens']:.1f} | {row['ar_decode_tokens_per_second']:.2f} | "
                f"{row['model_decode_tokens_per_second']:.2f} | {row['decode_speedup']:.3f}× | {length} |")
        text.append("")
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=["text", "vision"], required=True)
    args = parser.parse_args()
    rows = collect(args.root, args.datasets, args.modes)
    with (args.root/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root/"comparison.json").write_text(json.dumps(rows, indent=2))
    report = render(rows, args.modes)
    (args.root/"comparison.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
