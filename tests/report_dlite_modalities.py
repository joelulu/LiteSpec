"""Summarize staged text/native and image/reference diagnostics; optionally gate images."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(root):
    rows = []
    for stage in ("text_native", "text_reference", "multimodal_reference"):
        path = root / stage / "summary.json"
        if path.exists():
            rows.extend({"stage": stage, **row} for row in json.loads(path.read_text()))
    return rows


def text_ready(rows):
    native = next((row for row in rows if row["stage"] == "text_native" and row["model"] == "dlite_original"), None)
    reference = next((row for row in rows if row["stage"] == "text_reference" and row["model"] == "dlite_original"), None)
    return bool(native and reference and native["valid_token_match"] and reference["valid_token_match"]
                and native["speedup_valid"] and native["speedup_vs_ar"] > 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check-text", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.root)
    (args.root / "analysis.json").write_text(json.dumps(rows, indent=2))
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (args.root / "analysis.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print("stage                 model            match    tok/s    wall_s   accept   speedup")
    for row in rows:
        acceptance = row.get("mean_committed_length")
        speedup = row.get("speedup_vs_ar")
        a = f"{acceptance:.3f}" if acceptance is not None else "-"
        s = f"{speedup:.3f}x" if speedup is not None else "INVALID"
        print(f"{row['stage']:<21} {row['model']:<15} {row['matched']:>2}/{row['requests']:<2} "
              f"{row['output_tokens_per_second']:>8.2f} {row['wall_seconds']:>8.2f} {a:>8} {s:>9}")
        if row.get("stage_profile_available") and row["model"] != "ar":
            wall = row["wall_seconds"]
            print("  time: " + ", ".join(f"{name}={row[key]:.2f}s ({100*row[key]/wall:.1f}%)"
                  for name, key in (("prefill", "prefill_seconds"), ("draft", "draft_seconds"),
                                    ("verify+snapshot", "verify_and_snapshot_seconds"),
                                    ("replay", "recovery_seconds"), ("other", "other_seconds"))))
    if args.check_text and not text_ready(rows):
        raise SystemExit("TEXT GATE NOT PASSED: need token-exact native/reference text results and native speedup >1. Images are not started by default.")


if __name__ == "__main__":
    main()
