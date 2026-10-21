"""Show ETA for a training job already running, using new log arrivals only."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litespec.vlm.progress import StepProgress


def step_record(line):
    try:
        record = json.loads(line)
        if isinstance(record, dict) and "optimizer_step" in record:
            return record
    except ValueError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "effective_config.json").read_text())
    total = config["total_optimizer_steps"]
    if config.get("debug_max_steps"):
        total = min(total, config["debug_max_steps"])
    with (args.run / "train.log").open(encoding="utf-8", errors="replace") as source:
        last = 0
        # Historical records have no timestamps: do not infer their throughput.
        for line in source:
            record = step_record(line)
            if record:
                last = record["optimizer_step"]
        timer = StepProgress(total, last)
        print(f"Step {last}/{total}; ETA: waiting for new step logs (Ctrl-C stops monitoring only)", flush=True)
        pending = ""
        first_arrival = True
        while last < total:
            line = source.readline()
            if not line:
                time.sleep(1)
                continue
            pending += line
            if not pending.endswith("\n"):
                continue
            record = step_record(pending)
            pending = ""
            if not record or record["optimizer_step"] <= last:
                continue
            progress = timer.update(record["optimizer_step"])
            # First arrival is a partial interval; discard it from estimates.
            if first_arrival:
                first_arrival = False
                timer.intervals.clear()
                eta, speed = "estimating", "?"
            else:
                eta = progress["eta"]
                speed = f"{progress['seconds_per_optimizer_step']:.2f}"
            last = record["optimizer_step"]
            print(f"Step {last}/{total} ({progress['progress_percent']:.1f}%) | "
                  f"{speed} s/update | remaining {progress['remaining_steps']} | ETA {eta}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
