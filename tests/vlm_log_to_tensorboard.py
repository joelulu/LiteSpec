"""Convert an existing VLM train.log to TensorBoard; --follow watches a running job.

Does not import the trainer, load models, or use GPUs. Use a separate event
directory from native tracking to avoid duplicate points.
"""
import argparse
import json
from pathlib import Path
import time


def scalars_from_record(record):
    if not isinstance(record, dict) or "optimizer_step" not in record:
        return None
    values = {key: value for key, value in record.get("metrics", {}).items()
              if isinstance(value, (int, float))}
    values.setdefault("objective/loss", record.get("loss"))
    values["grad_norm"] = record.get("grad_norm")
    for key, value in record.get("learning_rates", {}).items():
        values[f"learning_rate/{key}"] = value
    values["completed_epochs"] = record.get("completed_epochs")
    return int(record["optimizer_step"]), {
        key: value for key, value in values.items() if isinstance(value, (int, float))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    output = args.output or args.log.parent / "tensorboard_from_log"
    count = 0
    print(f"Reading {args.log}; writing {output}", flush=True)
    try:
        with args.log.open(encoding="utf-8", errors="replace") as source, SummaryWriter(str(output)) as writer:
            pending = ""
            while True:
                line = source.readline()
                if not line:
                    writer.flush()
                    if not args.follow:
                        break
                    time.sleep(1)
                    continue
                pending += line
                if not pending.endswith("\n"):
                    continue
                try:
                    record = json.loads(pending)
                except (ValueError, TypeError):
                    record = None
                pending = ""
                parsed = scalars_from_record(record)
                if parsed is None:
                    continue
                step, values = parsed
                for key, value in values.items():
                    writer.add_scalar(key, value, global_step=step)
                count += 1
    except KeyboardInterrupt:
        pass
    print(f"Wrote {count} optimizer-step records to {output}", flush=True)


if __name__ == "__main__":
    main()
