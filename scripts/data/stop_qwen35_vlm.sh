#!/usr/bin/env bash
# Stop this model's local SGLang replicas, without matching unrelated models.
set -euo pipefail
TARGET_MODEL="${TARGET_MODEL:-$HOME/LMM/lmm_model/Qwen3.5-4B}" \
"${STOP_PYTHON:-python3}" - <<'PY'
import os
from pathlib import Path
import signal
import time

model = os.environ["TARGET_MODEL"]
ports = {str(p) for p in range(30000, 30008)}
selected = {}
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        command = (proc / "cmdline").read_bytes()
        args = command.decode().strip("\0").split("\0")
        if "sglang.launch_server" not in args:
            continue
        if args[args.index("--model-path") + 1] != model:
            continue
        if args[args.index("--port") + 1] not in ports:
            continue
        os.kill(int(proc.name), signal.SIGTERM)
        selected[int(proc.name)] = command
        print("Stopping server PID", proc.name, flush=True)
    except (OSError, ValueError, IndexError):
        continue

deadline = time.monotonic() + 20
while selected and time.monotonic() < deadline:
    for pid, command in list(selected.items()):
        try:
            if Path(f"/proc/{pid}/cmdline").read_bytes() == command:
                continue
        except OSError:
            pass
        del selected[pid]
    if selected:
        time.sleep(0.5)
if selected:
    raise SystemExit(f"Servers still stopping: {list(selected)}; inspect before training")
print("No matching server parents remain. Check nvidia-smi for GPU memory release.")
PY
