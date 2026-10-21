"""Own eight SGLang process groups until generation completes, then clean up."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .serving_data import http_json, output_lock


def stop_owned(processes, grace=30):
    # Every server is started in its own session. Never kill unrelated services.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        alive = False
        for process in processes:
            process.poll()
            try:
                os.killpg(process.pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(.25)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_ready(processes, addresses, model, timeout):
    deadline = time.monotonic() + timeout
    ready = set()
    while len(ready) < len(addresses):
        for index, (process, address) in enumerate(zip(processes, addresses)):
            if process.poll() is not None:
                raise RuntimeError(f"server {index} exited with {process.returncode}; inspect its log")
            if index in ready:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"servers not ready after {timeout}s: {sorted(set(range(len(addresses))) - ready)}")
            try:
                from urllib.request import build_opener, ProxyHandler
                with build_opener(ProxyHandler({})).open(address + "/health", timeout=2) as response:
                    if response.status != 200:
                        continue
                models = http_json(address + "/v1/models", timeout=2)
                if model not in {row["id"] for row in models.get("data", [])}:
                    continue
            except (OSError, ValueError, KeyError):
                continue
            ready.add(index)
            print(f"Ready {len(ready)}/{len(addresses)}: {address}", flush=True)
        if len(ready) < len(addresses):
            time.sleep(1)


def main():
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    gpus = env.get("DATA_GPUS", "0,1,2,3,4,5,6,7").split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or any(not x.strip() for x in gpus):
        raise ValueError("DATA_GPUS must identify eight distinct GPUs")
    server_python = env.get("SGLANG_PYTHON", str(Path.home() / ".conda/envs/dflash/bin/python"))
    subprocess.run([server_python, "-c", "import sglang, torch; assert torch.cuda.device_count() == 8, 'eight GPUs must be visible to the serving environment'"],
                   env={**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}, check=True)
    base_port = int(env.get("BASE_PORT", "30000"))
    if not 1 <= base_port <= 65528:
        raise ValueError("BASE_PORT must permit eight consecutive TCP ports")
    for port in range(base_port, base_port + 8):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    env.setdefault("MAX_SAMPLES", "20000")
    env.setdefault("IMAGE_MAX_TOKENS", "1024")
    env.setdefault("MAX_LENGTH", "4096")
    env.setdefault("CONTEXT_LENGTH", env["MAX_LENGTH"])
    env.setdefault("MAX_NEW_TOKENS", "256")
    env.setdefault("MIN_ANSWER_TOKENS", "1")
    if int(env["MAX_SAMPLES"]) < 0 or int(env["IMAGE_MAX_TOKENS"]) < 0:
        raise ValueError("sample/image caps must be nonnegative")
    if not 0 < int(env["MAX_NEW_TOKENS"]) < int(env["MAX_LENGTH"]) <= int(env["CONTEXT_LENGTH"]):
        raise ValueError("require 0 < MAX_NEW_TOKENS < MAX_LENGTH <= CONTEXT_LENGTH")
    if not 1 <= int(env["MIN_ANSWER_TOKENS"]) <= int(env["MAX_NEW_TOKENS"]):
        raise ValueError("require 1 <= MIN_ANSWER_TOKENS <= MAX_NEW_TOKENS")
    source = Path(env.get("SOURCE_DATA", str(Path.home() / "LMM/liyt/datasets/FineVision/FineVision_subset/finevision_sharegpt.jsonl")))
    target = Path(env.get("TARGET_MODEL", str(Path.home() / "LMM/lmm_model/Qwen3.5-4B")))
    if not source.is_file() or not target.is_dir():
        raise ValueError(f"missing source manifest or local target model: {source}; {target}")
    output = Path(env.get("PREPARE_OUTPUT", str(repo.parent / "data" / f"finevision_qwen35_sglang_{env['MAX_SAMPLES']}_img{env['IMAGE_MAX_TOKENS']}"))).resolve()
    env["PREPARE_OUTPUT"] = str(output)
    logs = output / "serving_logs"
    logs.mkdir(parents=True, exist_ok=True)
    model = env.get("SERVED_MODEL_NAME", "qwen35-vlm")
    addresses = [f"http://127.0.0.1:{base_port+i}" for i in range(8)]
    processes, handles = [], []
    client = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Separate lock: generator owns output/.writer.lock itself.
        with output_lock(output / "pipeline_lock"):
            try:
                for i, gpu in enumerate(gpus):
                    handle = (logs / f"gpu{gpu}_port{base_port+i}.log").open("a")
                    handles.append(handle)
                    child_env = {**env, "SGLANG_PYTHON": server_python, "CUDA_VISIBLE_DEVICES": gpu,
                                 "PORT": str(base_port+i), "TP_SIZE": "1", "HOST": "127.0.0.1"}
                    process = subprocess.Popen(["bash", str(repo / "scripts/data/serve_qwen35_vlm.sh")],
                                               env=child_env, stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    processes.append(process)
                    print(f"Starting GPU={gpu} port={base_port+i} PID={process.pid}", flush=True)
                (logs / "owned_servers.json").write_text(json.dumps([dict(pid=p.pid, address=a) for p, a in zip(processes, addresses)], indent=2))
                wait_ready(processes, addresses, model, int(env.get("STARTUP_TIMEOUT", "1800")))
                print("All eight services ready; generating data", flush=True)
                client = subprocess.Popen(["bash", str(repo / "scripts/data/prepare_finevision_serving.sh"), *addresses],
                                          env=env, start_new_session=True)
                while client.poll() is None:
                    if any(p.poll() is not None for p in processes):
                        raise RuntimeError("a serving process exited during generation; inspect serving_logs")
                    time.sleep(1)
                if client.returncode:
                    raise RuntimeError(f"generation failed with exit code {client.returncode}; output supports retry")
                print(f"Data ready: {output}", flush=True)
            finally:
                # Cleanup must also complete after a second interrupt.
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                if client is not None and client.poll() is None:
                    stop_owned([client], grace=5)
                stop_owned(processes)
    finally:
        for handle in handles:
            handle.close()
        print("Owned serving processes stopped", flush=True)


if __name__ == "__main__":
    main()
