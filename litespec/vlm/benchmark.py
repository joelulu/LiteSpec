"""Single-GPU, unbatched round-robin reference scheduler; not a serving engine."""
from collections import deque
from contextlib import contextmanager
import math
import time


class RequestPositionState:
    """Qwen3.5 stores MRoPE deltas on the shared model, outside the KV cache."""
    def __init__(self, target):
        self.slots = [(module, "rope_deltas") for module in target.modules()
                      if hasattr(module, "rope_deltas")]
        self.values = [None] * len(self.slots)

    @contextmanager
    def activate(self):
        previous = [getattr(module, name) for module, name in self.slots]
        for (module, name), value in zip(self.slots, self.values):
            setattr(module, name, value)
        try:
            yield
        finally:
            self.values = [getattr(module, name) for module, name in self.slots]
            for (module, name), value in zip(self.slots, previous):
                setattr(module, name, value)


def run_closed_loop(examples, factory, concurrency, *, synchronize=lambda: None):
    """Keep up to C independent caches active; execute one request step at a time.

    factory(index, example) -> (generator, request context manager provider).
    GPU forwards are unbatched/serialized. C measures in-flight requests, never
    multiple GPUs, copied model replicas, or simultaneous batch elements.
    """
    if concurrency < 1 or not examples:
        raise ValueError("positive concurrency and nonempty examples required")
    synchronize()
    start = time.perf_counter()
    queue = deque()
    records = [None] * len(examples)
    next_index, peak = 0, 0

    def fill():
        nonlocal next_index, peak
        while len(queue) < concurrency and next_index < len(examples):
            iterator, state = factory(next_index, examples[next_index])
            queue.append((next_index, iterator, state, time.perf_counter()))
            next_index += 1
        peak = max(peak, len(queue))

    fill()
    try:
        while queue:
            index, iterator, state, arrived = queue.popleft()
            try:
                with state.activate():
                    next(iterator)
            except StopIteration as completed:
                records[index] = dict(result=completed.value,
                                      latency_seconds=time.perf_counter() - arrived)
                # Free completed generator/position tensors before admitting more.
                del iterator, state
                fill()
            else:
                queue.append((index, iterator, state, arrived))
        synchronize()
    finally:
        for _, iterator, _, _ in queue:
            iterator.close()
    return records, time.perf_counter() - start, peak


def summarize(records, elapsed):
    lengths = [n for row in records for n in row["result"]["accept_lengths"]]
    tokens = sum(len(row["result"]["token_ids"]) for row in records)
    latencies = sorted(row["latency_seconds"] for row in records)
    prefill = sum(row["result"].get("prefill_seconds", 0) for row in records)
    draft = sum(row["result"].get("draft_seconds", 0) for row in records)
    verify = sum(row["result"].get("verify_seconds", 0) for row in records)
    recovery = sum(row["result"].get("recovery_seconds", 0) for row in records)
    return dict(requests=len(records), output_tokens=tokens, wall_seconds=elapsed,
                output_tokens_per_second=tokens / elapsed,
                requests_per_second=len(records) / elapsed,
                mean_latency_seconds=sum(latencies) / len(latencies),
                p95_latency_seconds=latencies[max(0, math.ceil(.95 * len(latencies)) - 1)],
                mean_committed_length=sum(lengths) / len(lengths) if lengths else None,
                mean_accepted_draft_tokens=sum(n-1 for n in lengths) / len(lengths) if lengths else None,
                speculative_rounds=len(lengths), prefill_seconds=prefill, draft_seconds=draft,
                verify_and_snapshot_seconds=verify, recovery_seconds=recovery,
                other_seconds=max(0, elapsed-prefill-draft-verify-recovery),
                scheduler="round_robin_unbatched", scope="single_gpu_hf_reference")
