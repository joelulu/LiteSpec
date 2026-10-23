"""Opt-in decode timeline instrumentation; resolve CUDA events only after decode."""
from collections import Counter
from contextlib import contextmanager
import importlib
import sys
import time


class DecodeTimeline:
    """Exclusive stage intervals on the current CUDA stream, or CPU wall time.

    CUDA intervals include stream idle time between launches: they are not a sum
    of kernel execution time. Host intervals are enqueue/control time and overlap
    device work; never add host and CUDA intervals together.
    """
    def __init__(self, device, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = str(device)
        self.cuda = self.device.startswith('cuda')
        self.intervals = []
        self.previous = None

    def mark(self, name):
        event = None
        if self.cuda:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record(self.torch.cuda.current_stream(self.device))
        now = time.perf_counter()
        if self.previous is not None:
            old_name, old_time, old_event = self.previous
            self.intervals.append((old_name, now-old_time, old_event, event))
        self.previous = (name, now, event) if name is not None else None

    def resolve(self, decode_seconds):
        if self.previous is not None:
            raise RuntimeError('decode timeline must end with mark(None)')
        host, timeline, counts = Counter(), Counter(), Counter()
        for name, wall, start, end in self.intervals:
            host[name] += wall
            timeline[name] += start.elapsed_time(end)/1000 if self.cuda else wall
            counts[name] += 1
        return dict(clock='cuda_stream_events' if self.cuda else 'cpu_wall',
                    stage_seconds=dict(timeline), host_stage_seconds=dict(host),
                    stage_calls=dict(counts), decode_seconds=decode_seconds,
                    unassigned_seconds=decode_seconds-sum(timeline.values()))


@contextmanager
def gdn_kernel_probe():
    """Trace actual HF Python fallback entry, separately from latency measurement.

    Function dispatch alone cannot tell whether a Hub/FLA kernel replaced the
    body. Count execution of the fallback body itself. Preserve an existing
    Python profiler callback. This probe adds host overhead, so run separately.
    """
    counts = Counter()
    old = sys.getprofile()
    names = {'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
             'causal_conv1d_fn', 'causal_conv1d_update'}
    def observe(frame, event, arg):
        if event == 'call':
            code = frame.f_code
            if code.co_name in names and code.co_filename.replace('\\', '/').endswith(
                    '/models/qwen3_5/modeling_qwen3_5.py'):
                query = frame.f_locals.get('query')
                length = query.shape[1] if query is not None and query.ndim >= 2 else 'unknown'
                counts[f'{code.co_name}:seq={length}'] += 1
        if old is not None:
            old(frame, event, arg)
    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(old)


def kernel_environment():
    import torch
    import transformers
    from importlib.util import find_spec
    result = dict(torch=torch.__version__, transformers=transformers.__version__,
                  cuda=torch.version.cuda)
    for name in ('fla', 'causal_conv1d', 'flash_attn', 'kernels'):
        try:
            result[name+'_importable'] = find_spec(name) is not None
        except (ValueError, ImportError):
            result[name+'_importable'] = False
    module = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    result['gdn_dispatch'] = {name: dict(module=getattr(module, name).__module__,
                                        name=getattr(module, name).__qualname__)
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')
        if hasattr(module, name)}
    return result
