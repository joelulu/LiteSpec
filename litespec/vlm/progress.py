"""Wall-clock progress estimates; no model or accelerator dependencies."""
from collections import deque
import time


class StepProgress:
    def __init__(self, total, step=0, *, now=None):
        self.total = total
        self.step = step
        self.last_time = time.monotonic() if now is None else now
        self.intervals = deque(maxlen=50)

    def update(self, step, *, now=None):
        now = time.monotonic() if now is None else now
        if step > self.step:
            self.intervals.append((now - self.last_time, step - self.step))
        self.step, self.last_time = step, now
        seconds = (sum(t for t, _ in self.intervals) / sum(n for _, n in self.intervals)
                   if self.intervals else None)
        remaining = max(0, self.total - step)
        eta = seconds * remaining if seconds is not None else None
        return dict(total_optimizer_steps=self.total, remaining_steps=remaining,
                    progress_percent=100 * min(step, self.total) / max(1, self.total),
                    seconds_per_optimizer_step=seconds, eta_seconds=eta,
                    eta=format_duration(eta))


def format_duration(seconds):
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
