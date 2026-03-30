"""metrics.py — In-memory counters exposed via /metrics in Prometheus format."""

import threading
import time

_lock = threading.Lock()
_counters: dict[str, int] = {}
_summaries: dict[str, tuple[float, int]] = {}  # (sum, count) — bounded, no list growth


def inc(name: str, labels: dict[str, str] | None = None) -> None:
    """Increment a counter."""
    key = _key(name, labels)
    with _lock:
        _counters[key] = _counters.get(key, 0) + 1


def observe(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Record a summary observation (keeps only sum + count)."""
    key = _key(name, labels)
    with _lock:
        total, count = _summaries.get(key, (0.0, 0))
        _summaries[key] = (total + value, count + 1)


class Timer:
    """Context manager that records duration to a summary."""
    def __init__(self, name: str, labels: dict[str, str] | None = None):
        self.name = name
        self.labels = labels
        self.start = 0.0

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        observe(self.name, time.monotonic() - self.start, self.labels)


def format_prometheus() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines: list[str] = []
    with _lock:
        for key, value in sorted(_counters.items()):
            lines.append(f"{key} {value}")
        for key, (total, count) in sorted(_summaries.items()):
            lines.append(f"{key}_sum {total:.3f}")
            lines.append(f"{key}_count {count}")
    return "\n".join(lines) + "\n"


def _key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"
