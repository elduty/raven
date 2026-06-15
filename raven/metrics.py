"""metrics.py — In-memory counters exposed via /metrics in Prometheus format."""

import threading
import time

_lock = threading.Lock()
# Counter values are int (from inc()) or float (from add() — e.g. fractional
# USD cost). Prometheus counters are floats by spec; we just store whichever
# Python type the arithmetic produces and format appropriately on exposition.
_counters: dict[str, float] = {}
_summaries: dict[str, tuple[float, int]] = {}  # (sum, count) — bounded, no list growth


def add(name: str, amount: float, labels: dict[str, str] | None = None) -> None:
    """Add ``amount`` to a counter (any non-negative number).

    Used for quantities that aren't unit increments — token counts, dollar
    cost. ``amount`` may be int or float; the stored value's type follows the
    arithmetic (int + int stays int; anything touching a float becomes float).
    """
    key = _key(name, labels)
    with _lock:
        _counters[key] = _counters.get(key, 0) + amount


def inc(name: str, labels: dict[str, str] | None = None) -> None:
    """Increment a counter by 1."""
    add(name, 1, labels)


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


# One-line HELP text per metric family. TYPE is inferred (counter vs
# summary) from which store a name lives in, so a new metric still gets a
# valid `# TYPE` line even if it's not listed here; HELP is emitted only
# when known. A missing entry is non-fatal (just no HELP line), so this
# never blocks adding a new counter.
_METRIC_HELP: dict[str, str] = {
    "raven_reviews_total": "PR reviews completed.",
    "raven_reviews_skipped_total": "Reviews skipped before running, by reason.",
    "raven_merges_total": "PRs auto-merged by Raven.",
    "raven_auto_merge_queued_total": "Auto-merges queued via Gitea native merge-when-checks-succeed.",
    "raven_ci_failures_total": "Reviews where CI failed and the merge was skipped.",
    "raven_errors_total": "Errors, by type.",
    "raven_review_failures_total": "Classified review failures, by reason (timeout/rate_limit/backend_5xx/usage_limit/auth/unknown) and repo.",
    "raven_responses_total": "Comment replies posted.",
    "raven_response_parse_errors_total": "Comment-reply AI outputs that failed to parse.",
    "raven_retractions_total": "Finding-retraction attempts, by result.",
    "raven_verdict_revisions_total": "Comment-driven verdict revisions.",
    "raven_revision_submit_errors_total": "Verdict-revision submit failures.",
    "raven_user_resolved_findings_dropped_total": "User-resolved findings dropped from carry-forward.",
    "raven_carried_findings_dropped_total": "Carried findings dropped by re-validation (model-reported as resolved by the push).",
    "raven_cache_save_failures_total": "Findings-cache disk-write failures, by exception type.",
    "raven_review_duration_seconds": "Wall-clock duration of a PR review.",
    "raven_ai_tokens_total": "AI tokens consumed, by kind (input/output).",
    "raven_ai_cost_usd_total": "AI cost in USD (provider-reported, or estimated from the price table).",
    "raven_ai_calls_total": "AI completion calls.",
}


def _fmt_value(v: float) -> str:
    """Render a counter value: integral values bare (``42``), fractional
    values trimmed to ≤6 decimals (``0.178749``). Keeps token counts clean
    while exposing fractional USD cost without scientific notation."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _split_key(key: str) -> tuple[str, str]:
    """Split a stored series key into ``(base_name, label_block)``.

    ``raven_x{a="b"}`` → ``("raven_x", '{a="b"}')``; ``raven_x`` →
    ``("raven_x", "")``. Used so the summary ``_sum`` / ``_count`` suffix
    lands on the base name BEFORE the label braces, per the Prometheus
    exposition spec — ``raven_x_sum{a="b"}``, not the invalid
    ``raven_x{a="b"}_sum``.
    """
    brace = key.find("{")
    if brace == -1:
        return key, ""
    return key[:brace], key[brace:]


def format_prometheus() -> str:
    """Render all metrics in Prometheus text exposition format.

    Emits ``# HELP`` (when known) and ``# TYPE`` once per metric family,
    then the series. Counters render as-is. Summaries render
    ``<name>_sum`` / ``<name>_count`` with the suffix on the base name
    (before the label braces) so Prometheus parses them correctly.
    """
    lines: list[str] = []
    with _lock:
        # Counters — group label-sets under their base name so HELP/TYPE
        # is emitted once per family, not once per label combination.
        counters_by_name: dict[str, list[tuple[str, int]]] = {}
        for key, value in _counters.items():
            base, labels = _split_key(key)
            counters_by_name.setdefault(base, []).append((labels, value))
        for base in sorted(counters_by_name):
            help_text = _METRIC_HELP.get(base)
            if help_text:
                lines.append(f"# HELP {base} {help_text}")
            lines.append(f"# TYPE {base} counter")
            for labels, value in sorted(counters_by_name[base]):
                lines.append(f"{base}{labels} {_fmt_value(value)}")

        # Summaries — _sum / _count suffix on the base name, before labels.
        summaries_by_name: dict[str, list[tuple[str, float, int]]] = {}
        for key, (total, count) in _summaries.items():
            base, labels = _split_key(key)
            summaries_by_name.setdefault(base, []).append((labels, total, count))
        for base in sorted(summaries_by_name):
            help_text = _METRIC_HELP.get(base)
            if help_text:
                lines.append(f"# HELP {base} {help_text}")
            lines.append(f"# TYPE {base} summary")
            for labels, total, count in sorted(summaries_by_name[base]):
                lines.append(f"{base}_sum{labels} {total:.3f}")
                lines.append(f"{base}_count{labels} {count}")
    return "\n".join(lines) + "\n"


def _key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"
