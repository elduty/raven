"""Tests for metrics.py — in-memory counters and Prometheus output."""

from raven.metrics import inc, observe, Timer, format_prometheus, _counters, _summaries, _lock


class TestMetrics:
    def setup_method(self):
        with _lock:
            _counters.clear()
            _summaries.clear()

    def test_inc_counter(self):
        inc("test_total")
        inc("test_total")
        assert "test_total 2" in format_prometheus()

    def test_inc_with_labels(self):
        inc("reviews_total", {"severity": "low", "repo": "owner/repo"})
        output = format_prometheus()
        assert 'reviews_total{repo="owner/repo",severity="low"} 1' in output

    def test_observe_summary(self):
        observe("duration_seconds", 1.5)
        observe("duration_seconds", 2.5)
        output = format_prometheus()
        assert "duration_seconds_sum 4.000" in output
        assert "duration_seconds_count 2" in output
        assert "_avg" not in output

    def test_timer(self):
        with Timer("test_timer"):
            pass  # near-zero duration
        output = format_prometheus()
        assert "test_timer_count 1" in output

    def test_empty_metrics(self):
        output = format_prometheus()
        assert output == "\n"
