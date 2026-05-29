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

    def test_labeled_summary_suffix_on_base_name(self):
        """Regression: the _sum/_count suffix must go on the BASE NAME,
        before the label braces. `name{labels}_sum` is invalid Prometheus
        and fails to scrape; `name_sum{labels}` is correct."""
        observe("raven_review_duration_seconds", 3.0, {"repo": "o/r"})
        observe("raven_review_duration_seconds", 1.0, {"repo": "o/r"})
        output = format_prometheus()
        assert 'raven_review_duration_seconds_sum{repo="o/r"} 4.000' in output
        assert 'raven_review_duration_seconds_count{repo="o/r"} 2' in output
        # The broken shape must NOT appear.
        assert 'raven_review_duration_seconds{repo="o/r"}_sum' not in output

    def test_help_and_type_headers_for_known_metric(self):
        inc("raven_reviews_total", {"severity": "low", "repo": "o/r"})
        output = format_prometheus()
        assert "# HELP raven_reviews_total PR reviews completed." in output
        assert "# TYPE raven_reviews_total counter" in output
        # HELP/TYPE precede the series line.
        assert output.index("# TYPE raven_reviews_total") < output.index("raven_reviews_total{")

    def test_type_inferred_for_unknown_metric(self):
        """A metric not in the HELP registry still gets a valid TYPE line
        (inferred from the store), just no HELP."""
        inc("some_new_total")
        observe("some_new_seconds", 0.5)
        output = format_prometheus()
        assert "# TYPE some_new_total counter" in output
        assert "# TYPE some_new_seconds summary" in output
        assert "# HELP some_new_total" not in output  # unknown → no HELP

    def test_type_emitted_once_per_family(self):
        """Multiple label-sets of the same metric share a single TYPE line."""
        inc("raven_errors_total", {"type": "a", "repo": "o/r"})
        inc("raven_errors_total", {"type": "b", "repo": "o/r"})
        output = format_prometheus()
        assert output.count("# TYPE raven_errors_total counter") == 1
