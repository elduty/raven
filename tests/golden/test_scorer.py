"""Offline unit tests for the golden-review scorer.

These run in **normal CI** — they exercise ``tests/golden/scorer.py``
against hand-written ``review`` dicts with NO live AI call and NO tokens.
This is the "test the scorer itself" half of the harness: it proves the
measurement instrument is calibrated (approvable/severity buckets, the
must-not-contain false-positive guard, the must-contain real-finding
guard, coverage-gap, and the precision/recall/FP-rate aggregation)
before any live run trusts its numbers.
"""

from __future__ import annotations

import pytest

from tests.golden.scorer import (
    AggregateScore,
    ScenarioScore,
    score_scenario,
)


# --------------------------------------------------------------------
# verdict-bucket: approvable / not_approvable
# --------------------------------------------------------------------

def test_approvable_passes_on_clean_low_review():
    review = {"severity": "low", "summary": "fine", "findings": []}
    score = score_scenario(review, {"approvable": True})
    assert score.passed
    assert [c.name for c in score.checks] == ["approvable"]


def test_approvable_fails_when_top_severity_medium():
    review = {"severity": "medium", "summary": "", "findings": []}
    score = score_scenario(review, {"approvable": True})
    assert not score.passed


def test_approvable_fails_when_a_finding_is_high_even_if_top_low():
    # The model can bury a high finding under a low top-level severity;
    # the effective severity must catch it.
    review = {
        "severity": "low",
        "findings": [{"severity": "high", "message": "boom"}],
    }
    score = score_scenario(review, {"approvable": True})
    assert not score.passed
    assert "high" in score.failures[0].detail


def test_approvable_fails_on_parse_error_even_if_low():
    review = {"severity": "low", "findings": [], "_parse_error": True}
    score = score_scenario(review, {"approvable": True})
    assert not score.passed
    assert "parse_error" in score.failures[0].detail


def test_not_approvable_passes_when_medium_plus():
    review = {"severity": "medium", "findings": [{"severity": "medium", "message": "x"}]}
    score = score_scenario(review, {"approvable": False})
    assert score.passed


def test_not_approvable_fails_on_clean_approve():
    review = {"severity": "low", "findings": []}
    score = score_scenario(review, {"approvable": False})
    assert not score.passed


# --------------------------------------------------------------------
# verdict-bucket: max_severity cap
# --------------------------------------------------------------------

@pytest.mark.parametrize("top,findings,cap,ok", [
    ("low", [], "low", True),
    ("low", [{"severity": "medium", "message": "m"}], "low", False),
    ("medium", [], "medium", True),
    ("high", [], "medium", False),
    ("low", [{"severity": "high", "message": "h"}], "high", True),
])
def test_max_severity_cap(top, findings, cap, ok):
    review = {"severity": top, "findings": findings}
    score = score_scenario(review, {"max_severity": cap})
    assert score.passed is ok


# --------------------------------------------------------------------
# off-scheme severities must not crash the scorer (LLM-variance robustness)
# --------------------------------------------------------------------

@pytest.mark.parametrize("bad_sev", ["critical", "CRITICAL", "blocker", "info", "", "warning"])
def test_off_scheme_finding_severity_does_not_crash_approvable(bad_sev):
    # A model is free to emit an off-scheme severity like 'critical'.
    # Indexing SEVERITY_ORDER[...] with brackets on that raw string would
    # KeyError and crash score_scenario mid-run; the scorer must clamp it.
    review = {"severity": "low",
              "findings": [{"severity": bad_sev, "message": "something"}]}
    score = score_scenario(review, {"approvable": True})
    # An unknown finding severity clamps to 'low' -> still reads approvable.
    assert score.passed


@pytest.mark.parametrize("bad_top", ["critical", "blocker", "", "n/a"])
def test_off_scheme_top_severity_does_not_crash(bad_top):
    review = {"severity": bad_top, "findings": []}
    # Both the approvable and max_severity checks index the effective
    # severity; neither may raise on an off-scheme top-level value.
    score = score_scenario(review, {"approvable": True, "max_severity": "high"})
    assert score.passed  # unknown top clamps to 'low'


def test_off_scheme_severity_still_orders_known_tiers():
    # A real high finding alongside an off-scheme one must still win the
    # effective severity (off-scheme clamps to low, doesn't mask the high).
    review = {"severity": "low", "findings": [
        {"severity": "critical", "message": "off-scheme"},
        {"severity": "high", "message": "real high"},
    ]}
    score = score_scenario(review, {"approvable": True})
    assert not score.passed  # the real high blocks approval


# --------------------------------------------------------------------
# coverage_gap
# --------------------------------------------------------------------

def test_coverage_gap_true_match():
    review = {"severity": "high", "findings": [], "coverage_gap": True}
    assert score_scenario(review, {"coverage_gap": True}).passed


def test_coverage_gap_false_when_absent_key():
    review = {"severity": "low", "findings": []}
    assert score_scenario(review, {"coverage_gap": False}).passed


def test_coverage_gap_mismatch_fails():
    review = {"severity": "low", "findings": []}
    assert not score_scenario(review, {"coverage_gap": True}).passed


# --------------------------------------------------------------------
# must-NOT-contain (false-positive guard) + FP counting
# --------------------------------------------------------------------

def test_must_not_contain_passes_when_no_match():
    review = {"severity": "low", "findings": [{"severity": "low", "message": "looks good"}]}
    expect = {"must_not_contain_patterns": [r"\bmissing\b"]}
    score = score_scenario(review, expect)
    assert score.passed
    assert score.false_positives == 0
    assert score.negatives == 1


def test_must_not_contain_fails_and_counts_fp():
    review = {
        "severity": "high",
        "findings": [{"severity": "high", "message": "the implementation is missing"}],
    }
    expect = {"must_not_contain_patterns": [r"\bmissing\b", r"\babsent\b"]}
    score = score_scenario(review, expect)
    assert not score.passed
    assert score.false_positives == 1  # only the 'missing' pattern matched
    assert score.negatives == 2


def test_must_not_contain_is_case_insensitive():
    review = {"findings": [{"message": "This Code Is ABSENT from the PR"}]}
    expect = {"must_not_contain_patterns": [r"\babsent\b"]}
    assert not score_scenario(review, expect).passed


def test_negatives_override_sets_denominator():
    review = {"findings": []}
    expect = {"must_not_contain_patterns": [r"x"], "negatives": 5}
    score = score_scenario(review, expect)
    assert score.negatives == 5
    assert score.false_positives == 0


def test_must_not_contain_does_not_match_across_separate_findings():
    # Anchor in finding A, negation in finding B. A blob-match with a
    # gap-spanning pattern would wrongly fire; per-message matching must
    # not. Neither message individually contains both halves.
    review = {"findings": [
        {"message": "The implementation looks solid."},
        {"message": "A docstring is missing on the helper."},
    ]}
    expect = {"must_not_contain_patterns": [
        r"\bimplementation\b[^.\n]{0,40}\bmissing\b",
    ]}
    score = score_scenario(review, expect)
    assert score.passed
    assert score.false_positives == 0


# --------------------------------------------------------------------
# must-contain (real-finding guard) + TP/FN counting
# --------------------------------------------------------------------

def test_must_contain_tp_when_file_and_keyword_match():
    review = {
        "severity": "high",
        "findings": [
            {"severity": "high", "file": "api/users.py",
             "message": "SQL injection via string concatenation"},
        ],
    }
    expect = {"must_contain": [
        {"file_pattern": r"api/users\.py", "message_patterns": [r"sql\s*injection"]},
    ]}
    score = score_scenario(review, expect)
    assert score.passed
    assert score.true_positives == 1
    assert score.false_negatives == 0


def test_must_contain_fn_when_keyword_present_but_wrong_file():
    review = {
        "findings": [
            {"file": "other/thing.py", "message": "SQL injection here"},
        ],
    }
    expect = {"must_contain": [
        {"file_pattern": r"api/users\.py", "message_patterns": [r"sql\s*injection"]},
    ]}
    score = score_scenario(review, expect)
    assert not score.passed
    assert score.false_negatives == 1
    assert score.true_positives == 0


def test_must_contain_fn_when_file_matches_but_no_keyword():
    review = {
        "findings": [
            {"file": "api/users.py", "message": "consider adding a docstring"},
        ],
    }
    expect = {"must_contain": [
        {"file_pattern": r"api/users\.py", "message_patterns": [r"sql\s*injection"]},
    ]}
    assert not score_scenario(review, expect).passed


def test_must_contain_matches_any_of_several_keywords():
    review = {"findings": [{"file": "api/users.py", "message": "unparameterized query"}]}
    expect = {"must_contain": [
        {"file_pattern": r"api/users\.py",
         "message_patterns": [r"sql\s*injection", r"unparameteri[sz]ed"]},
    ]}
    assert score_scenario(review, expect).passed


def test_must_contain_without_file_pattern_matches_on_message_only():
    review = {"findings": [{"message": "off-by-one in the loop bound"}]}
    expect = {"must_contain": [{"message_patterns": [r"off-by-one"]}]}
    assert score_scenario(review, expect).passed


# --------------------------------------------------------------------
# combined / multiple checks
# --------------------------------------------------------------------

def test_all_checks_must_pass_for_scenario_pass():
    review = {
        "severity": "high",
        "findings": [{"severity": "high", "file": "api/users.py",
                      "message": "SQL injection"}],
    }
    expect = {
        "approvable": False,
        "must_contain": [
            {"file_pattern": r"api/users\.py", "message_patterns": [r"sql\s*injection"]},
        ],
        "must_not_contain_patterns": [r"\bmissing\b"],
    }
    score = score_scenario(review, expect)
    assert score.passed
    assert score.true_positives == 1 and score.false_positives == 0


def test_empty_expect_yields_trivially_passing_scenario():
    score = score_scenario({"severity": "low", "findings": []}, {})
    assert score.passed
    assert score.checks == []


# --------------------------------------------------------------------
# aggregate: precision / recall / fp_rate
# --------------------------------------------------------------------

def _score(tp=0, fp=0, fn=0, neg=0, checks_pass=True):
    s = ScenarioScore(scenario="s")
    s.true_positives = tp
    s.false_positives = fp
    s.false_negatives = fn
    s.negatives = neg
    # one synthetic check reflecting overall pass/fail
    from tests.golden.scorer import CheckResult
    s.checks.append(CheckResult("synthetic", checks_pass))
    return s


def test_aggregate_precision_recall_fp_rate():
    agg = AggregateScore(scores=[
        _score(tp=2, fp=1, fn=0, neg=3),
        _score(tp=1, fp=0, fn=1, neg=2),
    ])
    # TP=3 FP=1 FN=1 neg=5
    assert agg.true_positives == 3
    assert agg.false_positives == 1
    assert agg.false_negatives == 1
    assert agg.negatives == 5
    assert agg.precision == pytest.approx(3 / 4)
    assert agg.recall == pytest.approx(3 / 4)
    assert agg.fp_rate == pytest.approx(1 / 5)


def test_aggregate_ratios_none_on_zero_denominator():
    agg = AggregateScore(scores=[_score(tp=0, fp=0, fn=0, neg=0)])
    assert agg.precision is None
    assert agg.recall is None
    assert agg.fp_rate is None
    # summary still renders with n/a
    line = agg.summary_line("max")
    assert "precision=n/a" in line and "fp_rate=n/a" in line
    assert "[golden:max]" in line


def test_aggregate_scenarios_passed_counts_only_passing():
    agg = AggregateScore(scores=[
        _score(checks_pass=True),
        _score(checks_pass=False),
        _score(checks_pass=True),
    ])
    assert agg.scenarios_passed == 2
    assert len(agg.scores) == 3


def test_summary_line_includes_effort_and_counts():
    agg = AggregateScore(scores=[_score(tp=1, fp=0, fn=0, neg=1)])
    line = agg.summary_line("high")
    assert "[golden:high]" in line
    assert "TP=1" in line and "FP=0" in line and "N-neg=1" in line
    assert "precision=1.00" in line
