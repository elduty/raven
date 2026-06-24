"""Offline tests for the golden scenario corpus + its patterns.

Runs in **normal CI** (no live AI). Two jobs:

1. Validate the loader (``tests/golden/corpus.py``) — structure, defaults,
   file_contents resolution, error handling.
2. Validate the checked-in scenarios' *expectation patterns* against
   hand-written reviews — proving each scenario's must-contain matches
   the defect class it targets and its must-not-contain catches the
   confident-false-positive phrasing WITHOUT firing on benign text.
   These guard against regex-authoring mistakes (a too-loose absence
   pattern that flags every review, or a real-bug pattern that never
   matches) landing silently in the corpus, since the live run that
   would otherwise catch them is opt-in.
"""

from __future__ import annotations

import pytest

from tests.golden.corpus import load_scenarios, _load_one
from tests.golden.scorer import score_scenario

SCENARIOS = load_scenarios()
BY_NAME = {s.name: s for s in SCENARIOS}


# --------------------------------------------------------------------
# loader structure
# --------------------------------------------------------------------

def test_corpus_is_not_empty():
    # CI-visible guard: a moved/renamed/emptied scenarios dir must fail
    # here (in the normal suite) rather than let the opt-in live harness
    # green-light vacuously on zero scenarios.
    assert SCENARIOS, "no golden scenarios loaded from tests/golden/scenarios/"


def test_expected_scenarios_present():
    names = set(BY_NAME)
    assert {"clean_small", "tests_only_no_phantom_absence", "real_bug"} <= names


def test_every_scenario_has_diff_and_expect():
    for s in SCENARIOS:
        assert s.diff.strip(), f"{s.name}: empty diff"
        assert isinstance(s.expect, dict) and s.expect, f"{s.name}: empty expect"
        assert s.repo_name, f"{s.name}: empty repo_name"


def test_review_kwargs_minimal_by_default():
    s = BY_NAME["real_bug"]
    kwargs = s.review_kwargs()
    assert set(kwargs) == {"diff", "repo_name"}


def test_review_kwargs_includes_incremental_context():
    s = BY_NAME["tests_only_no_phantom_absence"]
    kwargs = s.review_kwargs()
    assert kwargs["is_incremental"] is True
    assert kwargs["unchanged_files"] == ["billing/discount.py"]
    assert "billing/discount.py" in kwargs["file_contents"]
    assert "apply_discount" in kwargs["file_contents"]["billing/discount.py"]


def test_default_repo_name_when_unset(tmp_path):
    d = tmp_path / "nameless"
    d.mkdir()
    (d / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (d / "expectation.json").write_text('{"expect": {"approvable": true}}', encoding="utf-8")
    s = _load_one(d)
    assert s.repo_name == "golden/nameless"


def test_missing_expect_raises(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (d / "expectation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_one(d)


def test_missing_file_contents_ref_raises(tmp_path):
    d = tmp_path / "bad2"
    d.mkdir()
    (d / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (d / "expectation.json").write_text(
        '{"file_contents": {"a.py": "files/nope.py"}, "expect": {}}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _load_one(d)


def test_load_scenarios_skips_dir_without_diff(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "README.md").write_text("hi", encoding="utf-8")
    real = tmp_path / "real"
    real.mkdir()
    (real / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (real / "expectation.json").write_text('{"expect": {}}', encoding="utf-8")
    loaded = load_scenarios(tmp_path)
    assert [s.name for s in loaded] == ["real"]


def test_max_diff_lines_defaults_to_none():
    # Scenarios that don't pin a cap leave the ambient value alone.
    assert BY_NAME["real_bug"].max_diff_lines is None


def test_max_diff_lines_loaded_when_set():
    assert BY_NAME["oversized_chunk"].max_diff_lines == 50


@pytest.mark.parametrize("bad", [0, -5, 3.5, True, "50"])
def test_max_diff_lines_rejects_non_positive_int(tmp_path, bad):
    import json as _json
    d = tmp_path / "badcap"
    d.mkdir()
    (d / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (d / "expectation.json").write_text(
        _json.dumps({"max_diff_lines": bad, "expect": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_one(d)


# --------------------------------------------------------------------
# real_bug: the must_contain pattern matches an SQLi finding, and the
# scenario passes a representative "good" review while failing a "missed
# it" review.
# --------------------------------------------------------------------

def test_real_bug_passes_when_sqli_flagged():
    s = BY_NAME["real_bug"]
    good_review = {
        "severity": "high",
        "findings": [
            {"severity": "high", "file": "api/users.py",
             "message": "SQL injection: query built via string concatenation of an "
                        "untrusted name; use parameterised queries."},
        ],
    }
    assert score_scenario(good_review, s.expect, s.name).passed


def test_real_bug_fails_when_sqli_missed():
    s = BY_NAME["real_bug"]
    miss_review = {
        "severity": "high",
        "findings": [
            {"severity": "high", "file": "api/users.py",
             "message": "Consider adding a docstring to delete_user."},
        ],
    }
    score = score_scenario(miss_review, s.expect, s.name)
    assert not score.passed
    assert score.false_negatives == 1


def test_real_bug_marks_clean_approve_as_failure():
    s = BY_NAME["real_bug"]
    clean = {"severity": "low", "findings": []}
    assert not score_scenario(clean, s.expect, s.name).passed


# --------------------------------------------------------------------
# tests_only_no_phantom_absence: the must_not_contain catches the
# phantom-absence phrasing but does NOT fire on a sane tests-only review.
# --------------------------------------------------------------------

def test_tests_only_passes_on_benign_review():
    s = BY_NAME["tests_only_no_phantom_absence"]
    benign = {
        "severity": "low",
        "findings": [
            {"severity": "low", "file": "tests/test_discount.py",
             "message": "Good coverage; consider a test for a fractional percent."},
        ],
    }
    score = score_scenario(benign, s.expect, s.name)
    assert score.passed
    assert score.false_positives == 0


@pytest.mark.parametrize("bad_message", [
    "The apply_discount implementation is missing from this PR.",
    "This PR adds tests but the implementation does not exist.",
    "The function is not implemented anywhere in the diff.",
    "billing/discount.py module is absent — only tests were added.",
])
def test_tests_only_catches_phantom_absence_phrasings(bad_message):
    s = BY_NAME["tests_only_no_phantom_absence"]
    bad = {"severity": "high", "findings": [{"severity": "high", "message": bad_message}]}
    score = score_scenario(bad, s.expect, s.name)
    assert not score.passed, f"phantom-absence phrasing slipped through: {bad_message!r}"
    assert score.false_positives >= 1


@pytest.mark.parametrize("ok_message", [
    "Tests look correct and cover the boundary cases.",
    "Consider asserting the exception message, not just the type.",
    "The rounding test documents the expected behaviour well.",
    # Mentions BOTH the implementation AND 'missing' — but in unrelated
    # clauses separated by a sentence boundary / newline. The bounded gap
    # (`[^.\n]{0,40}`, not the old unbounded `[^.]*`) must NOT bridge
    # them, or every such review would spuriously fail the FP guard.
    "The implementation in billing/discount.py is sound. A docstring is "
    "missing on the test module, though.",
    "The apply_discount implementation is covered.\nA negative-percent "
    "edge case is missing from the suite.",
])
def test_tests_only_does_not_fire_on_ordinary_test_feedback(ok_message):
    # Guard the OTHER direction: ordinary tests-review prose mentioning
    # neither absence-of-implementation (nor bridging it across clauses)
    # must not trip the FP guard.
    s = BY_NAME["tests_only_no_phantom_absence"]
    review = {"severity": "low", "findings": [{"severity": "low", "message": ok_message}]}
    score = score_scenario(review, s.expect, s.name)
    assert score.passed, f"benign prose tripped the FP guard: {ok_message!r}"
    assert score.false_positives == 0


# --------------------------------------------------------------------
# clean_small: benign review passes; an over-eager invented-defect review
# (severity bump + absence phrasing) fails.
# --------------------------------------------------------------------

def test_clean_small_passes_on_low_no_findings():
    s = BY_NAME["clean_small"]
    assert score_scenario({"severity": "low", "findings": []}, s.expect, s.name).passed


def test_clean_small_fails_when_severity_inflated():
    # A behaviour-preserving refactor scored medium must fail the
    # approvable / max_severity:low buckets regardless of the message.
    s = BY_NAME["clean_small"]
    review = {"severity": "medium",
              "findings": [{"severity": "medium", "message": "questionable change"}]}
    score = score_scenario(review, s.expect, s.name)
    assert not score.passed
    assert any(c.name in ("approvable", "max_severity<=low") and not c.passed
               for c in score.checks)


def test_clean_small_does_not_fire_fp_guard_on_benign_none_mention():
    # 'None'/'null' are everyday review vocabulary; after tightening the
    # clean_small FP pattern they must not register as false positives.
    s = BY_NAME["clean_small"]
    review = {"severity": "low", "findings": [
        {"severity": "low", "message": "correctly handles None and empty-string input"},
    ]}
    score = score_scenario(review, s.expect, s.name)
    assert score.passed
    assert score.false_positives == 0


# --------------------------------------------------------------------
# oversized_chunk: coverage_gap expectation behaves.
# --------------------------------------------------------------------

def test_oversized_chunk_expects_gap_and_not_approvable():
    s = BY_NAME["oversized_chunk"]
    assert s.expect.get("coverage_gap") is True
    assert s.expect.get("approvable") is False
    # A review that flags the gap + non-approvable severity passes.
    gap_review = {"severity": "high", "findings": [], "coverage_gap": True}
    assert score_scenario(gap_review, s.expect, s.name).passed
    # One that silently approves does not.
    silent = {"severity": "low", "findings": [], "coverage_gap": False}
    assert not score_scenario(silent, s.expect, s.name).passed
