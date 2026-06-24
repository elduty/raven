"""Golden-review eval harness — live runner (opt-in, costs tokens).

Replays the recorded scenarios under ``tests/golden/scenarios/`` against
the **live AI backend** via ``raven.reviewer.review_diff`` and scores each
result with the coarse, robust checks in ``tests/golden/scorer.py``
(verdict bucket, must-not-contain false-positive guard, must-contain
real-finding guard, coverage-gap). It then prints one A/B-comparable
summary line (precision / recall / FP-rate) at the configured reasoning
effort.

**Opt-in only — SKIPPED in normal CI.** Mirrors the gating of
``tests/test_prompt_injection_live.py``: the module-level ``skipif``
requires ``RAVEN_LIVE_AI_TESTS=1`` and the ``slow`` marker keeps it out
of default runs. With the env var unset, collection still works (cheap
imports only) but every test here skips, so no backend is invoked and no
tokens are spent.

Run it::

    RAVEN_LIVE_AI_TESTS=1 CLAUDE_CODE_OAUTH_TOKEN=<token> \\
        pytest -m slow tests/golden/test_golden_review_live.py -v -s

A/B reasoning effort (run twice, compare the two ``[golden:...]`` lines)::

    RAVEN_AI_EFFORT=high RAVEN_LIVE_AI_TESTS=1 ... pytest -m slow tests/golden/ -s
    RAVEN_AI_EFFORT=max  RAVEN_LIVE_AI_TESTS=1 ... pytest -m slow tests/golden/ -s

The runner reports the effort straight off ``reviewer.RAVEN_AI_EFFORT``
(the value review_diff actually uses, read from ``RAVEN_AI_EFFORT`` at
import), so the printed summary can never drift from what was run.

The SCORING logic itself is exercised offline (no backend) by
``tests/golden/test_scorer.py`` and ``tests/golden/test_corpus.py``,
which DO run in normal CI.
"""

from __future__ import annotations

import os

import pytest

from tests.golden.corpus import load_scenarios
from tests.golden.scorer import AggregateScore, score_scenario

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("RAVEN_LIVE_AI_TESTS"),
        reason="set RAVEN_LIVE_AI_TESTS=1 (with real backend creds) to opt in",
    ),
]


def _effort() -> str:
    """The reasoning effort review_diff will actually use this run.

    Read live from the reviewer module (which bound RAVEN_AI_EFFORT at
    import) rather than re-reading the env, so the reported effort and
    the effort used by review_diff can never disagree."""
    from raven import reviewer
    return reviewer.RAVEN_AI_EFFORT


# Shared across the parametrized per-scenario tests + the aggregate test
# within a single pytest process, so we score each scenario exactly once
# and the summary reflects the same runs the per-scenario tests asserted
# on. Keyed by scenario name.
_SCORE_CACHE: dict = {}


def _scored(scenario):
    """Run review_diff for ``scenario`` once and memoise the score."""
    if scenario.name in _SCORE_CACHE:
        return _SCORE_CACHE[scenario.name]
    # Import here (not at module top) so collection stays cheap and the
    # skip path never imports the heavy reviewer dispatch chain.
    from raven import reviewer
    from raven.reviewer import review_diff

    # A scenario may pin MAX_DIFF_LINES (e.g. the coverage-gap scenario,
    # which must trigger the oversized-chunk path off a FIXED cap so an
    # ambient MAX_DIFF_LINES override can't dispatch the oversized chunk
    # to a real backend and spend tokens). review_diff reads the module
    # global at call time, so patch it around the call and restore.
    if scenario.max_diff_lines is not None:
        original = reviewer.MAX_DIFF_LINES
        reviewer.MAX_DIFF_LINES = scenario.max_diff_lines
        try:
            review = review_diff(**scenario.review_kwargs())
        finally:
            reviewer.MAX_DIFF_LINES = original
    else:
        review = review_diff(**scenario.review_kwargs())

    score = score_scenario(review, scenario.expect, scenario=scenario.name)
    _SCORE_CACHE[scenario.name] = (review, score)
    return _SCORE_CACHE[scenario.name]


_SCENARIOS = load_scenarios()

# Fail loudly if the corpus is empty (dir moved/renamed/emptied). Without
# this, an empty parametrize would collect ZERO per-scenario tests and
# test_aggregate_summary would pass vacuously with scores==[] — a gate
# that silently green-lights on zero scenarios is worse than useless. This
# assert runs at import, so it also trips during normal (skipped)
# collection, surfacing a vanished corpus everywhere rather than only
# under the live opt-in.
assert _SCENARIOS, (
    "no golden scenarios loaded from tests/golden/scenarios/ — the corpus "
    "is empty or its directory moved; refusing to let the harness pass "
    "vacuously"
)


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s.name)
def test_scenario_passes_coarse_checks(scenario):
    """Each scenario's review must satisfy all of its coarse checks.

    Failures print every unmet check with a human-readable detail so a
    regression in the false-positive (must-not-contain) or false-negative
    (must-contain) direction is immediately legible.
    """
    review, score = _scored(scenario)
    if not score.passed:
        lines = [f"\nScenario {scenario.name!r} ({scenario.description}) failed:"]
        for c in score.failures:
            lines.append(f"  - [{c.name}] {c.detail}")
        lines.append(f"  review.severity={review.get('severity')!r} "
                     f"coverage_gap={review.get('coverage_gap')!r} "
                     f"findings={len(review.get('findings', []))}")
        pytest.fail("\n".join(lines))


def test_aggregate_summary():
    """Score every scenario, print the A/B summary line, and assert the
    corpus-level guards hold.

    Two aggregate guards (loose on purpose — coarse, not pinned):
      * **no regressions**: every scenario passed its own checks.
      * **FP-rate**: when the corpus carries negative assertions, the
        false-positive rate must be 0 — the harness exists to catch the
        confident-false-positive class, so any FP is a hard fail.

    Recall/precision are reported for A/B comparison but not hard-gated
    here (the per-scenario must_contain checks already gate the real-bug
    recall); the aggregate exists so two effort runs produce one
    comparable number each.
    """
    agg = AggregateScore()
    for scenario in _SCENARIOS:
        _review, score = _scored(scenario)
        agg.scores.append(score)

    # Belt-and-braces against a vacuous green: a zero-scenario aggregate
    # must never read as a pass (the module-load assert already guards the
    # corpus, but the gate's own summary test asserts it too).
    assert agg.scores, "no scenarios scored — the harness must not pass on an empty corpus"

    effort = _effort()
    # -s makes this visible; also pushed to the report via a section.
    print("\n" + agg.summary_line(effort))

    failed = [s.scenario for s in agg.scores if not s.passed]
    assert not failed, f"scenarios failed at effort={effort!r}: {failed}"

    if agg.negatives:
        assert agg.fp_rate == 0, (
            f"false-positive rate {agg.fp_rate} > 0 at effort={effort!r} — "
            f"the harness flags any confident false positive as a regression"
        )
