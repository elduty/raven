"""Coarse, robust scoring for golden-review eval scenarios.

This module is the **measurement core** of the golden-review harness and
is deliberately free of any AI/network dependency, so the scoring logic
itself is unit-tested offline (``tests/golden/test_scorer.py``) with
hand-written ``review`` dicts — no live backend, no tokens.

Philosophy (see ``tests/golden/README.md``): LLM output varies run to
run, so we never assert exact finding text or counts. We assert on a
handful of *robust properties* of a review dict (the shape returned by
``raven.reviewer.review_diff``):

* **verdict bucket** — is the review approvable / within a severity cap?
* **must-NOT-contain** — the false-positive guard: no finding message
  may match a forbidden regex (e.g. a tests-only diff must not claim the
  implementation is "missing").
* **must-contain** — the real-finding guard: at least one finding must
  land on the expected file *and* name the expected defect class (by
  keyword), without pinning the exact wording.
* **coverage_gap** — did the chunked path flag unreviewed code?

Each scenario's checks also contribute to aggregate **precision / recall
/ false-positive-rate**, computed from confusion-matrix counts:

* a satisfied ``must_contain`` clause is a **true positive** (TP) — the
  model surfaced a defect we know is real; an unsatisfied one is a
  **false negative** (FN) — it missed a known defect.
* a violated ``must_not_contain`` pattern is a **false positive** (FP) —
  the model raised a finding we know is spurious; the scenario also
  declares how many "negative" assertions it carries (``negatives``,
  defaulting to the number of must_not_contain patterns) so the
  FP-rate denominator (negatives evaluated) is well-defined.

``precision = TP / (TP + FP)``, ``recall = TP / (TP + FN)``,
``fp_rate = FP / negatives``. All guard against a zero denominator
(reported as ``None``) so an aggregate over scenarios that carry only
positive or only negative assertions still renders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Local import: keep the scorer dependency-light. SEVERITY_ORDER mirrors
# raven.reviewer; we import from there so a future severity tier stays in
# one place. The reviewer import is cheap (no network) and already used
# across the test suite.
from raven.reviewer import SEVERITY_ORDER


@dataclass
class CheckResult:
    """One coarse assertion's outcome: a label, pass/fail, and a human
    detail string for the failure report."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ScenarioScore:
    """Aggregated outcome of scoring one scenario's review against its
    expectation. ``passed`` is the AND of every check; the confusion-
    matrix counts feed the run-level precision/recall summary."""

    scenario: str
    checks: list[CheckResult] = field(default_factory=list)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    negatives: int = 0  # number of negative (must-not-contain) assertions evaluated

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _finding_messages(review: dict) -> list[str]:
    return [str(f.get("message", "")) for f in review.get("findings", []) or []]


def _clamp_severity(value) -> str:
    """Map any model-supplied severity to a known tier, defaulting to
    ``'low'`` for off-scheme values.

    LLMs routinely emit severities outside Raven's three-tier scheme —
    ``'critical'``, ``'blocker'``, ``'info'``, ``''`` — and the scorer
    indexes ``SEVERITY_ORDER`` with brackets in several places, which
    would ``KeyError`` and crash ``score_scenario`` mid-run (e.g. during
    a live effort A/B). Clamping here, at every point a model-supplied
    severity enters the ordering, keeps the scorer robust to that
    variance and mirrors ``reviewer._validate_review``'s own
    "unknown -> low" normalisation. Always returns a key present in
    ``SEVERITY_ORDER``, so callers may safely use ``[...]`` on the
    result.
    """
    s = str(value).lower()
    return s if s in SEVERITY_ORDER else "low"


def _max_finding_severity(review: dict) -> str:
    """Highest (clamped) severity across the review's findings ('low'
    when none).

    Note this is the finding-level max, distinct from the review's
    top-level ``severity`` field (which the model sets and the chunked
    path may floor for a coverage gap). The verdict-bucket checks below
    consider BOTH so a model that buries a high finding under a 'low'
    top-level severity still fails the approvable check. Each finding
    severity is clamped to a known tier so an off-scheme value
    ('critical', '') can never escape into a bracket-indexed lookup.
    """
    sevs = [_clamp_severity(f.get("severity", "low"))
            for f in review.get("findings", []) or []]
    if not sevs:
        return "low"
    return max(sevs, key=lambda s: SEVERITY_ORDER[s])


def _overall_severity(review: dict) -> str:
    """Effective severity = max(top-level severity, highest finding).

    Robust against the two ways a review can express seriousness: the
    top-level ``severity`` string and the per-finding severities. A
    clean approve requires both to be low. Both inputs are clamped to a
    known tier, so the return value is always a valid ``SEVERITY_ORDER``
    key (callers index it with brackets).
    """
    top = _clamp_severity(review.get("severity", "low"))
    finding_max = _max_finding_severity(review)
    return top if SEVERITY_ORDER[top] >= SEVERITY_ORDER[finding_max] else finding_max


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def score_scenario(review: dict, expect: dict, scenario: str = "?") -> ScenarioScore:
    """Score one review dict against a scenario's coarse ``expect`` spec.

    ``review`` is the dict returned by ``raven.reviewer.review_diff``.
    ``expect`` is the ``expect`` block of a scenario's expectation file.
    Returns a :class:`ScenarioScore`. Pure function — no I/O, no AI —
    so it is exhaustively unit-tested offline.

    Recognised ``expect`` keys (all optional; absent => that check is
    skipped, so a scenario only pays for the assertions it declares):

    * ``approvable`` (bool): when True, the effective severity must be
      ``low`` (no medium/high anywhere). When False, it must be medium+.
    * ``max_severity`` (str): effective severity must be <= this tier.
    * ``coverage_gap`` (bool): ``review["coverage_gap"]`` must equal it.
    * ``must_not_contain_patterns`` (list[str]): regexes; NO finding
      message may match any (false-positive guard). Each *pattern* is one
      negative assertion for the FP-rate denominator.
    * ``negatives`` (int): override the count of negative assertions
      (defaults to ``len(must_not_contain_patterns)``); lets a scenario
      weight its FP-rate contribution explicitly.
    * ``must_contain`` (list[dict]): each clause ``{file_pattern?,
      message_patterns:[...]}`` is satisfied when SOME finding matches
      both the file pattern (if given) and at least one message pattern.
      Satisfied => TP; unsatisfied => FN.
    """
    score = ScenarioScore(scenario=scenario)
    messages = _finding_messages(review)
    parse_error = bool(review.get("_parse_error"))

    # ---- verdict-bucket checks -------------------------------------
    if "approvable" in expect:
        eff = _overall_severity(review)
        approvable = SEVERITY_ORDER[eff] == SEVERITY_ORDER["low"]
        if expect["approvable"]:
            score.checks.append(CheckResult(
                "approvable",
                approvable and not parse_error,
                "" if (approvable and not parse_error)
                else f"expected approvable (low) but effective severity={eff!r}"
                     + (" + parse_error" if parse_error else ""),
            ))
        else:
            score.checks.append(CheckResult(
                "not_approvable",
                not approvable,
                "" if not approvable
                else "expected medium+ severity but review reads as a clean approve",
            ))

    if "max_severity" in expect:
        cap = str(expect["max_severity"]).lower()
        eff = _overall_severity(review)
        ok = SEVERITY_ORDER.get(eff, 0) <= SEVERITY_ORDER.get(cap, 0)
        score.checks.append(CheckResult(
            f"max_severity<={cap}",
            ok,
            "" if ok else f"effective severity={eff!r} exceeds cap {cap!r}",
        ))

    if "coverage_gap" in expect:
        want = bool(expect["coverage_gap"])
        got = bool(review.get("coverage_gap"))
        score.checks.append(CheckResult(
            "coverage_gap",
            got == want,
            "" if got == want else f"expected coverage_gap={want}, got {got}",
        ))

    # ---- must-NOT-contain (false-positive guard) -------------------
    mnc = expect.get("must_not_contain_patterns") or []
    # The number of negative assertions evaluated (FP-rate denominator).
    # Defaults to the pattern count; a scenario may override to weight
    # its contribution (e.g. a clean PR asserting "no findings at all").
    score.negatives = int(expect.get("negatives", len(mnc)))
    for pat in mnc:
        # Match each finding's message INDIVIDUALLY, never a concatenated
        # blob — that structurally prevents a pattern's anchor and its
        # negation from straddling two separate findings (or the gap
        # between them) and spuriously flipping a correct review to a
        # failure. The denominator above counts patterns, so a per-pattern
        # match is one false positive regardless of how many findings hit.
        offenders = [m for m in messages if re.search(pat, m, re.IGNORECASE)]
        passed = not offenders
        if not passed:
            score.false_positives += 1
        score.checks.append(CheckResult(
            f"must_not_contain:{pat}",
            passed,
            "" if passed else f"forbidden pattern matched finding(s): {offenders!r}",
        ))

    # ---- must-contain (real-finding guard) -------------------------
    for clause in expect.get("must_contain") or []:
        file_pat = clause.get("file_pattern")
        msg_pats = clause.get("message_patterns") or []
        label = clause.get("label") or (file_pat or (msg_pats[0] if msg_pats else "?"))
        matched = False
        for f in review.get("findings", []) or []:
            ffile = str(f.get("file", ""))
            fmsg = str(f.get("message", ""))
            if file_pat and not re.search(file_pat, ffile, re.IGNORECASE):
                continue
            if msg_pats and not _matches_any(msg_pats, fmsg):
                continue
            matched = True
            break
        if matched:
            score.true_positives += 1
        else:
            score.false_negatives += 1
        score.checks.append(CheckResult(
            f"must_contain:{label}",
            matched,
            "" if matched
            else f"no finding matched file~{file_pat!r} & message~{msg_pats!r}; "
                 f"findings={review.get('findings', [])!r}",
        ))

    return score


@dataclass
class AggregateScore:
    """Run-level rollup across every scored scenario."""

    scores: list[ScenarioScore] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(s.true_positives for s in self.scores)

    @property
    def false_positives(self) -> int:
        return sum(s.false_positives for s in self.scores)

    @property
    def false_negatives(self) -> int:
        return sum(s.false_negatives for s in self.scores)

    @property
    def negatives(self) -> int:
        return sum(s.negatives for s in self.scores)

    @property
    def scenarios_passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    def _ratio(self, num: int, den: int):
        """Return num/den, or None when the denominator is zero (so an
        aggregate over only-positive or only-negative corpora still
        renders without a ZeroDivisionError)."""
        return (num / den) if den else None

    @property
    def precision(self):
        return self._ratio(self.true_positives,
                           self.true_positives + self.false_positives)

    @property
    def recall(self):
        return self._ratio(self.true_positives,
                           self.true_positives + self.false_negatives)

    @property
    def fp_rate(self):
        return self._ratio(self.false_positives, self.negatives)

    def summary_line(self, effort: str = "?") -> str:
        """One-line, A/B-comparable summary (printed per effort run).

        Renders ``None`` ratios as ``n/a`` so an aggregate that carries
        only positive or only negative assertions still prints cleanly.
        """
        def _fmt(x):
            return "n/a" if x is None else f"{x:.2f}"
        return (
            f"[golden:{effort}] scenarios {self.scenarios_passed}/{len(self.scores)} passed | "
            f"precision={_fmt(self.precision)} recall={_fmt(self.recall)} "
            f"fp_rate={_fmt(self.fp_rate)} | "
            f"TP={self.true_positives} FP={self.false_positives} "
            f"FN={self.false_negatives} N-neg={self.negatives}"
        )
