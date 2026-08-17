"""Unit tests for raven/severity.py — the severity vocabulary value object."""

import json as _json

import pytest

from raven.severity import (
    BLOCKING,
    SeverityScale,
    default_scale,
    from_json,
    InvalidScale,
    render_severity_block,
)


def _scale(**kw):
    """A 5-tier scale for tests. Ranks deliberately non-contiguous."""
    base = dict(
        ranks={"nit": 10, "low": 20, "medium": 30, "high": 40, "critical": 50},
        blocks_at_or_above="medium",
        descriptions={"critical": "Exploitable or unrecoverable."},
    )
    base.update(kw)
    return SeverityScale(**base)


class TestOrdering:
    def test_most_severe_is_highest_rank(self):
        assert _scale().most_severe == "critical"

    def test_least_severe_is_lowest_rank(self):
        assert _scale().least_severe == "nit"

    def test_ordered_is_most_severe_first(self):
        assert _scale().ordered() == ["critical", "high", "medium", "low", "nit"]


class TestUnknownNamesFailClosed:
    """Model-emitted names Raven does not know must land on the MOST severe
    tier. This is shipped behaviour (#211) — a regression here silently
    re-arms the merge gate."""

    def test_rank_of_unknown_is_max(self):
        s = _scale()
        assert s.rank("blocker") == s.rank("critical")

    def test_normalize_unknown_to_most_severe(self):
        assert _scale().normalize("blocker") == "critical"

    def test_normalize_strips_and_lowercases(self):
        assert _scale().normalize("  MEDIUM  ") == "medium"

    def test_is_known_rejects_unknown(self):
        s = _scale()
        assert s.is_known("medium") is True
        assert s.is_known("blocker") is False


class TestBlocking:
    def test_at_threshold_blocks(self):
        assert _scale().blocks("medium") is True

    def test_above_threshold_blocks(self):
        assert _scale().blocks("critical") is True

    def test_below_threshold_does_not_block(self):
        assert _scale().blocks("low") is False
        assert _scale().blocks("nit") is False

    def test_unknown_name_blocks(self):
        """Fail closed: an unrecognised name is treated as most severe."""
        assert _scale().blocks("blocker") is True

    def test_none_threshold_blocks_nothing(self):
        """blocks_at_or_above=None is the 'nothing blocks' configuration."""
        s = _scale(blocks_at_or_above=None)
        assert s.blocks("critical") is False


class TestEmoji:
    """Colour is POSITION-based, not gate-based: top tier red, bottom tier
    yellow, everything in between orange — regardless of blocks_at_or_above.

    (Task 4 correction: Task 1 shipped emoji() keyed off blocks(), which
    made colour vary with REVIEW_APPROVE_MAX_SEVERITY. Verified divergence
    against today's fixed high=red/medium=orange/low=yellow: at threshold
    medium or high, gate-based rendered medium as yellow instead of
    orange. Position is stable and reproduces today's colours for every
    threshold value — see test_colour_independent_of_blocking_threshold.)
    """

    def test_most_severe_is_red(self):
        assert _scale().emoji("critical") == "🔴"

    def test_least_severe_is_yellow(self):
        assert _scale().emoji("nit") == "🟡"

    def test_everything_between_is_orange(self):
        # "low" sits BELOW the blocking threshold (blocks_at_or_above=
        # "medium" on this fixture) yet is neither the top nor the bottom
        # tier, so it is orange — this is exactly where the old gate-based
        # emoji() diverged (it rendered "low" as yellow because it didn't
        # block).
        assert _scale().emoji("low") == "🟠"
        assert _scale().emoji("medium") == "🟠"
        assert _scale().emoji("high") == "🟠"

    def test_unknown_name_is_red(self):
        """Unknown names normalize to most_severe (fail closed), so they
        render like the top tier."""
        assert _scale().emoji("blocker") == "🔴"

    def test_colour_independent_of_blocking_threshold(self):
        """Changing blocks_at_or_above must not change any colour — that
        was the Task 1 bug. Position in ``ranks`` is the only input."""
        gate_at_low = _scale(blocks_at_or_above="low")
        gate_at_critical = _scale(blocks_at_or_above="critical")
        gate_none = _scale(blocks_at_or_above=None)
        for name in ("nit", "low", "medium", "high", "critical"):
            assert (gate_at_low.emoji(name)
                    == gate_at_critical.emoji(name)
                    == gate_none.emoji(name))


class TestFingerprint:
    def test_stable_across_key_order(self):
        a = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b", descriptions={})
        b = SeverityScale(ranks={"b": 2, "a": 1}, blocks_at_or_above="b", descriptions={})
        assert a.fingerprint() == b.fingerprint()

    def test_changes_when_threshold_changes(self):
        a = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b", descriptions={})
        b = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="a", descriptions={})
        assert a.fingerprint() != b.fingerprint()

    def test_changes_when_descriptions_change(self):
        """Descriptions reach the prompt, so they are part of review config."""
        a = SeverityScale(ranks={"a": 1}, blocks_at_or_above=None, descriptions={})
        b = SeverityScale(ranks={"a": 1}, blocks_at_or_above=None, descriptions={"a": "x"})
        assert a.fingerprint() != b.fingerprint()


class TestDefaultScale:
    def test_is_todays_three_tiers(self):
        s = default_scale()
        assert s.ordered() == ["high", "medium", "low"]

    def test_blocks_above_approve_threshold(self, monkeypatch):
        monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", "low")
        assert default_scale().blocks_at_or_above == "medium"

    def test_threshold_high_blocks_nothing(self, monkeypatch):
        """REVIEW_APPROVE_MAX_SEVERITY=high means approve everything —
        there is no tier above it."""
        monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", "high")
        assert default_scale().blocks_at_or_above is None

    def test_unknown_threshold_is_strictest_not_loosest(self, monkeypatch):
        """An operator typo must NOT approve everything. Today an unknown
        threshold gets rank 0 via severity_gte, i.e. the strictest setting;
        that must be preserved."""
        monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", "typo")
        assert default_scale().blocks_at_or_above == "medium"

    def test_read_at_call_time(self, monkeypatch):
        monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", "medium")
        assert default_scale().blocks_at_or_above == "high"
        monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", "low")
        assert default_scale().blocks_at_or_above == "medium"

    def test_emoji_colours_identical_across_approve_threshold(self, monkeypatch):
        """Position-based emoji must not depend on REVIEW_APPROVE_MAX_SEVERITY.

        Verified divergence against gate-based emoji() (Task 1's
        implementation): at threshold medium or high, medium rendered
        yellow instead of orange. Position reproduces today's fixed
        colours no matter what the operator sets the threshold to."""
        expected = {"high": "🔴", "medium": "🟠", "low": "🟡"}
        for threshold in ("low", "medium", "high"):
            monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", threshold)
            s = default_scale()
            assert {n: s.emoji(n) for n in ("high", "medium", "low")} == expected


class TestDefaultScaleDescriptions:
    """default_scale() must still carry the category-anchoring guidance that
    prompts/review.md used to hand-write in its '## Severity Definitions'
    section. That section was replaced by {{severity_scale}} (rendered via
    render_severity_block), but default_scale() shipped with
    descriptions={} — so the ~100% of repos with no severities.json lost
    all category guidance from their review prompt, invisibly (no test
    caught it because nothing asserted on rendered content)."""

    def test_rendered_default_block_restores_category_guidance(self):
        rendered = render_severity_block(default_scale())
        expected_phrases = [
            "injection",
            "auth bypass",
            "exposed secrets",
            "insecure deserialization",
            "path traversal",
            "data loss",
            "race condition",
            "resource leak",
            "missing error handling",
            "n+1",
            "missing input validation",
            "hard-coded credentials",
            "minor bugs with limited blast radius",
            "missing tests for non-critical behaviour",
        ]
        lowered = rendered.lower()
        missing = [p for p in expected_phrases if p not in lowered]
        assert not missing, f"default severity scale prompt lost guidance: {missing}"


class TestRenderSeverityBlockUnknownNameClause:
    """The trailing 'unknown name' instruction must not claim a block that
    the scale doesn't actually apply. For an advisory-only scale
    (blocks_at_or_above=None) the per-tier loop already suppresses the
    block marker; the tail sentence must match."""

    def test_advisory_only_scale_does_not_claim_unknown_blocks(self):
        s = SeverityScale(ranks={"nit": 10, "bug": 20}, blocks_at_or_above=None)
        rendered = render_severity_block(s)
        assert "and blocks the merge" not in rendered
        assert "treated as `bug`" in rendered

    def test_blocking_scale_still_states_unknown_blocks(self):
        s = SeverityScale(ranks={"nit": 10, "bug": 20}, blocks_at_or_above="bug")
        rendered = render_severity_block(s)
        assert "treated as `bug` and blocks the merge" in rendered


VALID = {
    "severities": {"nit": 10, "low": 20, "medium": 30, "high": 40, "critical": 50},
    "blocks_at_or_above": "medium",
    "descriptions": {"critical": "Exploitable or unrecoverable."},
}


class TestFromJson:
    def test_parses_a_valid_file(self):
        s = from_json(_json.dumps(VALID))
        assert s.ordered() == ["critical", "high", "medium", "low", "nit"]
        assert s.blocks_at_or_above == "medium"
        assert s.descriptions["critical"].startswith("Exploitable")

    def test_descriptions_optional(self):
        body = {k: v for k, v in VALID.items() if k != "descriptions"}
        assert from_json(_json.dumps(body)).descriptions == {}

    def test_descriptions_explicit_null_is_absent(self):
        body = {**VALID, "descriptions": None}
        assert from_json(_json.dumps(body)).descriptions == {}

    @pytest.mark.parametrize("bad_descriptions", [[], [1, 2], "", 0, False, "text"])
    def test_descriptions_wrong_shape_rejected(self, bad_descriptions):
        """A falsy-but-present 'descriptions' (``[]``, ``""``, ``0``,
        ``false``) must be rejected exactly like a truthy wrong shape
        (``[1, 2]``, ``"text"``) — the old ``body.get(...) or {}`` silently
        treated falsy malformed values as absent."""
        body = {**VALID, "descriptions": bad_descriptions}
        with pytest.raises(InvalidScale) as e:
            from_json(_json.dumps(body))
        assert "must be an object" in str(e.value)

    def test_omitted_threshold_blocks_nothing(self):
        body = {k: v for k, v in VALID.items() if k != "blocks_at_or_above"}
        assert from_json(_json.dumps(body)).blocks_at_or_above is None

    @pytest.mark.parametrize("body,fragment", [
        ("not json at all", "not valid JSON"),
        (_json.dumps([1, 2]), "must be a JSON object"),
        (_json.dumps({"severities": {}}), "at least two"),
        (_json.dumps({"severities": {"only": 1}}), "at least two"),
        (_json.dumps({"severities": {"a": 1, "b": 1}}), "duplicate rank"),
        (_json.dumps({"severities": {"a": 1, "b": "2"}}), "integer"),
        (_json.dumps({"severities": {"High": 1, "high": 2}}), "duplicate tier name"),
        (_json.dumps({"severities": {"a": 1, " a ": 2}}), "duplicate tier name"),
        (_json.dumps({"severities": {"a": 1, "b": 2}, "blocks_at_or_above": "z"}), "not one of"),
        (_json.dumps({"severities": {"a": 1, "B!": 2}}), "invalid tier name"),
        (_json.dumps({"severities": {"a": 1, "9x": 2}}), "invalid tier name"),
        (_json.dumps({"severities": {"a": 1, "x" * 33: 2}}), "invalid tier name"),
    ])
    def test_rejections(self, body, fragment):
        with pytest.raises(InvalidScale) as e:
            from_json(body)
        assert fragment in str(e.value)

    def test_duplicate_names_impossible_in_json(self):
        """JSON objects cannot hold duplicate keys — later wins. Documented
        so the spec's 'duplicate names' rule is not mistaken for missing."""
        s = from_json(_json.dumps({"severities": {"a": 1, "b": 2}}))
        assert set(s.ranks) == {"a", "b"}

    def test_rejects_tier_named_blocking(self):
        """'blocking' is the reserved notification-threshold sentinel
        (severity.BLOCKING) — documented as the one min_severity value that
        means the same thing in every repo's vocabulary. A repo defining a
        tier literally named 'blocking' makes that sentinel ambiguous, so
        it must be rejected at parse time rather than left to precedence."""
        body = {"severities": {BLOCKING: 10, "bad": 20}}
        with pytest.raises(InvalidScale) as e:
            from_json(_json.dumps(body))
        assert "reserved" in str(e.value)

    def test_non_reserved_names_still_parse(self):
        """Sanity check: rejecting 'blocking' must not collaterally reject
        an otherwise-valid scale that doesn't use that name."""
        body = {"severities": {"nit": 10, "bad": 20}}
        s = from_json(_json.dumps(body))
        assert set(s.ranks) == {"nit", "bad"}


class TestRepoAuthoredTextIsBounded:
    """severities.json content renders into the review prompt, and the
    rendered block is substituted into the TEMPLATE itself — not into a
    tagged untrusted/policy block. It is therefore the one repo-authored
    input entering trusted prompt space unwrapped. Same commit+merge gate
    as CLAUDE.md so there is no privilege escalation, but the size is
    unbounded: an oversized descriptions map inflates every review prompt
    for that repo, crowding out the diff it is supposed to review
    (audit 2026-08-14 LOW).
    """

    def test_rejects_oversized_description(self):
        with pytest.raises(InvalidScale, match="description"):
            from_json(_json.dumps({
                "severities": {"nit": 10, "bug": 20},
                "descriptions": {"bug": "x" * 5000},
            }))

    def test_rejects_absurd_tier_count(self):
        with pytest.raises(InvalidScale, match="tier"):
            from_json(_json.dumps({
                "severities": {f"t{i}": i for i in range(64)},
            }))

    def test_reasonable_descriptions_still_accepted(self):
        scale = from_json(_json.dumps({
            "severities": {"nit": 10, "bug": 20},
            "descriptions": {"bug": "A real defect. " * 20},
        }))
        assert "A real defect." in scale.descriptions["bug"]
