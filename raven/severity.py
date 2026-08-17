"""severity.py — the single definition of a repository's severity vocabulary.

Everything that needs to know what the tiers are called, how they order,
and which one blocks a merge reads a ``SeverityScale``. Nothing hardcodes
tier names: not the gate, not the emoji, not the review prompt.

This module is a leaf. It must not import ``reviewer``, ``server``, or
``notifier`` — they all import it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

# Notification-threshold sentinel: "notify when the review blocks the
# merge". The only threshold that means the same thing in every repo's
# vocabulary, so it is the recommended value for multi-repo deployments.
BLOCKING = "blocking"


@dataclass(frozen=True)
class SeverityScale:
    """An ordered severity vocabulary plus the tier at which merges block.

    ``ranks`` maps tier name -> rank; higher is more severe. Ranks need
    not be contiguous — gaps let a tier be inserted later without
    renumbering.

    ``blocks_at_or_above`` names the least-severe blocking tier, or is
    ``None`` for "nothing blocks on severity alone" (the
    ``REVIEW_APPROVE_MAX_SEVERITY=high`` configuration, where the merge is
    gated only by the other guards).

    Frozen by convention as well as by dataclass: the dicts are not deep
    copied, so callers must not mutate what they pass in. Build instances
    through ``default_scale()`` or ``from_json()``, which construct fresh
    dicts.
    """

    ranks: dict[str, int]
    blocks_at_or_above: str | None = None
    descriptions: dict[str, str] = field(default_factory=dict)

    # ── vocabulary ────────────────────────────────────────────────── #

    @property
    def most_severe(self) -> str:
        return max(self.ranks, key=lambda n: self.ranks[n])

    @property
    def least_severe(self) -> str:
        return min(self.ranks, key=lambda n: self.ranks[n])

    def ordered(self) -> list[str]:
        """Tier names, most severe first — prompt and display order."""
        return sorted(self.ranks, key=lambda n: self.ranks[n], reverse=True)

    def is_known(self, name: str) -> bool:
        return self._clean(name) in self.ranks

    def normalize(self, name: str) -> str:
        """Map a MODEL-EMITTED severity onto this scale.

        Fail CLOSED: a name Raven doesn't know means the model didn't
        honour the output contract, and the safe reading of "I don't know
        how bad this is" is "assume the worst" — the alternative silently
        approves. See reviewer._validate_review and PR #211.

        Do NOT use this for operator-configured thresholds; there an
        unknown name must resolve to the *strictest* reading, not the
        most severe tier.
        """
        clean = self._clean(name)
        return clean if clean in self.ranks else self.most_severe

    def rank(self, name: str) -> int:
        """Rank of a MODEL-EMITTED severity; unknown -> most severe."""
        return self.ranks[self.normalize(name)]

    # ── the gate ──────────────────────────────────────────────────── #

    def blocks(self, name: str) -> bool:
        """Does a review at this severity block the merge?"""
        if self.blocks_at_or_above is None:
            return False
        return self.rank(name) >= self.ranks[self.blocks_at_or_above]

    # ── presentation ──────────────────────────────────────────────── #

    def emoji(self, name: str) -> str:
        """Colour by position in the scale, so any tier count works and the
        result never depends on operator config: top tier red, bottom tier
        yellow, anything between orange.

        Deliberately NOT gate-based. Keying colour off blocks() would make
        emoji vary with REVIEW_APPROVE_MAX_SEVERITY — with the default
        three-tier scale and the threshold at medium or high, medium renders
        yellow where today it renders orange. Position is stable and
        reproduces today's colours for every threshold value.
        """
        norm = self.normalize(name)
        if norm == self.most_severe:
            return "🔴"
        if norm == self.least_severe:
            return "🟡"
        return "🟠"

    # ── identity ──────────────────────────────────────────────────── #

    def fingerprint(self) -> str:
        """Stable hash of everything that affects a review's outcome or
        prompt. Folded into the findings-cache key so a scale change
        invalidates cached verdicts computed under the old vocabulary."""
        payload = json.dumps(
            {
                "ranks": self.ranks,
                "blocks_at_or_above": self.blocks_at_or_above,
                "descriptions": self.descriptions,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _clean(name: str) -> str:
        return str(name or "").strip().lower()


# Today's built-in vocabulary. Ranks are 0/1/2, byte-identical to the
# pre-SeverityScale hand-written SEVERITY_ORDER in reviewer.py — several
# call sites (severity_gte and its callers in reviewer.py/server.py) read
# SEVERITY_ORDER.get(name, 0) directly, and that 0 default must keep
# TYING with "low" rather than sitting below it, or an unknown/typo'd
# severity would compare as stricter than intended (see severity_gte's
# "unknown -> strictest" contract). The gap-spaced-by-10 convention is for
# repo-authored severities.json files (a starting point for inserting a
# tier later) — not for this built-in scale, which must reproduce the old
# numbering exactly.
_DEFAULT_RANKS = {"low": 0, "medium": 1, "high": 2}

# Recovered verbatim (condensed to one description string per tier) from
# prompts/review.md's hand-written "## Severity Definitions" section, as it
# stood immediately before e17346f replaced it with the {{severity_scale}}
# placeholder. default_scale() is what actually renders into the prompt for
# every repo without a severities.json — i.e. effectively all of them — so
# shipping it with descriptions={} silently dropped this category-anchoring
# guidance from the default review path. This is recovery, not authorship:
# do not add or remove categories here without updating this comment.
_DEFAULT_DESCRIPTIONS = {
    "high": (
        "- Security vulnerabilities: injection, auth bypass, exposed secrets, "
        "insecure deserialization, path traversal\n"
        "- Data loss or corruption: missing transactions, silent error "
        "swallowing, destructive operations without guards\n"
        "- Logic errors that will cause incorrect behavior in production\n"
        "- Race conditions, deadlocks, or undefined behavior under concurrency\n"
        "- Breaking changes to public APIs or data schemas without migration"
    ),
    "medium": (
        "- Resource leaks: unclosed connections, files, or handles\n"
        "- Missing error handling on operations that can fail (network, "
        "disk, external services)\n"
        "- N+1 queries or obvious performance problems at scale\n"
        "- Incorrect or missing input validation\n"
        "- Hard-coded credentials, IPs, or environment-specific values\n"
        "- Missing or inadequate tests for changed behavior"
    ),
    "low": (
        "- Minor bugs with limited blast radius (edge cases, rare code paths)\n"
        "- Missing tests for non-critical behaviour\n"
        "- Small inefficiencies with real measurable impact"
    ),
}


def default_scale() -> SeverityScale:
    """The built-in three-tier scale, with the blocking tier derived from
    ``REVIEW_APPROVE_MAX_SEVERITY``.

    Read at call time, not import time, so it tracks operator config the
    same way ``server.py``'s approve decision does.

    ``REVIEW_APPROVE_MAX_SEVERITY`` names the highest severity that still
    APPROVES, so the blocking tier is the one immediately above it. A
    threshold of the top tier leaves nothing above -> ``None``.

    An unrecognised value resolves to the LEAST severe tier, i.e. the
    strictest gate. That preserves today's behaviour exactly: the current
    ``severity_gte`` gives an unknown threshold rank 0, which approves
    only at the lowest tier. Failing to the most severe tier here would
    silently turn an operator typo into approve-everything.
    """
    ordered_low_to_high = sorted(_DEFAULT_RANKS, key=lambda n: _DEFAULT_RANKS[n])
    raw = os.environ.get("REVIEW_APPROVE_MAX_SEVERITY", "low").strip().lower()
    approve_max = raw if raw in _DEFAULT_RANKS else ordered_low_to_high[0]

    idx = ordered_low_to_high.index(approve_max)
    blocks_at = (ordered_low_to_high[idx + 1]
                 if idx + 1 < len(ordered_low_to_high) else None)

    return SeverityScale(
        ranks=dict(_DEFAULT_RANKS),
        blocks_at_or_above=blocks_at,
        descriptions=dict(_DEFAULT_DESCRIPTIONS),
    )


# ── repo-configured scale (severities.json) ─────────────────────────── #

# Tier names become Prometheus label values and prompt text, so the charset
# is deliberately strict: lowercase start, then letters/digits/underscore/
# hyphen, capped at 32 chars total. A quote, brace, or newline here could
# break /metrics exposition for the whole endpoint.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# render_severity_block()'s output is substituted into the review-prompt
# TEMPLATE itself (see reviewer._apply_scale_to_template), not into a
# tagged untrusted/policy block — so tier names and descriptions are the
# one repo-authored input that reaches trusted prompt space unwrapped.
# That is the same commit+merge gate as CLAUDE.md, so it is not a
# privilege escalation; the risk being bounded here is size. Every review
# of the repo pays for this block, and an oversized one crowds out the
# diff it is supposed to be reviewing. Limits are generous enough that no
# honest scale hits them.
_MAX_TIERS = 24
_MAX_DESCRIPTION_CHARS = 2000


class InvalidScale(ValueError):
    """A severities.json that failed validation.

    The message is operator-facing: it is logged verbatim and names the
    offending value, because the only person who can fix it is whoever
    wrote the file.
    """


def from_json(text: str) -> SeverityScale:
    """Parse a repo's ``severities.json``. Raises ``InvalidScale``.

    Callers fall back to ``default_scale()`` on failure — a defined
    conservative gate beats an undefined one, and refusing to review a
    repo because one config file has a typo is worse than reviewing it
    under the built-in vocabulary. This function itself never falls back;
    that decision belongs to the caller.
    """
    try:
        body = json.loads(text)
    except (ValueError, TypeError) as e:
        raise InvalidScale(f"not valid JSON: {e}") from e

    if not isinstance(body, dict):
        raise InvalidScale("must be a JSON object")

    raw_ranks = body.get("severities")
    if not isinstance(raw_ranks, dict):
        raise InvalidScale("'severities' must be an object mapping tier name to rank")

    ranks: dict[str, int] = {}
    for name, rank in raw_ranks.items():
        clean = str(name).strip().lower()
        if not _NAME_RE.match(clean):
            raise InvalidScale(
                f"invalid tier name {name!r} — must match {_NAME_RE.pattern} "
                "(names become Prometheus label values and prompt text)"
            )
        if clean == BLOCKING:
            raise InvalidScale(
                f"tier name {name!r} is reserved as the notification-threshold "
                "sentinel (see severity.BLOCKING / a channel's "
                '"min_severity": "blocking") and cannot also be a tier name — '
                "pick a different name for this tier"
            )
        # bool is an int subclass; reject it explicitly.
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise InvalidScale(f"rank for {clean!r} must be an integer, got {rank!r}")
        # Duplicate-name check must run POST-normalization. JSON objects
        # cannot hold literally duplicate keys, but case/whitespace variants
        # are distinct keys that collapse after strip().lower():
        # {"High": 1, "high": 2} is two JSON keys and one tier. Checking the
        # two-tier minimum against raw_ranks would pass it through as a
        # single-tier scale where most_severe == least_severe.
        if clean in ranks:
            raise InvalidScale(
                f"duplicate tier name {clean!r} — {name!r} collides with an "
                "earlier key after case/whitespace normalization"
            )
        ranks[clean] = rank

    if len(ranks) < 2:
        raise InvalidScale("'severities' needs at least two tiers")

    if len(ranks) > _MAX_TIERS:
        raise InvalidScale(
            f"too many tiers ({len(ranks)}) — the maximum is {_MAX_TIERS}. "
            "Every tier is rendered into the review prompt for this repo, "
            "so an oversized scale crowds out the diff being reviewed"
        )

    if len(set(ranks.values())) != len(ranks):
        raise InvalidScale("duplicate rank — every tier needs a distinct rank")

    threshold = body.get("blocks_at_or_above")
    if threshold is not None:
        threshold = str(threshold).strip().lower()
        if threshold not in ranks:
            raise InvalidScale(
                f"'blocks_at_or_above' is {threshold!r}, not one of the "
                f"defined tiers: {', '.join(sorted(ranks))}"
            )

    raw_desc = body.get("descriptions")
    if raw_desc is None:
        raw_desc = {}
    if not isinstance(raw_desc, dict):
        raise InvalidScale(
            f"'descriptions' must be an object, got {type(raw_desc).__name__}")
    descriptions = {
        str(k).strip().lower(): str(v)
        for k, v in raw_desc.items()
        if str(k).strip().lower() in ranks
    }
    for tier, desc in descriptions.items():
        if len(desc) > _MAX_DESCRIPTION_CHARS:
            raise InvalidScale(
                f"description for tier {tier!r} is {len(desc)} characters — "
                f"the maximum is {_MAX_DESCRIPTION_CHARS}. Descriptions are "
                "rendered into every review prompt for this repo"
            )

    return SeverityScale(ranks=ranks, blocks_at_or_above=threshold,
                          descriptions=descriptions)


# ── prompt rendering ─────────────────────────────────────────────────── #

# The review-prompt opt-in escape hatch: the built-in template always
# carries this placeholder (see reviewer._apply_scale_to_template); an
# override author can opt into the rendered block by including it too.
SCALE_PLACEHOLDER = "{{severity_scale}}"


def render_severity_block(scale: SeverityScale) -> str:
    """Render the severity vocabulary as review-prompt markdown.

    This is the whole point of the design: the prompt's tier names and the
    gate's tier names are not two statements to be kept in agreement, they
    are one object read twice.
    """
    # Heading text is deliberately NOT "Severity Definitions": a repo can
    # legitimately name a tier "nit", and "Defi-nit-ions" would then
    # contain that tier name as a substring before its own heading does —
    # breaking any ordering check (incl. this module's own tests) that
    # scans the rendered block for the first occurrence of a tier name.
    lines = ["## Severity Scale", ""]
    for name in scale.ordered():
        if scale.blocks_at_or_above is None:
            marker = ""
        elif scale.blocks(name):
            marker = " — **blocks the merge**"
        else:
            marker = " — does not block a merge"
        lines.append(f"### {name}{marker}")
        desc = scale.descriptions.get(name)
        if desc:
            lines.append(desc)
        lines.append("")

    enum = "|".join(scale.ordered())
    # blocks_at_or_above is None means "nothing blocks on severity alone"
    # (an advisory-only scale) — the "and blocks the merge" clause must
    # mirror the per-tier loop above and drop out in that case, or the tail
    # contradicts the tiers it just rendered. The "treated as <most severe>"
    # half is unconditional: it's true regardless of blocking config, since
    # SeverityScale.normalize()/.rank() fail closed to most_severe no
    # matter what blocks_at_or_above is set to.
    if scale.blocks_at_or_above is None:
        unknown_name_clause = (
            "Use ONLY these severity names, exactly as spelled above. A name "
            "outside this list is treated as "
            f"`{scale.most_severe}`."
        )
    else:
        unknown_name_clause = (
            "Use ONLY these severity names, exactly as spelled above. A name "
            "outside this list is treated as "
            f"`{scale.most_severe}` and blocks the merge."
        )
    lines += [
        unknown_name_clause,
        "",
        f"In the JSON output, `severity` must be one of: `{enum}` — at the "
        "top level and on every finding.",
        "",
        f"Order findings by severity: {' → '.join(scale.ordered())}.",
    ]
    return "\n".join(lines)
