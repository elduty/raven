"""notifier.py — Channel-based notification dispatch."""

import json
import logging
import os
from urllib.parse import urlsplit

import requests

from . import metrics
from .reviewer import severity_gte
from .severity import BLOCKING, SeverityScale, default_scale

logger = logging.getLogger(__name__)

# Channels that already warned about an unresolvable min_severity for a
# given repo — logged once, not on every review, so a misconfigured
# channel doesn't spam the log on every push.
_warned_thresholds: set[str] = set()

# ── Channel registry ──────────────────────────────────────────────── #

def _load_channels() -> list[dict]:
    """Parse NOTIFY_CHANNELS JSON from env. Returns empty list if unset or invalid.

    Re-read on every notify() call so operators can update channel config without
    restarting the container — notifications are rare (severity-matching reviews
    only), so the parse cost is negligible compared to the HTTP call that follows.
    """
    raw = os.environ.get("NOTIFY_CHANNELS", "")
    if not raw:
        return []
    try:
        channels = json.loads(raw)
        if not isinstance(channels, list):
            logger.error("NOTIFY_CHANNELS must be a JSON array, got %s", type(channels).__name__)
            return []
        valid = []
        for i, ch in enumerate(channels):
            if not isinstance(ch, dict) or not ch.get("type") or not ch.get("url"):
                logger.error("NOTIFY_CHANNELS[%d]: missing required 'type' or 'url' — skipping", i)
                continue
            valid.append(ch)
        return valid
    except json.JSONDecodeError as e:
        logger.error("NOTIFY_CHANNELS is not valid JSON: %s", e)
        return []


# ── Public API ────────────────────────────────────────────────────── #

def notify(repo_name: str, ref: str, review: dict, link: str = "", action: str = "") -> bool:
    """Dispatch notification to all matching channels.

    Returns True if at least one channel succeeded, False otherwise.
    """
    channels = _load_channels()
    if not channels:
        return False

    text = _format_message(repo_name, ref, review, link, action)
    any_sent = False

    for channel in channels:
        # Per-repo filter
        repos = channel.get("repos")
        if repos and repo_name not in repos:
            continue

        # Per-channel severity filter. min_severity is operator config
        # naming a tier — meaningless in a repo that defined its own
        # vocabulary. Resolve by rank when the name exists in this repo's
        # scale, otherwise fall back to gate semantics ("notify when the
        # review blocks"), the only threshold that means the same thing in
        # every vocabulary. Failure direction is notify-rather-than-suppress:
        # a missed alert is worse than a redundant one, and this path has no
        # merge authority.
        min_sev = channel.get("min_severity")
        if min_sev and not _passes_threshold(review, min_sev, repo_name):
            continue

        channel_type = channel.get("type", "")
        try:
            if channel_type == "slack":
                _send_slack(channel, text)
                any_sent = True
            elif channel_type == "webhook":
                _send_webhook(channel, text)
                any_sent = True
            else:
                logger.warning("Unknown notification channel type: %s", channel_type)
        except Exception as e:
            # Never log str(e): requests exceptions embed the full request URL,
            # and webhook URLs are bearer-equivalent secrets (for Slack incoming
            # webhooks the path IS the credential). Log only the exception class,
            # the HTTP status when available (no secret material; distinguishes a
            # revoked/mistyped webhook 404 from a transient 5xx), and the
            # hostname (NOT netloc, which would leak userinfo like user:pass@).
            logger.error(
                "Notification failed for channel %s (%s): %s",
                channel_type,
                _redacted_host(channel.get("url", "")),
                _exception_summary(e),
            )

    return any_sent


def _scale_from_review(review: dict) -> SeverityScale:
    """Reconstruct the reviewed repo's ``SeverityScale`` from the fields
    that cross the reviewer -> notifier boundary on the review dict
    (``severity_scale_names`` / ``severity_blocks_at``).

    Shared by ``_passes_threshold`` and ``_format_message`` so there is one
    reconstruction, not two independently-maintained copies. Ranks are
    positional (``len(names) - i``), which preserves *order* but not the
    scale's original rank numbers — correct for every comparison this
    module makes (emoji position, threshold rank), and why ``fingerprint()``
    is never called on the result.

    Falls back to ``default_scale()`` when the review dict carries no scale
    at all — a legacy or cached review from before this feature — matching
    today's exact behaviour rather than inventing new colours/defaults for
    old data.
    """
    names = review.get("severity_scale_names") or []
    if not names:
        return default_scale()
    ranks = {n: len(names) - i for i, n in enumerate(names)}
    # The blocking tier arrives on its own field, so a malformed or
    # partially-updated review dict can name a tier this scale does not
    # contain — SeverityScale.blocks() would then raise KeyError on
    # ranks[blocks_at_or_above] and lose the notification for a review
    # that already ran. Every current writer emits both fields from one
    # scale object, so this is an invariant guard rather than a live
    # bug; degrade to "nothing blocks on severity alone", which fails in
    # this module's usual direction — toward notifying.
    blocks_at = review.get("severity_blocks_at")
    if blocks_at not in ranks:
        blocks_at = None
    return SeverityScale(ranks=ranks, blocks_at_or_above=blocks_at)


def _passes_threshold(review: dict, min_sev: str, repo_name: str) -> bool:
    """Does ``review`` clear a channel's ``min_severity`` config?

    ``min_severity`` is operator config naming a tier in SOME severity
    vocabulary — but with per-repo scales, a channel filtering on
    "medium" is meaningless for a repo whose tiers are nit/bug/blocker.
    Resolution mirrors the emoji decision: compare by rank when the name
    exists in this review's scale; otherwise fall back to gate semantics
    (``BLOCKING`` sentinel, or any name absent from the scale) — "notify
    when the review blocks the merge" is the only threshold that means
    the same thing in every vocabulary.

    Unknown-name handling is deliberately NOT ``SeverityScale.normalize``'s
    fail-closed-to-most-severe: that contract is for MODEL-emitted
    severities (an untrusted claim, where "I don't know" must assume the
    worst). Here the unknown value is OPERATOR config, and failing closed
    would mean silently suppressing notifications on a typo — the wrong
    direction when this path has no merge authority and a missed alert
    outweighs a redundant one.

    ``severity`` on ``review`` itself defaults to ``scale.least_severe``,
    matching ``_format_message`` — not ``""``. The reviewer always
    populates it, so this only matters for a degenerate/legacy dict, but
    the two must agree: ``""`` isn't a tier ``SeverityScale.rank()`` can
    see, so it used to normalize fail-closed to the scale's MOST severe
    tier here while ``_format_message`` rendered the identical dict as the
    LEAST severe one — the exact contradictory-defaults defect class this
    task exists to prevent (PR #216 review, Finding 3).
    """
    scale = _scale_from_review(review)
    severity = review.get("severity", scale.least_severe)
    names = review.get("severity_scale_names") or []
    # Read the blocking tier off the reconstructed scale, not the raw
    # dict field: _scale_from_review drops a tier it cannot resolve, and
    # the two must agree or the `blocks_at is None` short-circuit below
    # would fall through to a scale.blocks() that disagrees with it.
    blocks_at = scale.blocks_at_or_above
    clean = str(min_sev).strip().lower()

    # No scale on the review dict at all — a legacy or cached review from
    # before this feature. Fall back to today's exact behaviour rather
    # than to gate semantics: with no scale, blocks_at is also absent,
    # and "notify when it blocks" would degrade into "always notify" and
    # break every existing channel filter.
    if not names and clean != BLOCKING:
        return severity_gte(severity, clean)

    if clean != BLOCKING and names and clean in names:
        return scale.rank(severity) >= scale.rank(clean)

    if clean != BLOCKING and names and clean not in names:
        key = f"{repo_name}:{clean}"
        if key not in _warned_thresholds:
            _warned_thresholds.add(key)
            logger.warning(
                "Channel min_severity %r is not a tier in %s's severity scale "
                "(%s) — falling back to 'notify when the review blocks'. Use "
                "'blocking' to make this explicit.",
                clean, repo_name, ", ".join(names),
            )
        metrics.inc("raven_notify_threshold_fallback_total", {"repo": repo_name})

    if blocks_at is None:
        return True
    return scale.blocks(severity)


def _redacted_host(url: str) -> str:
    """Hostname (plus port, if any) of a channel URL — safe to log.

    Excludes the path (Slack webhook credential) and userinfo (basic-auth
    credentials), both of which urlsplit's netloc would include.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        return f"{host}:{parts.port}" if parts.port else host
    except ValueError:
        return ""


def _exception_summary(e: Exception) -> str:
    """Exception class name, plus the HTTP status code when present."""
    status = getattr(getattr(e, "response", None), "status_code", "")
    return f"{type(e).__name__} (HTTP {status})" if status else type(e).__name__


# ── Message formatting ────────────────────────────────────────────── #

def _format_message(repo_name: str, ref: str, review: dict, link: str, action: str) -> str:
    scale = _scale_from_review(review)
    severity = review.get("severity", scale.least_severe)
    summary = review.get("summary", "")
    emoji, label = scale.badge(severity, review.get("findings"),
                               blocking=bool(review.get("blocking")))

    if action == "merge_failed":
        header = "🦅 *Raven* — ⚠️ Auto-merge failed"
    elif action == "ci_failed":
        header = "🦅 *Raven* — ❌ CI failed"
    elif action == "ci_timeout":
        header = "🦅 *Raven* — ⏳ CI timed out"
    elif action == "comment_failed":
        header = "🦅 *Raven* — ⚠️ Failed to post review comment"
    elif action == "review_failed":
        header = "🦅 *Raven* — ⚠️ Review failed — could not parse output"
    elif action == "review_submit_failed":
        header = "🦅 *Raven* — ⚠️ Failed to submit review"
    elif action == "needs_review":
        header = f"🦅 *Raven* — {emoji} {label} — needs your review"
    else:
        header = f"🦅 *Raven Alert* — {emoji} {label}"

    text = (
        f"{header}\n"
        f"*{repo_name}* · {ref}\n"
        f"{summary}"
    )
    if link:
        text += f"\n{link}"

    return text


# ── Channel senders ───────────────────────────────────────────────── #

def _send_slack(channel: dict, text: str) -> None:
    """Send a notification via Slack incoming webhook. Raises on failure."""
    url = channel.get("url", "")
    if not url:
        raise ValueError("Slack channel missing 'url'")

    resp = requests.post(url, json={"text": text}, timeout=10)
    resp.raise_for_status()
    logger.info("Slack notification sent")


def _send_webhook(channel: dict, text: str) -> None:
    """Send a notification via generic webhook (POST JSON). Raises on failure."""
    url = channel.get("url", "")
    if not url:
        raise ValueError("Webhook channel missing 'url'")

    token = channel.get("token", "")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(url, json={"text": text}, headers=headers, timeout=10)
    resp.raise_for_status()
    logger.info("Webhook notification sent")
