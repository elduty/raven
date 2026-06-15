"""notifier.py — Channel-based notification dispatch."""

import json
import logging
import os
from urllib.parse import urlsplit

import requests

from .reviewer import severity_gte

logger = logging.getLogger(__name__)

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

    severity = review.get("severity", "low")

    for channel in channels:
        # Per-repo filter
        repos = channel.get("repos")
        if repos and repo_name not in repos:
            continue

        # Per-channel severity filter
        min_sev = channel.get("min_severity")
        if min_sev and not severity_gte(severity, min_sev):
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

SEVERITY_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _format_message(repo_name: str, ref: str, review: dict, link: str, action: str) -> str:
    severity = review.get("severity", "low")
    summary = review.get("summary", "")
    emoji = SEVERITY_EMOJI.get(severity, "🟡")

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
        header = f"🦅 *Raven* — {emoji} {severity.upper()} — needs your review"
    else:
        header = f"🦅 *Raven Alert* — {emoji} {severity.upper()}"

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
