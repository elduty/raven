"""server.py — Flask app with webhook endpoints for git platform providers."""

import atexit
import contextlib
import functools
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request

from .providers import GitProvider, DiffTruncatedError, get_provider, register_provider, registered_providers
from .providers.gitea import GiteaProvider
from .metrics import add, inc, Timer, format_prometheus
from .notifier import notify
from .reviewer import review_diff, respond_to_comment, severity_gte, SEVERITY_ORDER, review_config_hash, _strip_lockfiles_and_binaries, split_diff_by_file, MAX_DIFF_LINES, terminate_active_processes, RespondParseError, RAVEN_AI_MODEL, RAVEN_AI_EFFORT, RAVEN_AI_TIMEOUT, RAVEN_AI_RETRY
from .ai.base import AIError

_SEVERITY_NAME = {v: k for k, v in SEVERITY_ORDER.items()}


def _review_failure_reason(exc: Exception) -> str:
    """Classify an unhandled review exception into a failure reason.

    An :class:`AIError` already carries the classified ``.reason`` (set by
    the backend). A :class:`DiffTruncatedError` (provider returned a
    truncated/partial diff — too large for the platform's diff limit) maps
    to ``"diff_truncated"``. Anything else (a non-AI bug in the flow, an
    out-of-tree backend raising plain ``RuntimeError``) is ``"unknown"``.
    """
    if isinstance(exc, AIError):
        return exc.reason
    if isinstance(exc, DiffTruncatedError):
        return "diff_truncated"
    return "unknown"


# Operator-facing failure messages, keyed by reason. Each NAMES the cause
# in plain language and states the actionable next step — so an operator
# reading the PR (not the host logs) knows what happened and what to do.
# These are static templates: no exception text / str(e) is interpolated,
# so a credential-bearing error message can never leak into the comment
# (the redaction concern from PR #156). The retry-class messages reflect
# that reviewer.py already retried RAVEN_AI_RETRY time(s) before this.
_RETRY_NOTE = (
    f" Raven retried automatically {RAVEN_AI_RETRY}× and it still failed."
    if RAVEN_AI_RETRY else ""
)
_FAILURE_MESSAGES = {
    "timeout": (
        f"⏱️ The review timed out after {RAVEN_AI_TIMEOUT}s.{_RETRY_NOTE} "
        f"For large diffs at high effort, raise `RAVEN_AI_TIMEOUT` (and re-trigger "
        f"by pushing a commit)."
    ),
    "usage_limit": (
        "🚦 The AI provider's usage/session limit was reached — this resets "
        "automatically after a cooldown. Raven will retry on the next push or "
        "when you re-trigger the review; no config change is needed."
    ),
    "rate_limit": (
        f"🚦 The AI provider rate-limited the request (429).{_RETRY_NOTE} "
        f"Push a commit to re-trigger, or retry shortly."
    ),
    "backend_5xx": (
        f"🌧️ The AI backend returned a transient server error.{_RETRY_NOTE} "
        f"Push a commit to re-trigger, or retry shortly."
    ),
    "auth": (
        "🔒 The AI backend rejected Raven's credentials (auth error). This "
        "needs an operator to check the configured API key / OAuth token — "
        "re-triggering won't help until it's fixed."
    ),
    "diff_truncated": (
        "📐 The diff is too large — the platform (Bitbucket) truncated it, so "
        "Raven received only part of the change and won't review a partial diff "
        "(it could otherwise approve or merge code it never saw). Split the PR "
        "into smaller changes, or raise the server's diff size limit "
        "(`diff.max.lines` / related `*.diff.*` properties), then push a commit "
        "to re-trigger."
    ),
    "unknown": (
        "⚠️ Internal error — review could not be completed. Check the service "
        "logs for details, then push a commit to re-trigger."
    ),
}


def _failure_comment(reason: str) -> str:
    """Build the operator-facing failure comment for a classified reason.

    Keeps the 🦅 header. Falls back to the generic ``unknown`` message for
    any unrecognised reason so a new backend reason can never produce a
    KeyError / empty comment.
    """
    detail = _FAILURE_MESSAGES.get(reason, _FAILURE_MESSAGES["unknown"])
    return f"🦅 **Raven Review**\n\n{detail}"

logger = logging.getLogger(__name__)

_GITEA_AUTO_MERGE = os.environ.get("RAVEN_GITEA_AUTO_MERGE", "").lower() in ("1", "true", "yes")

# Review-engagement mode. Single source of truth for "which PRs does
# Raven engage with, and how blocking is the output?"
#   * all       — auto-add to every PR, submit formal review, auto-merge when sole reviewer.
#   * gap       — only auto-add when no other reviewer is listed; submit formal review.
#   * advisory  — never auto-add; post a non-blocking recommendation comment.
_VALID_REVIEW_MODES = {"all", "gap", "advisory"}


def _resolve_review_mode() -> str:
    # Treat empty string as unset. docker-compose `${VAR:-}` substitutes the
    # empty string into the container when the host var is unset; without the
    # `or "all"` fallback, the strict validator below would reject "" and
    # crash-loop the container on default deployment.
    raw = os.environ.get("RAVEN_REVIEW_MODE", "").strip().lower() or "all"
    if raw not in _VALID_REVIEW_MODES:
        logger.error(
            "Invalid RAVEN_REVIEW_MODE=%r (expected one of %s) — exiting",
            raw, sorted(_VALID_REVIEW_MODES),
        )
        sys.exit(1)
    return raw


RAVEN_REVIEW_MODE = _resolve_review_mode()

# Which review output channels Raven emits on a PR review:
#   * both     — the summary comment (verdict + findings list) AND per-line
#                inline comments (default).
#   * summary  — only the summary comment; no inline comments.
#   * inline   — only per-line inline comments; the summary body is trimmed
#                to verdict + one-liner. Findings that have no postable
#                file/line still appear in the body (nothing is lost).
# Orthogonal to RAVEN_REVIEW_MODE (which controls blocking vs advisory).
_VALID_REVIEW_OUTPUTS = {"both", "summary", "inline"}


def _resolve_review_output() -> str:
    # Same empty-string-as-unset handling as _resolve_review_mode: an unset
    # docker-compose `${VAR:-}` becomes "" and must fall back to the default.
    raw = os.environ.get("RAVEN_REVIEW_OUTPUT", "").strip().lower() or "both"
    if raw not in _VALID_REVIEW_OUTPUTS:
        logger.error(
            "Invalid RAVEN_REVIEW_OUTPUT=%r (expected one of %s) — exiting",
            raw, sorted(_VALID_REVIEW_OUTPUTS),
        )
        sys.exit(1)
    return raw


RAVEN_REVIEW_OUTPUT = _resolve_review_output()

executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("RAVEN_MAX_WORKERS", "16")),
    thread_name_prefix="raven-review",
)

# Dedicated pool for the post-review wait-and-merge phase. CI polling
# spends nearly all its time sleeping between status checks (up to
# ``CI_WAIT_TIMEOUT``, default 300s). Running it on the main review
# executor pins pool slots while doing no useful work, so during a burst
# (team pushes a batch of PRs at release time) every new webhook queues
# behind workers that are asleep. Give the wait phase its own larger,
# sleep-heavy pool so review throughput is preserved.
ci_wait_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get("RAVEN_CI_WAIT_WORKERS", "32")),
    thread_name_prefix="raven-ci-wait",
)


def _log_future_exception(fut, repo: str = "unknown") -> None:
    """Future.exception() hides exceptions until .result() is called.
    The wait-pool tasks are fire-and-forget, so nobody calls .result()
    — attach this as a done_callback to surface unhandled errors in
    logs and metrics instead of silently losing them.

    ``fut.cancelled()`` is checked first because ``Future.exception()``
    on a cancelled future raises ``CancelledError``, which is a
    ``BaseException`` (not ``Exception``) subclass since Python 3.8 and
    would therefore escape the ``except Exception`` clause. Cancellation
    during ``_shutdown_executor``'s drain is expected, not an error.
    """
    if fut.cancelled():
        return
    try:
        exc = fut.exception()
    except Exception:
        return
    if exc is not None:
        logger.error("Unhandled exception in CI-wait task for %s: %s", repo, exc, exc_info=exc)
        inc("raven_errors_total", {"type": "ci_wait_unhandled", "repo": repo})


def _shutdown_executor() -> None:
    """Cancel queued reviews and terminate in-flight Claude subprocesses.

    Three steps:

    1. ``executor.shutdown(cancel_futures=True, wait=False)`` drops
       reviews still sitting in the queue so the pool doesn't pick them
       up during interpreter shutdown.
    2. ``ci_wait_executor`` is drained the same way, so queued
       wait-and-merge tasks are dropped. Running wait tasks block in
       ``time.sleep`` between polls; they'll exit within one poll
       interval rather than consuming the full ``CI_WAIT_TIMEOUT``.
    3. ``terminate_active_processes()`` sends SIGTERM (then SIGKILL) to
       any Claude CLI subprocesses that are mid-call. Running reviews
       that survive step 1 are almost always blocked in
       ``proc.communicate()`` waiting for LLM inference; killing the
       subprocess unblocks the worker thread so gunicorn's graceful
       timeout isn't consumed by work whose result we'll throw away.

    ``wait=False`` is cosmetic on its own — Python's own
    ``concurrent.futures.thread._python_exit`` atexit handler joins
    every live worker thread anyway. Step 3 is what actually shortens
    shutdown: without it, a worker blocked in a Claude call could
    still hold up exit for the full ``RAVEN_AI_TIMEOUT``.

    Gunicorn's sync worker handles SIGTERM by calling ``sys.exit(0)``
    (not via the OS default signal action), which raises SystemExit
    and triggers atexit. SIGKILL bypasses atexit entirely.
    """
    # Guard against "I/O operation on closed file": the logging
    # module's handlers may have already been torn down by the time
    # atexit runs. atexit catches exceptions itself, but suppressing
    # here avoids the traceback-on-stderr noise.
    with contextlib.suppress(Exception):
        logger.info("Shutting down review + CI-wait executors — cancelling queued work")
    executor.shutdown(wait=False, cancel_futures=True)
    ci_wait_executor.shutdown(wait=False, cancel_futures=True)
    with contextlib.suppress(Exception):
        terminate_active_processes()


# Record what we hand to atexit so tests can assert registration
# happened without relying on CPython internals (``_exithandlers``
# doesn't exist; ``unregister`` returns None; ``_ncallbacks`` doesn't
# decrement on unregister). Regression guards check membership here.
_ATEXIT_HOOKS: list = []


def _register_atexit(fn):
    atexit.register(fn)
    _ATEXIT_HOOKS.append(fn)


_register_atexit(_shutdown_executor)

# ── PR dedup: prevent concurrent reviews for the same PR ──────────── #
_recent_prs: dict[str, float] = {}
_recent_prs_lock = threading.Lock()
DEDUP_WINDOW = 30  # seconds

# ── Per-PR reply circuit breaker ───────────────────────────────────── #
# Backstop against unbounded comment-reply loops the bot-author name
# heuristic misses (e.g. a second Raven, or an auto-responder under a
# human-looking name). On BB DC, Raven replies to any in-thread reply
# without an @mention, so each bot reply — a fresh comment_id the 30s
# dedup never catches — would trigger another paid AI call. We cap the
# number of *dispatched* replies per PR over a sliding 1-hour window.
# Mirrors the _recent_prs pattern: module-level dict of
# ``f"{provider}:{repo}#{pr}" -> [timestamps]``, guarded by a lock,
# pruned on every check.
_recent_pr_replies: dict[str, list[float]] = {}
_recent_pr_replies_lock = threading.Lock()
REPLY_BUDGET_WINDOW = 3600  # seconds (sliding 1-hour window)

# ── In-progress guard ─────────────────────────────────────────────── #
# Dedup's 30s window is shorter than a full review (diff fetch +
# Claude CLI + CI wait up to CI_WAIT_TIMEOUT). If a second webhook
# arrives after dedup expires while the original review is still
# running, both threads race on the findings cache, the submit_review
# API, and the merge decision. _in_progress_prs tracks keys currently
# being processed by _process_pr; a second concurrent review on the
# same PR exits immediately.
_in_progress_prs: set[str] = set()
# Separate set for comment-driven mutations: gives push priority (push
# webhooks check ONLY _in_progress_prs and never wait on a comment-flow)
# AND serializes concurrent comment-flows on the same PR (without it, two
# comments arriving within ~10s would both pass the TOCTOU re-check before
# either wrote the cache, then both submit_review + dismiss each other's
# review).
_comment_mutating_prs: set[str] = set()
_in_progress_lock = threading.Lock()

# ── Comment response history window ──────────────────────────────── #
COMMENT_HISTORY = int(os.environ.get("RAVEN_COMMENT_HISTORY", "20"))

# ── Previous diff cache for incremental reviews ──────────────────── #
@dataclass
class CacheEntry:
    """A cached PR review state. Verdict + summary are populated since the
    comment-thread-context feature (2026-05-13); legacy 3-tuple entries
    loaded from older cache files default these to None. Per-finding
    `comment_id` (set at submit time) lives inside `findings[fname][i]`
    so retraction can match cached findings back to provider comments."""
    timestamp: float
    hashes: dict[str, str]            # filename -> diff_hash
    findings: dict[str, list]         # filename -> [findings]
    verdict: str | None = None        # 'approve' | 'needs_work' | None
    summary: str | None = None        # last review's top-level body
    # Unreviewed files from the last review: chunks skipped as oversized
    # or whose review failed (reviewer.py returns ``coverage_gap_files``
    # on the review dict, keyed by the same diff-split filenames as
    # ``hashes``). While non-empty, _process_pr forces the verdict to
    # needs_work and both merge-dispatch paths (_process_pr and
    # _process_comment) refuse auto-merge — a comment-driven verdict
    # flip can't see the unreviewed files any better than the original
    # review did. Per-file (not a bool) so the gap CLEARS once a named
    # file changes and re-reviews cleanly: an incremental pass carries
    # forward only the gap files that are NOT in changed_files (a file
    # REMOVED from the PR clears via the removed-files full-re-review
    # gate instead). Defaults empty so cache files written before this
    # field load as no-gap. Lifecycle documented in CLAUDE.md
    # ("Coverage-gap tracking"); pinned end-to-end by
    # tests/test_server.py::TestCoverageGapBlocksMerge.
    coverage_gap_files: list[str] = field(default_factory=list)


_previous_diffs: dict[str, CacheEntry] = {}
_previous_diffs_lock = threading.Lock()
_MAX_CACHED_PRS = int(os.environ.get("RAVEN_MAX_CACHED_PRS", "200"))
_CACHE_DIR = Path(os.environ.get("RAVEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "raven")))
_CACHE_FILE = _CACHE_DIR / "findings_cache.json"


def _load_cache() -> None:
    """Load findings cache from disk on startup. Wipes cache if config changed.

    Entries are dicts with the full ``CacheEntry`` schema. Legacy
    3-tuple entries (pre-2026-05-13) are no longer recognized — on a
    legacy cache file the entries fail per-row guards and the cache
    re-warms from the next push.
    """
    try:
        if not _CACHE_FILE.exists():
            return
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        # Check config hash — wipe cache if model/prompt changed
        stored_hash = data.get("_config_hash", "")
        current_hash = review_config_hash()
        if stored_hash != current_hash:
            logger.info("Review config changed (hash %s -> %s) — discarding cached findings",
                        stored_hash[:8] or "none", current_hash[:8])
            return
        entries = data.get("entries", {})
        skipped = 0
        with _previous_diffs_lock:
            for key, entry in entries.items():
                # Per-entry guard: one corrupt row (missing required key,
                # wrong type) must not abort the load loop and leave the
                # other 199 healthy entries behind.
                try:
                    _previous_diffs[key] = CacheEntry(
                        timestamp=entry["timestamp"],
                        hashes=entry["hashes"],
                        findings=entry["findings"],
                        verdict=entry.get("verdict"),
                        summary=entry.get("summary"),
                        coverage_gap_files=list(entry.get("coverage_gap_files") or []),
                    )
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning("Skipping malformed cache entry %s: %s", key, e)
                    skipped += 1
        if skipped:
            logger.warning("Loaded %d cached PR reviews from %s (skipped %d malformed)",
                           len(_previous_diffs), _CACHE_FILE, skipped)
        else:
            logger.info("Loaded %d cached PR reviews from %s",
                        len(_previous_diffs), _CACHE_FILE)
    except Exception as e:
        logger.warning("Could not load findings cache from %s: %s", _CACHE_FILE, e)


def _save_cache() -> None:
    """Persist findings cache to disk (atomic write).

    The cache dict is snapshotted under ``_previous_diffs_lock`` via
    ``asdict()`` (which deep-copies the dataclass + nested findings
    lists), then the lock is released BEFORE the disk write. Holding
    the lock across I/O would block concurrent reviewers for the
    duration of the write; the snapshot is independent of further
    mutations so this is safe.

    All exceptions are caught and surfaced via
    ``raven_cache_save_failures_total`` so persistent disk/permission
    problems are alertable in monitoring. The function never propagates
    — callers can invoke it bare without try/except.
    """
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _previous_diffs_lock:
            data = {
                "_config_hash": review_config_hash(),
                "entries": {k: asdict(v) for k, v in _previous_diffs.items()},
            }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
        try:
            f = os.fdopen(tmp_fd, "w", encoding="utf-8")
        except Exception:
            os.close(tmp_fd)
            raise
        try:
            with f:
                json.dump(data, f)
            os.replace(tmp_path, _CACHE_FILE)
        except Exception:
            os.unlink(tmp_path)
            raise
    except Exception as e:
        logger.warning("Could not save findings cache to %s: %s", _CACHE_FILE, e)
        inc("raven_cache_save_failures_total", {"reason": type(e).__name__})


def _evict_cache() -> None:
    """Evict oldest entries if cache exceeds _MAX_CACHED_PRS."""
    with _previous_diffs_lock:
        if len(_previous_diffs) <= _MAX_CACHED_PRS:
            return
        sorted_keys = sorted(_previous_diffs, key=lambda k: _previous_diffs[k].timestamp)
        to_remove = len(_previous_diffs) - _MAX_CACHED_PRS
        for key in sorted_keys[:to_remove]:
            del _previous_diffs[key]


def _should_skip_duplicate(repo: str, pr_number: int | str, head_sha: str | None = None) -> bool:
    """Return True if this dispatch target was already seen within DEDUP_WINDOW.

    ``pr_number`` is the natural use case but the parameter accepts any
    stringifiable dedup identifier — the comment-flow at the webhook
    route passes ``f"comment-{id}-v{version}"`` so each comment edit
    gets its own slot. The key is just ``f"{repo}#{pr_number}"`` (plus
    an optional SHA suffix) — anything stringifiable works.

    When ``head_sha`` is provided, it's appended to the dedup key so that
    a push carrying a new SHA is treated as a fresh event (not a webhook
    redelivery of the original). Dedup's purpose is to absorb duplicate
    deliveries of the *same* event — different SHAs are different events.
    """
    suffix = f"@{head_sha}" if head_sha else ""
    key = f"{repo}#{pr_number}{suffix}"
    now = time.time()
    with _recent_prs_lock:
        if key in _recent_prs and now - _recent_prs[key] < DEDUP_WINDOW:
            return True
        _recent_prs[key] = now
        # Prune stale entries
        stale = [k for k, t in _recent_prs.items() if now - t > DEDUP_WINDOW * 2]
        for k in stale:
            del _recent_prs[k]
    return False


def _reply_budget_exceeded(key: str, max_per_hour: int) -> bool:
    """Atomic check-and-record for the per-PR reply circuit breaker.

    ``key`` is ``f"{provider}:{repo}#{pr}"``. Returns ``True`` (and records
    nothing) when the PR has already received ``max_per_hour`` dispatched
    replies inside the sliding ``REPLY_BUDGET_WINDOW``; otherwise records a
    fresh timestamp and returns ``False``. The record-on-allow is inside the
    same lock as the count check so two concurrent webhooks can't both slip
    past a budget of N (no TOCTOU). Timestamps outside the window are pruned
    on every call, and empty buckets are dropped so the dict can't grow
    unbounded across PRs. ``max_per_hour <= 0`` disables the limiter (always
    allows), matching the env-driven "off" switch convention."""
    if max_per_hour <= 0:
        return False
    now = time.time()
    cutoff = now - REPLY_BUDGET_WINDOW
    with _recent_pr_replies_lock:
        # Prune stale buckets/timestamps across all tracked PRs.
        for k in list(_recent_pr_replies):
            fresh = [t for t in _recent_pr_replies[k] if t > cutoff]
            if fresh:
                _recent_pr_replies[k] = fresh
            else:
                del _recent_pr_replies[k]
        bucket = _recent_pr_replies.get(key, [])
        if len(bucket) >= max_per_hour:
            return True
        bucket.append(now)
        _recent_pr_replies[key] = bucket
    return False


# ── Comment-mention detection ─────────────────────────────────────── #
# A comment "tags" Raven when it @-mentions the bot's account username OR a
# configured display name. Shared by the comment-reply gate in both modes.

_CODE_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _mention_names() -> list[str]:
    """Display names that count as a mention, in addition to the bot's
    account username. Comma-separated ``RAVEN_MENTION_NAMES`` (default
    ``"Raven"``); set it empty to recognize ONLY the account username. Read
    at request time so it's tunable without a restart."""
    raw = os.environ.get("RAVEN_MENTION_NAMES", "Raven")
    return [n.strip() for n in raw.split(",") if n.strip()]


def _comment_tags_raven(body: str, names: list[str]) -> bool:
    r"""True when ``body`` @-mentions any of ``names`` (case-insensitive).

    Accepts ``@name`` and the BB-DC-quoted ``@"name"`` form. Hardened against
    false positives:
      * ``(?<!\w)`` — the ``@`` must not follow a word char, so email local
        parts / ``user@host`` strings don't match (``deploy@ravenbot.io``).
      * fenced code blocks and inline code spans are stripped first, so a tag
        quoted inside ``code`` never triggers a reply.
    """
    valid = [n for n in names if n]
    if not valid:
        return False
    cleaned = _INLINE_CODE_RE.sub(" ", _CODE_FENCE_RE.sub(" ", body))
    alt = "|".join(re.escape(n) for n in valid)
    pattern = rf'(?<!\w)@(?:"(?:{alt})"|(?:{alt})\b)'
    return bool(re.search(pattern, cleaned, re.IGNORECASE))


# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def create_app() -> Flask:
    # Register providers based on available env vars
    gitea_url = os.environ.get("GITEA_URL")
    gitea_token = os.environ.get("GITEA_TOKEN")
    gitea_secret = os.environ.get("GITEA_WEBHOOK_SECRET")
    if gitea_url and gitea_token and gitea_secret:
        register_provider("gitea", GiteaProvider(gitea_url, gitea_token, gitea_secret))

    bb_dc_url = os.environ.get("BITBUCKET_DC_URL")
    bb_dc_token = os.environ.get("BITBUCKET_DC_TOKEN")
    bb_dc_secret = os.environ.get("BITBUCKET_DC_WEBHOOK_SECRET")
    if bb_dc_url and bb_dc_token and bb_dc_secret:
        from .providers.bitbucket_dc import BitbucketDCProvider
        bb_dc_username = os.environ.get("BITBUCKET_DC_USERNAME", "")
        register_provider("bitbucket-dc", BitbucketDCProvider(
            bb_dc_url, bb_dc_token, bb_dc_secret, username=bb_dc_username,
        ))
        logger.info("Registered provider: bitbucket-dc (user=%s)", bb_dc_username or "<unset>")
        if os.environ.get("MERGE_STRATEGY") and os.environ.get("MERGE_STRATEGY") != "squash":
            logger.warning("MERGE_STRATEGY is set but Bitbucket DC controls merge strategy via repo settings — this value is ignored for BB DC repos")

    if not registered_providers():
        raise RuntimeError(
            "No git providers configured. Set GITEA_URL + GITEA_TOKEN + GITEA_WEBHOOK_SECRET, "
            "or BITBUCKET_DC_URL + BITBUCKET_DC_TOKEN + BITBUCKET_DC_WEBHOOK_SECRET + BITBUCKET_DC_USERNAME."
        )

    app = Flask(__name__)

    # Cap the request body so an unauthenticated client can't spike worker
    # memory by POSTing a huge payload BEFORE the HMAC check buffers it
    # (validate_signature -> request.get_data()). Flask returns 413 before the
    # body is read. 25 MB is far above any real webhook+diff payload — Raven
    # fetches diffs via the provider API, not from the webhook body. (audit #12)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    _load_cache()

    metrics_token = os.environ.get("RAVEN_METRICS_TOKEN", "")
    if not metrics_token:
        logging.warning("RAVEN_METRICS_TOKEN not set — /metrics endpoint disabled (returns 404)")

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.route("/metrics")
    def metrics():
        if not metrics_token:
            abort(404)
        header = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix) or not hmac.compare_digest(header[len(prefix):], metrics_token):
            abort(404)
        return format_prometheus(), 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.route("/hook/<provider_name>", methods=["POST"])
    def hook(provider_name):
        provider = get_provider(provider_name)
        if not provider:
            abort(404, f"Unknown provider: {provider_name}")

        provider.validate_signature(request)
        result = provider.parse_webhook(request)
        if result is None:
            return jsonify({"status": "ignored"})

        event_type, payload = result
        repo = payload["repo"]
        sender = payload.get("sender", "")

        if _is_skipped_repo(repo):
            return jsonify({"status": "skipped"})

        if event_type == "push":
            if _is_bot_author(sender):
                return jsonify({"status": "skipped"})
            branch = payload["branch"]
            default_branch = payload["default_branch"]
            if branch == default_branch:
                return jsonify({"status": "skipped", "reason": "push to default branch"})
            try:
                pr = provider.find_open_pr_for_branch(repo, branch)
            except Exception as e:
                logger.warning("Could not look up PR for branch %s: %s", branch, e)
                pr = None
            if not pr:
                return jsonify({"status": "skipped", "reason": "no open PR for branch"})
            pr_number = pr.get("number")
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            head_sha = pr.get("head", {}).get("sha", "HEAD")
            if _should_skip_duplicate(f"{provider.name}:{repo}", pr_number, head_sha=head_sha):
                logger.info("Push to PR branch %s — skipping duplicate for PR #%s", branch, pr_number)
                return jsonify({"status": "skipped", "reason": "duplicate"})
            logger.info("Push to PR branch %s — triggering re-review for PR #%s", branch, pr_number)
            # Enrich payload with PR details from the lookup
            payload["pr_number"] = pr_number
            payload["pr_title"] = pr.get("title", f"PR #{pr_number}")
            payload["pr_url"] = pr.get("html_url", "")
            payload["head_sha"] = head_sha
            payload["head_ref"] = pr.get("head", {}).get("ref", "")
            payload["base_ref"] = pr.get("base", {}).get("ref", "")
            executor.submit(_process_pr, provider, payload)
            return jsonify({"status": "accepted", "reason": "re-review triggered"})

        elif event_type in ("pr_opened", "pr_updated", "pr_reopened"):
            if _is_bot_author(sender):
                return jsonify({"status": "skipped"})
            pr_number = payload["pr_number"]
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            if _should_skip_duplicate(f"{provider.name}:{repo}", pr_number,
                                     head_sha=payload.get("head_sha") or "HEAD"):
                logger.info("Skipping duplicate review for %s PR #%s", repo, pr_number)
                return jsonify({"status": "skipped", "reason": "duplicate"})
            executor.submit(_process_pr, provider, payload)
            return jsonify({"status": "accepted"})

        elif event_type == "review_requested":
            pr_number = payload["pr_number"]
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            # Verify the review was requested for the bot
            requested = payload.get("requested_reviewer", "")
            try:
                raven_user = provider.get_authenticated_user()
            except Exception:
                raven_user = ""
            if not requested or not raven_user:
                return jsonify({"status": "ignored", "reason": "cannot verify reviewer identity"})
            if requested.lower() != raven_user.lower():
                return jsonify({"status": "ignored", "reason": "review requested for another user"})
            # Ignore the webhook that fires when Raven adds itself as a reviewer.
            # Otherwise the add_self_as_reviewer call at the start of _process_pr
            # triggers a second _process_pr via pr:reviewer:updated.
            if sender and sender.lower() == raven_user.lower():
                return jsonify({"status": "ignored", "reason": "self-triggered"})
            # Dedup review requests separately from normal PR events
            if _should_skip_duplicate(f"{provider.name}:{repo}", f"review-{pr_number}",
                                     head_sha=payload.get("head_sha") or "HEAD"):
                return jsonify({"status": "skipped", "reason": "duplicate"})
            executor.submit(_process_pr, provider, payload)
            return jsonify({"status": "accepted", "reason": "review requested"})

        elif event_type in ("review_approved", "review_rejected"):
            # Co-approval auto-merge was removed. Route preserved so
            # Gitea/BB DC don't see 404s on deliveries from existing
            # webhook configurations.
            return jsonify({"status": "ignored", "reason": "no action taken"})

        elif event_type in ("comment", "diff_comment"):
            pr_number = payload.get("pr_number")
            comment_body = payload.get("comment_body", "")
            comment_user = payload.get("comment_user", "")
            comment_id = payload.get("comment_id")
            # Self-comment check
            try:
                raven_user = provider.get_authenticated_user()
            except Exception:
                raven_user = ""
            if raven_user and comment_user.lower() == raven_user.lower():
                return jsonify({"status": "skipped", "reason": "own comment"})
            # Bot-author filter — same heuristic push/PR events use. On BB DC
            # Raven auto-replies to any in-thread reply (no @mention needed),
            # so another auto-responding bot (or a second Raven under a
            # different name) would loop: each bot reply is a fresh comment_id
            # the 30s dedup never catches, and every iteration is a paid AI
            # call. Skip bot-authored comments before any dispatch decision.
            if _is_bot_author(comment_user):
                return jsonify({"status": "skipped", "reason": "bot author"})
            # Mention check — accepts @user or @"user.with.dots" (BB DC wraps
            # usernames containing dots in double quotes inside comment.text).
            # Mention check: does the comment tag Raven by its account
            # username OR a configured display name (default "Raven")?
            # Recognized in BOTH modes — the mention-only switch governs only
            # the untagged-thread-reply behaviour, not which tags count.
            # Gated on raven_user being resolved: if identity lookup failed,
            # don't act (fail-safe, as before).
            is_mention = bool(raven_user) and _comment_tags_raven(
                comment_body, [raven_user] + _mention_names())
            # Mention-only mode (RAVEN_REPLY_REQUIRE_MENTION): reply ONLY when
            # explicitly tagged; never on an untagged in-thread reply. Default
            # (unset) keeps the current behaviour (a tag OR any thread reply).
            # Read at request time so it's tunable without a restart.
            require_mention = os.environ.get(
                "RAVEN_REPLY_REQUIRE_MENTION", "").lower() in ("1", "true", "yes")
            parent_comment_id = payload.get("parent_comment_id")
            if require_mention:
                if not is_mention:
                    inc("raven_responses_skipped_total",
                        {"reason": "no_mention", "repo": repo})
                    return jsonify({"status": "ignored",
                                    "reason": "not tagged (mention-only mode)"})
            elif not is_mention and not parent_comment_id:
                return jsonify({"status": "ignored", "reason": "not directed at Raven"})
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            # Include ``comment_version`` in the dedup key when the provider
            # supplies it (BB DC bumps it on every edit). Lets a user edit a
            # comment to add ``@raven`` and have the edit trigger a reply,
            # while the original add's webhook (different version) doesn't
            # double-process. Gitea doesn't surface a version so dedup falls
            # back to the bare comment_id, preserving existing behavior.
            comment_version = payload.get("comment_version")
            dedup_suffix = (
                f"comment-{comment_id}-v{comment_version}"
                if comment_version is not None
                else f"comment-{comment_id}"
            )
            if comment_id and _should_skip_duplicate(f"{provider.name}:{repo}", dedup_suffix):
                return jsonify({"status": "skipped", "reason": "duplicate"})
            # Per-PR reply circuit breaker — backstop for reply loops the
            # bot-author name heuristic above misses. Checked last (after the
            # mention/thread + dedup gates) so it only counts replies we'd
            # actually dispatch, and records a timestamp only on dispatch.
            # Read at request time (not module load) so the limit is tunable
            # without a restart; a malformed value falls back to the default.
            try:
                reply_budget = int(os.environ.get("RAVEN_MAX_PR_REPLIES_PER_HOUR", "20"))
            except ValueError:
                reply_budget = 20
            if _reply_budget_exceeded(f"{provider.name}:{repo}#{pr_number}", reply_budget):
                logger.warning(
                    "Reply budget exceeded for %s PR #%s (>%s replies/hour) — "
                    "skipping to break a possible reply loop", repo, pr_number,
                    reply_budget,
                )
                inc("raven_responses_skipped_total", {"reason": "rate_limit", "repo": repo})
                return jsonify({"status": "skipped", "reason": "reply budget exceeded"})
            # The thread-author lookup (HTTP GET) runs inside _process_comment
            # so the webhook always returns 200 promptly — slow provider APIs
            # can't stall webhook delivery or trigger retries. The worker
            # decides whether to actually respond; if the thread doesn't
            # contain Raven, it quietly exits without posting anything.
            payload["_is_mention"] = bool(is_mention)
            executor.submit(_process_comment, provider, payload)
            return jsonify({"status": "accepted", "reason": "responding to comment"})

        else:
            logger.debug("Ignoring unsupported event: %s", event_type)
            return jsonify({"status": "ignored", "event": event_type})

    return app


# ------------------------------------------------------------------ #
#  Background PR review                                               #
# ------------------------------------------------------------------ #

def _should_auto_add_reviewer(provider: GitProvider, repo_full_name: str,
                               pr_number: int) -> bool | None:
    """Return:
      * ``True``  — auto-add Raven.
      * ``False`` — don't auto-add; reviewer state confirmed (Raven
        already listed, or fill-gap mode and other reviewers present).
      * ``None``  — couldn't determine; an upstream call (auth, transport)
        failed. Caller should treat this as "conservatively don't
        auto-add" but log it distinctly so the operator sees the auth
        failure rather than thinking the PR legitimately has other
        reviewers.

    Behaviour depends on the ``RAVEN_REVIEW_MODE`` switch:

    * ``all`` (default): auto-add unless Raven is already a reviewer
      (or a requested reviewer). Every PR gets a Raven review.
    * ``gap``: auto-add only when the PR has no other reviewer and
      no other requested reviewer — the "fill-the-gap" mode.
    * ``advisory``: never auto-add — Raven posts a non-blocking
      recommendation comment instead of a formal review.

    Raven being already listed counts as "don't re-add" in all modes
    (keeps re-review triggers idempotent).
    """
    # Advisory mode never auto-adds: a listed Raven reviewer that doesn't
    # submit a formal approval would block the PR — the opposite of
    # "advisory only".
    if RAVEN_REVIEW_MODE == "advisory":
        return False
    try:
        raven_user = provider.get_authenticated_user().lower()
    except Exception as e:
        logger.warning("Could not resolve bot user for auto-add check on %s PR #%d: %s — service account may lack repo access",
                       repo_full_name, pr_number, e)
        return None
    try:
        existing = provider.get_pr_reviews(repo_full_name, pr_number)
    except Exception as e:
        logger.warning("Could not list reviews on %s PR #%d: %s — service account may lack repo access",
                       repo_full_name, pr_number, e)
        return None
    try:
        requested = provider.get_pr_requested_reviewers(repo_full_name, pr_number)
    except Exception as e:
        logger.warning("Could not list requested reviewers on %s PR #%d: %s — service account may lack repo access",
                       repo_full_name, pr_number, e)
        return None

    raven_already_listed = any(
        ((r.get("user") or {}).get("login") or "").lower() == raven_user
        for r in existing
    ) or any(
        (login or "").lower() == raven_user for login in requested
    )
    if raven_already_listed:
        return False

    if RAVEN_REVIEW_MODE == "all":
        return True

    other_present = any(
        ((r.get("user") or {}).get("login") or "").lower() not in ("", raven_user)
        for r in existing
    ) or any(
        (login or "").lower() not in ("", raven_user) for login in requested
    )
    return not other_present


def _is_sole_reviewer(provider: GitProvider, repo_full_name: str, pr_number: int) -> bool:
    """Auto-merge gate: True when Raven is the only reviewer on the PR —
    no submitted reviews and no requested reviewers other than Raven
    itself. Raven appears in the requested list whenever it auto-added
    itself or a human re-requested its review; neither represents
    another reviewer waiting to weigh in, so it is filtered out
    (case-insensitive) — without that filter every self-requested PR
    would be falsely classified as "has other reviewers" and never merge.

    Shared by the push flow (_process_pr) and the comment flow
    (_process_comment) so no merge-dispatch path can omit the gate.
    Fail closed: any provider error returns False — if reviewer state
    can't be verified, don't auto-merge (a human can still merge
    manually).
    """
    try:
        raven_user_lc = (provider.get_authenticated_user() or "").lower()
    except Exception as e:
        logger.warning("Could not verify reviewer state on PR #%d (get_authenticated_user failed: %s) — leaving open without auto-merge",
                       pr_number, e)
        return False
    try:
        other_reviews = [r for r in provider.get_pr_reviews(repo_full_name, pr_number)
                         if ((r.get("user") or {}).get("login") or "").lower() != raven_user_lc]
    except Exception as e:
        logger.warning("Could not verify reviewer state on PR #%d (get_pr_reviews failed: %s) — leaving open without auto-merge",
                       pr_number, e)
        return False
    try:
        requested = [u for u in provider.get_pr_requested_reviewers(repo_full_name, pr_number)
                     if (u or "").lower() != raven_user_lc]
    except Exception as e:
        logger.warning("Could not verify requested-reviewers on PR #%d (get_pr_requested_reviewers failed: %s) — leaving open without auto-merge",
                       pr_number, e)
        return False
    if other_reviews or requested:
        logger.info("PR #%d has other reviewers — leaving open for human review", pr_number)
        return False
    return True


def _maybe_dispatch_cached_merge(provider: GitProvider, repo_full_name: str,
                                 pr_number: int, pr_title: str, pr_url: str,
                                 head_sha: str | None = None,
                                 current_hashes: dict[str, str] | None = None) -> bool:
    """Dispatch auto-merge from a CACHED approve verdict, without a fresh
    AI review pass. Returns True when a merge was dispatched to the
    CI-wait pool, False on any decline.

    Why this exists (TODO review-ops entry, 2026-06-12): merge dispatch
    used to happen only at the tail of a review pass (_process_pr) or a
    comment-driven verdict flip (_process_comment). Four wedge variants
    observed in one day each left an approved PR unmergeable with no
    in-band recovery — most directly the no-changes skip in _process_pr,
    which returned before any merge logic even when the cache held a
    standing approval (PR #161).

    Every gate fails CLOSED, and a dispatch reuses the SAME
    ``_safe_do_merge`` path as the review flows, so the CI gate and the
    force-push head-SHA recheck still apply downstream. Gates, in order:

      * advisory mode never merges (matches both existing dispatch paths);
      * ``head_sha`` must be usable — when the caller supplies one it must
        predate the diff that produced ``current_hashes`` (the webhook
        payload SHA qualifies); when absent AND the helper is about to
        fetch the diff itself, it fetches the SHA *first* so the pinned
        SHA can never be newer than the hashed diff. A push landing in
        between then fails either the hash check here or ``_do_merge``'s
        recheck — both in the safe direction. The ``"HEAD"`` sentinel is
        rejected rather than re-fetched: a fresh SHA would postdate the
        already-computed hashes and reopen the stale-approval race;
      * cache entry must exist (missing → unverifiable → decline),
        verdict must be ``approve``, ``coverage_gap_files`` must be empty;
      * the cached per-file diff hashes must EXACTLY match hashes of the
        current head's diff — the cached approval must describe the code
        being merged, not some prior commit (stale-approval wedge);
      * PR must be open (mirrors _process_comment's fail-closed state gate);
      * Raven must be the sole reviewer (shared ``_is_sole_reviewer`` gate).

    Outcomes land in ``raven_cached_merge_dispatch_total{outcome,repo}``
    as ``dispatched`` / ``declined_<reason>``.
    """
    def _decline(reason: str) -> bool:
        inc("raven_cached_merge_dispatch_total",
            {"outcome": f"declined_{reason}", "repo": repo_full_name})
        return False

    pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"

    if RAVEN_REVIEW_MODE == "advisory":
        logger.info("PR #%d: advisory mode — cached merge dispatch does not apply",
                    pr_number)
        return _decline("advisory_mode")

    if not head_sha or head_sha == "HEAD":
        if current_hashes is not None:
            # Hashes were computed from a diff we didn't fetch — fetching
            # a SHA now could pin a commit newer than that diff. Decline.
            logger.warning("PR #%d: no usable head SHA for cached merge dispatch "
                           "— declining (fail-closed)", pr_number)
            return _decline("no_head_sha")
        try:
            head_sha = provider.get_pr_head_sha(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("PR #%d: could not fetch head SHA for cached merge "
                           "dispatch: %s — declining (fail-closed)", pr_number, e)
            return _decline("no_head_sha")
        if not head_sha or head_sha == "HEAD":
            logger.warning("PR #%d: empty head SHA for cached merge dispatch "
                           "— declining (fail-closed)", pr_number)
            return _decline("no_head_sha")

    if current_hashes is None:
        try:
            clean_diff = _strip_lockfiles_and_binaries(
                provider.fetch_pr_diff(repo_full_name, pr_number))
            current_hashes = {
                f: hashlib.sha256(c.encode()).hexdigest()
                for f, c in split_diff_by_file(clean_diff)
            }
        except Exception as e:
            logger.warning("PR #%d: could not fetch diff for cached merge "
                           "dispatch: %s — declining (fail-closed)", pr_number, e)
            return _decline("diff_fetch_failed")

    with _previous_diffs_lock:
        entry = _previous_diffs.get(pr_key)
        if entry is None:
            logger.info("PR #%d: no cached review state — skipping cached merge "
                        "dispatch (fail-closed)", pr_number)
            return _decline("no_cache_entry")
        if entry.verdict != "approve":
            logger.info("PR #%d: cached verdict is %s — not dispatching cached merge",
                        pr_number, entry.verdict or "none")
            return _decline("verdict_not_approve")
        if entry.coverage_gap_files:
            logger.warning("PR #%d: cached review has unreviewed files (%s) — "
                           "skipping cached merge dispatch",
                           pr_number, ", ".join(entry.coverage_gap_files))
            return _decline("coverage_gap")
        if entry.hashes != current_hashes:
            logger.info("PR #%d: cached approval does not describe the current "
                        "head (per-file diff hashes diverge) — skipping cached "
                        "merge dispatch", pr_number)
            return _decline("hash_mismatch")
        cached_summary = entry.summary or ""
        remaining_findings = [f for fl in entry.findings.values() for f in fl]

    # PR state gate — fail closed, mirroring _process_comment's mutation
    # gate. The existing review-pass dispatch sites reach merge only from
    # open-PR webhook events; cached dispatch can fire from a stale cache
    # entry, so verify explicitly.
    try:
        pr_state = provider.get_pr_state(repo_full_name, pr_number)
    except Exception as e:
        logger.warning("get_pr_state failed for %s: %s — skipping cached merge "
                       "dispatch (fail-closed)", pr_key, e)
        return _decline("pr_state_unverifiable")
    if pr_state != "open":
        logger.info("PR %s state=%s — skipping cached merge dispatch",
                    pr_key, pr_state)
        return _decline("pr_not_open")

    # Same sole-reviewer gate as both existing dispatch paths; it logs
    # its own decline detail and fails closed on provider errors.
    if not _is_sole_reviewer(provider, repo_full_name, pr_number):
        return _decline("not_sole_reviewer")

    # Synthesize a review dict from residual cache state for notify
    # payloads inside _do_merge — same shape _process_comment builds.
    synthetic_review = {
        "approve": True,
        "severity": _max_severity_from_findings(remaining_findings),
        "summary": cached_summary,
        "findings": remaining_findings,
    }
    merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
    logger.info("PR #%d: dispatching auto-merge from cached approve verdict "
                "(head %s)", pr_number, head_sha[:8])
    inc("raven_cached_merge_dispatch_total",
        {"outcome": "dispatched", "repo": repo_full_name})
    fut = ci_wait_executor.submit(_safe_do_merge, provider, repo_full_name,
                                  pr_number, pr_title, pr_url, synthetic_review,
                                  head_sha, merge_strategy)
    fut.add_done_callback(functools.partial(_log_future_exception, repo=repo_full_name))
    return True


# Max carried findings offered to the model for re-validation in one
# incremental review. The drop-or-keep block bypasses the PR-context
# budgets (it's not conversation), so an unbounded carried set would
# inflate every incremental prompt — and on a small-context backend
# overflow fails the whole review. Overflow findings (lowest severity
# first) are carried verbatim instead, the pre-re-validation behavior.
RAVEN_CARRIED_REVALIDATION_MAX = max(
    int(os.environ.get("RAVEN_CARRIED_REVALIDATION_MAX", "20")), 1)


def _is_coverage_gap_marker(finding: dict, gap_files: set[str]) -> bool:
    """True when ``finding`` is a coverage-gap ⚠️ marker for one of
    ``gap_files``. Primary check is the structural ``gap_marker`` flag
    set at creation (``reviewer.review_diff``). The shape heuristic —
    gap filename in ``file``, no ``line``, message prefixed "⚠️", all
    three so a real (model-raised) finding on a gap file is never
    mistaken for the marker — remains as a fallback for markers cached
    before the flag existed. Used to keep markers out of the
    carried-findings drop-or-keep set: their lifecycle is the per-file
    gap carry (clears when the gap file changes), and the model must
    not get a path to drop an active gap signal."""
    if finding.get("gap_marker"):
        return True
    return (
        finding.get("file") in gap_files
        and not finding.get("line")
        and str(finding.get("message", "")).startswith("⚠️")
    )


def _process_pr(provider: GitProvider, payload: dict) -> None:
    """Review a PR in a background thread. All exceptions caught here."""
    repo_full_name = None
    pr_number = None
    pr_key = None
    try:
        repo_full_name = payload["repo"]
        pr_number = payload["pr_number"]
        pr_title = payload.get("pr_title") or f"PR #{pr_number}"
        pr_url = payload.get("pr_url", "")
        head_sha = payload.get("head_sha") or "HEAD"

        # Skip if another thread is already processing this PR. Guards
        # against the dedup window (30s) being shorter than the review
        # duration — a race that would otherwise cause two concurrent
        # reviews to fight over the findings cache and merge decision.
        pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"
        with _in_progress_lock:
            if pr_key in _in_progress_prs:
                logger.info("PR %s already being reviewed — skipping concurrent run", pr_key)
                inc("raven_reviews_skipped_total", {"reason": "in_progress", "repo": repo_full_name})
                pr_key = None  # don't clear the entry in finally
                return
            _in_progress_prs.add(pr_key)

        # Auto-add Raven as a reviewer only when there are no other
        # reviewers or requested reviewers on the PR — the "fill the
        # gap" case where no human is assigned. If humans are already
        # involved, stay out of the way: they can still pull Raven in
        # by @mentioning or manually adding Raven as a reviewer, which
        # fires the review_requested webhook and triggers _process_pr
        # the same as any other review trigger. (Raven still runs the
        # review regardless; it just doesn't claim the reviewer slot.)
        # Best-effort — failures on the check or the add don't block
        # the review itself.
        try:
            should_add = _should_auto_add_reviewer(provider, repo_full_name, pr_number)
            if should_add is True:
                provider.add_self_as_reviewer(repo_full_name, pr_number)
            elif should_add is False:
                logger.info("PR #%d already has other reviewers — not auto-adding Raven", pr_number)
            else:
                # None = couldn't verify reviewer state due to API error;
                # the helper already warned with the auth-failure detail.
                # Stay conservative and skip auto-add; the review itself
                # still runs below.
                logger.info("PR #%d — couldn't verify reviewer state; not auto-adding Raven", pr_number)
        except Exception as e:
            logger.warning("Failed to add Raven as reviewer on PR #%d: %s", pr_number, e)
            inc("raven_errors_total", {"type": "self_reviewer_failed", "repo": repo_full_name})

        # Reviewer-status gate: review only runs if Raven is listed as
        # a reviewer (or requested reviewer) on this PR. Combined with
        # the auto-add step above, this means:
        #   * RAVEN_REVIEW_MODE=all + any PR → auto-added → gate passes.
        #   * RAVEN_REVIEW_MODE=gap + no humans → auto-added → gate passes.
        #   * RAVEN_REVIEW_MODE=gap + humans → no auto-add → gate
        #     fails unless a human manually added Raven earlier.
        #   * RAVEN_REVIEW_MODE=advisory → gate bypassed (advisory mode
        #     engages on every webhook regardless of reviewer assignment).
        if RAVEN_REVIEW_MODE != "advisory":
            try:
                raven_user_lc = (provider.get_authenticated_user() or "").lower()
                reviews_for_gate = provider.get_pr_reviews(repo_full_name, pr_number)
                requested_for_gate = provider.get_pr_requested_reviewers(repo_full_name, pr_number)
            except Exception as e:
                logger.warning("Could not verify Raven's reviewer status on PR #%d — proceeding anyway: %s",
                               pr_number, e)
                raven_user_lc = ""
                reviews_for_gate = []
                requested_for_gate = []

            if raven_user_lc:
                raven_listed = any(
                    ((r.get("user") or {}).get("login") or "").lower() == raven_user_lc
                    for r in reviews_for_gate
                ) or any(
                    (login or "").lower() == raven_user_lc for login in requested_for_gate
                )
                if not raven_listed:
                    logger.debug("PR #%d — Raven not a reviewer, skipping review", pr_number)
                    inc("raven_reviews_skipped_total", {"reason": "not_reviewer", "repo": repo_full_name})
                    return

        # Fetch diff
        diff = provider.fetch_pr_diff(repo_full_name, pr_number)

        # Fetch repo context — both CLAUDE.md and the .claude/rules/
        # directory (if either exist). Each is optional; any fetch
        # failure degrades to empty content rather than blocking the
        # review.
        #
        # BOTH are read from the PR *base* ref so the content carries
        # the same trust as the review prompt itself — a change had to
        # land via a PR Raven reviewed without the new content applied.
        # Fetching either from PR head would let an author add
        # ``CLAUDE.md`` / ``.claude/rules/policy.md`` saying "approve
        # SQL concatenation" alongside hostile code, biasing Raven's
        # review of that same PR. The reviewer renders both inside
        # ``<repo_policy_TAGID>`` blocks (the trusted tier from the
        # prompt preamble); see ``reviewer._build_trust_preamble``.
        base_ref = payload.get("base_ref") or "HEAD"
        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref=base_ref)
        except Exception as e:
            # 404 (missing file) returns "" without raising; reaching this
            # except means an auth/transport/server-side failure that the
            # operator probably wants to see. Warn instead of debug.
            logger.warning("CLAUDE.md fetch for %s@%s failed (review proceeds without repo context): %s",
                           repo_full_name, base_ref, e)
        # NOTE: rule + prompt-override fetches deferred to after the
        # no-changes-skip path below so a re-review on an unchanged diff
        # doesn't incur those network calls only to return early.

        # Guard: empty diff after stripping lockfiles/binaries
        clean_diff = _strip_lockfiles_and_binaries(diff)
        if not clean_diff.strip():
            provider.post_pr_comment(repo_full_name, pr_number,
                "🦅 **Raven Review**\n\nEmpty diff after stripping lockfiles/binaries — skipping review.")
            inc("raven_reviews_skipped_total", {"reason": "empty_diff", "repo": repo_full_name})
            return

        # Incremental review: only review files that changed since last review
        pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"
        file_chunks = {f: c for f, c in split_diff_by_file(clean_diff)}
        current_hashes = {f: hashlib.sha256(c.encode()).hexdigest() for f, c in file_chunks.items()}
        now = time.time()
        with _previous_diffs_lock:
            cached = _previous_diffs.get(pr_key)
            if cached:
                previous_hashes = cached.hashes
                cached_findings = cached.findings
            else:
                previous_hashes = {}
                cached_findings = {}

        changed_files = {f for f, h in current_hashes.items() if previous_hashes.get(f) != h}
        removed_files = previous_hashes.keys() - current_hashes.keys()
        if previous_hashes and not changed_files and not removed_files:
            logger.info("PR #%d re-review: no files changed since last review — skipping", pr_number)
            inc("raven_reviews_skipped_total", {"reason": "no_changes", "repo": repo_full_name})
            # A re-trigger with zero changed files is exactly the recovery
            # case for a standing cached approval that never merged (the
            # merge-dispatch wedges from the PR #160-162 rollout): returning
            # blind here left the PR unmergeable with no in-band recovery.
            # Attempt the gated cached dispatch instead. ``head_sha`` is the
            # webhook payload's — it predates the diff fetch above, so the
            # pinned SHA can never be newer than the hashed diff (see the
            # helper's docstring for the race direction).
            _maybe_dispatch_cached_merge(provider, repo_full_name, pr_number,
                                         pr_title, pr_url, head_sha=head_sha,
                                         current_hashes=current_hashes)
            return

        is_incremental = False
        unchanged_files: list[str] = []
        if previous_hashes and removed_files:
            # Files were removed — do a full review to clear stale findings
            logger.info("PR #%d files removed since last review — full re-review", pr_number)
            review_diff_text = clean_diff
        elif previous_hashes and changed_files:
            # Incremental: rebuild diff from only changed file chunks.
            # The unchanged files are declared to review_diff so the
            # prompt discloses the delta scope — without it the model
            # judges PR-level claims ("the implementation is missing
            # from this PR") from files it was never shown.
            is_incremental = True
            unchanged_files = sorted(set(current_hashes) - changed_files)
            review_diff_text = "".join(file_chunks[f] for f in sorted(changed_files))
            logger.info("PR #%d incremental review: %d/%d files changed", pr_number, len(changed_files), len(current_hashes))
        else:
            review_diff_text = clean_diff

        # Fetch full file contents for all PR files (cross-file context even on incremental)
        file_contents, omitted_files = _fetch_changed_files(provider, repo_full_name, head_sha, clean_diff)

        # Fetch PR description + recent comments so author intent and
        # prior-reviewer context reach the prompt. Best-effort: any API
        # failure degrades to empty context rather than blocking the review.
        pr_description = ""
        try:
            pr_description = provider.get_pr_description(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("Failed to fetch PR #%d description: %s", pr_number, e)

        pr_comments: list[dict] = []
        try:
            pr_comments = provider.get_pr_comments(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("Failed to fetch PR #%d comments: %s", pr_number, e)

        # Resolve the bot's own login so the prompt-context filter can
        # strip Raven's prior review comments (otherwise they re-enter
        # the prompt as if they were new developer context). The login
        # is deployment-specific (not always "raven"), so we ask the
        # provider. get_authenticated_user() is cached — calling it
        # here and again later for the sole-reviewer check is free.
        bot_user = ""
        try:
            bot_user = provider.get_authenticated_user()
        except Exception as e:
            logger.warning("Failed to resolve bot user for PR #%d context filter: %s", pr_number, e)

        # Fetch rules + prompt override now (after the no-changes-skip
        # path returned). Doing it here keeps re-reviews of unchanged
        # diffs from incurring rules/list_directory + per-file fetches
        # only to bail before the AI call.
        rules = _fetch_rules(provider, repo_full_name, base_ref)
        review_prompt_override = _fetch_prompt_override(
            provider, repo_full_name, base_ref, "review",
        )

        # User-resolved-comment filter, pass 1 (pre-review). Findings
        # whose backing inline comment the developer marked resolved via
        # the platform UI (Gitea ≥1.24 /resolve; BB DC "Resolve thread"
        # → threadResolved=true) are excluded from the carry-forward set
        # — the model must never be asked to drop-or-keep a finding the
        # developer already dismissed. Symmetric to the AI-driven
        # retraction in _process_comment: that flow is Raven resolving
        # on the AI's behalf; this one is Raven respecting the user's
        # direct dismissal. A second fetch after the AI call (below)
        # catches resolutions landing during the multi-minute review.
        #
        # NOTE: nothing is written to the cache here. ALL cache effects
        # of this pass — user-resolved drops, model drops — are applied
        # atomically at the post-submit cache write, so a failed
        # submit_review leaves the cache matching the standing platform
        # review (the comment-flow all-retracted backstop counts cached
        # findings; a premature wipe could synthesize flip-to-approve).
        resolved_ids: set = set()
        if is_incremental and cached_findings:
            try:
                resolved_ids = provider.get_resolved_comment_ids(repo_full_name, pr_number)
            except Exception as e:
                logger.warning(
                    "get_resolved_comment_ids for PR #%d failed: %s — proceeding without filter",
                    pr_number, e,
                )

        # Build the carry-forward set BEFORE the review call so the
        # candidates can ride into the prompt's drop-or-keep block.
        # Coverage-gap ⚠️ markers are split out and NEVER offered to the
        # model: they keep their own lifecycle (carried while the gap
        # file is unchanged, dropped when it changes — see the gap-carry
        # block below), and re-validation must not become a path to
        # erase an active gap signal — same reason the consolidation
        # pass excludes them. File-less ('' bucket) findings ARE
        # candidates: they used to be carried unconditionally on every
        # pass (immortal — one could pin the verdict severity forever);
        # drop-or-keep is their expiry mechanism.
        carried_candidates: list[dict] = []
        carried_markers: list[dict] = []
        user_resolved_count = 0
        if is_incremental and cached_findings:
            carried_gap_files = set(cached.coverage_gap_files or []) if cached else set()
            for fname in current_hashes:
                if fname not in changed_files:
                    for f in cached_findings.get(fname, []):
                        if f.get("comment_id") in resolved_ids:
                            user_resolved_count += 1
                        elif _is_coverage_gap_marker(f, carried_gap_files):
                            carried_markers.append(f)
                        else:
                            carried_candidates.append(f)
            for f in cached_findings.get("", []):
                if f.get("comment_id") in resolved_ids:
                    user_resolved_count += 1
                else:
                    carried_candidates.append(f)

        # Cap the re-validation set offered to the model (top-K by
        # severity, original order preserved within). The block bypasses
        # the PR-context budgets, so an unbounded carried set would
        # inflate every incremental prompt — and overflow a small-
        # context backend into a whole-review parse error. Overflow
        # findings are carried verbatim (the fail-safe behavior); the
        # cap is applied HERE, before the call, so the prompt's
        # carry_id ↔ candidate index mapping stays aligned.
        revalidation_candidates = carried_candidates
        overflow_candidates: list[dict] = []
        if len(carried_candidates) > RAVEN_CARRIED_REVALIDATION_MAX:
            by_sev = sorted(
                range(len(carried_candidates)),
                key=lambda i: -SEVERITY_ORDER.get(
                    carried_candidates[i].get("severity", "low"), 0),
            )
            top = set(by_sev[:RAVEN_CARRIED_REVALIDATION_MAX])
            revalidation_candidates = [
                f for i, f in enumerate(carried_candidates) if i in top]
            overflow_candidates = [
                f for i, f in enumerate(carried_candidates) if i not in top]
            logger.info(
                "PR #%d: %d carried findings exceed re-validation cap %d — "
                "%d carried verbatim without re-validation",
                pr_number, len(carried_candidates),
                RAVEN_CARRIED_REVALIDATION_MAX, len(overflow_candidates),
            )

        # Run review
        with Timer("raven_review_duration_seconds", {"repo": repo_full_name}):
            review = review_diff(
                review_diff_text, repo_full_name,
                claude_md=claude_md, file_contents=file_contents,
                omitted_files=omitted_files,
                pr_title=pr_title, pr_description=pr_description,
                pr_comments=pr_comments, bot_user=bot_user,
                rules=rules,
                prompt_override=review_prompt_override,
                is_incremental=is_incremental,
                unchanged_files=unchanged_files,
                carried_findings=revalidation_candidates or None,
            )
        # Save original findings before merging carried ones (used for cache write)
        fresh_findings = list(review.get("findings", []))

        # Apply the model's drop-or-keep answer to the carried set —
        # LOCALLY only; the cache is untouched until the post-submit
        # write. ``dropped_carried`` lists the carry_ids (indices into
        # revalidation_candidates) this push clearly resolved. Drop is
        # the explicit action: a missing key, an empty list, or ids
        # that don't map to any candidate (incl. booleans — bool
        # subclasses int) mean "drop nothing", so schema echo /
        # truncation / chunked-path absence all keep every carried
        # finding. Degraded is better than dropping findings on error.
        dropped_raw = review.pop("dropped_carried", None)
        dropped_findings: list[dict] = []
        if revalidation_candidates and isinstance(dropped_raw, list) and dropped_raw:
            if all(isinstance(i, int) and not isinstance(i, bool)
                   and 0 <= i < len(revalidation_candidates) for i in dropped_raw):
                drop_set = set(dropped_raw)
                dropped_findings = [
                    f for i, f in enumerate(revalidation_candidates) if i in drop_set]
                revalidation_candidates = [
                    f for i, f in enumerate(revalidation_candidates) if i not in drop_set]
                logger.info(
                    "PR #%d: re-validation dropped %d/%d carried finding(s)",
                    pr_number, len(dropped_findings),
                    len(dropped_findings) + len(revalidation_candidates),
                )
                add("raven_carried_findings_dropped_total",
                    len(dropped_findings), {"repo": repo_full_name})
            else:
                logger.warning(
                    "PR #%d: dropped_carried ids %r don't map to the %d "
                    "re-validation candidate(s) — keeping all carried findings",
                    pr_number, dropped_raw, len(revalidation_candidates),
                )

        kept_candidates = revalidation_candidates + overflow_candidates

        # User-resolved-comment filter, pass 2 (post-review). The AI
        # call takes minutes; a finding the developer resolves DURING it
        # would otherwise be re-posted by the carry merge and re-tagged
        # with a fresh comment_id (the standing re-tagging below), after
        # which the next pass's filter — matching the OLD id — could
        # never catch it: the user's resolution would be permanently
        # lost. One extra provider GET per incremental review.
        if is_incremental and kept_candidates:
            try:
                resolved_post = provider.get_resolved_comment_ids(repo_full_name, pr_number)
            except Exception as e:
                logger.warning(
                    "post-review get_resolved_comment_ids for PR #%d failed: %s — using pre-review set",
                    pr_number, e,
                )
                resolved_post = set()
            if resolved_post:
                resolved_ids |= resolved_post
                before_count = len(kept_candidates)
                kept_candidates = [
                    f for f in kept_candidates
                    if f.get("comment_id") not in resolved_ids
                ]
                user_resolved_count += before_count - len(kept_candidates)
        if user_resolved_count:
            logger.info(
                "PR #%d: dropped %d user-resolved finding(s) from carry-forward",
                pr_number, user_resolved_count,
            )
            add("raven_user_resolved_findings_dropped_total",
                user_resolved_count, {"repo": repo_full_name})

        # Merge carried findings from unchanged files into the review.
        # Dedupe first: the prompt forbids copying carried findings into
        # `findings`, but a model may restate one anyway — without this,
        # the restated copy posts a duplicate inline comment AND (naming
        # an unchanged file outside changed_files) lands a clone in the
        # '' cache bucket at the write below. Verbatim (file, line,
        # message) duplicates lose to the carried copy, which holds the
        # comment_id retraction needs.
        carried = kept_candidates + carried_markers
        if carried:
            carried_keys = {
                (f.get("file"), f.get("line"), f.get("message")) for f in carried
            }
            fresh_findings = [
                f for f in fresh_findings
                if (f.get("file"), f.get("line"), f.get("message")) not in carried_keys
            ]
            review["findings"] = fresh_findings + carried
            review["carried_count"] = len(carried)
            # Recompute severity across all findings
            max_sev = SEVERITY_ORDER.get(review["severity"], 0)
            for finding in carried:
                max_sev = max(max_sev, SEVERITY_ORDER.get(finding.get("severity", "low"), 0))
            review["severity"] = _SEVERITY_NAME.get(max_sev, "low")

        # Per-file sticky coverage gap. An incremental pass only
        # re-reviews CHANGED files, so an unchanged oversized file from
        # the prior review is still unreviewed — carry its gap forward.
        # But ONLY for unchanged files: once a gap file changes and is
        # re-reviewed, the fresh review's own gap list (which re-names
        # the file if it's still oversized/failed) is authoritative —
        # otherwise the gap could never clear for a live PR and the PR
        # would be blocked from approval for its entire lifetime. Both
        # sides key off the same split_diff_by_file filenames as
        # ``changed_files``/``current_hashes``, so membership lines up.
        # Full re-reviews (non-incremental) cover everything: fresh list
        # only.
        fresh_gap_files = set(review.get("coverage_gap_files") or [])
        carried_gap_files = (
            {f for f in (cached.coverage_gap_files or []) if f not in changed_files}
            if is_incremental and cached is not None else set()
        )
        effective_gap_files = sorted(fresh_gap_files | carried_gap_files)
        review["coverage_gap_files"] = effective_gap_files
        review["coverage_gap"] = bool(effective_gap_files)

        logger.info("PR #%d review: severity=%s summary=%s", pr_number, review["severity"], review["summary"][:80])
        inc("raven_reviews_total", {"severity": review["severity"], "repo": repo_full_name})

        # Guard: if review could not be parsed, post error comment, notify, bail — never auto-merge
        if review.get("_parse_error"):
            logger.warning("PR #%d review had parse error — skipping merge", pr_number)
            provider.post_pr_comment(repo_full_name, pr_number,
                "🦅 **Raven Review**\n\n⚠️ Could not parse review output — skipping auto-merge.")
            notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
                   link=pr_url, action="review_failed")
            inc("raven_errors_total", {"type": "parse_error", "repo": repo_full_name})
            return

        # Output-channel selection (RAVEN_REVIEW_OUTPUT):
        #   both     — summary body + inline comments
        #   summary  — summary body only (no inline)
        #   inline   — inline comments only; body trimmed to verdict +
        #              one-liner, but findings WITHOUT a postable file/line
        #              stay in the body so nothing is silently dropped.
        post_inline = RAVEN_REVIEW_OUTPUT in ("both", "inline")

        def _is_inline_postable(f: dict) -> bool:
            return bool(f.get("file")) and isinstance(f.get("line"), int) and f["line"] > 0

        # Submit formal review — must succeed before dismissing old reviews.
        # Verdict + inline comments are computed first; the inline-mode body
        # (below) needs to know whether anything was posted inline.
        approve_sev = os.environ.get("REVIEW_APPROVE_MAX_SEVERITY", "low")
        approve = severity_gte(approve_sev, review["severity"])
        # Coverage gap forces needs_work: a formal APPROVE is externally
        # visible (branch protection counts bot approvals; humans trust
        # "Raven approved") and must never post for code Raven didn't
        # see. Reachable in two ways the severity floor can't cover:
        # a sticky gap carried over a clean incremental pass (fresh
        # severity 'low', never floored), and
        # REVIEW_APPROVE_MAX_SEVERITY=high where the floor caps at an
        # approvable 'high'. Safe to force only because the gap CLEARS
        # once the named file re-reviews (per-file carry above).
        if approve and review.get("coverage_gap"):
            logger.info(
                "PR #%d: unreviewed files remain (%s) — forcing needs_work verdict",
                pr_number, ", ".join(review.get("coverage_gap_files") or []))
            approve = False
        default_emoji = "\U0001f7e1"
        inline_comments = [
            {
                "file": f["file"],
                "line": f["line"],
                "body": f"{SEVERITY_EMOJI.get(f.get('severity', 'low'), default_emoji)} **[{f.get('severity', 'low')}]** {f['message']}",
            }
            for f in review.get("findings", [])
            if _is_inline_postable(f)
        ] if post_inline else []

        # Build the summary body per output channel:
        #   both / summary → full review summary.
        #   inline         → NO recommendation comment. Inline-able findings
        #                    live on their lines; only findings with no
        #                    postable file/line (PR-wide notes and ⚠️
        #                    coverage-gap markers) get a MINIMAL body so they
        #                    aren't silently dropped. Empty body when there
        #                    are none — a clean PR posts just the verdict.
        if RAVEN_REVIEW_OUTPUT == "inline":
            leftover = [f for f in review.get("findings", [])
                        if not _is_inline_postable(f)]
            body = _format_inline_leftovers(leftover)
            # Never submit a content-less non-approve review: Gitea rejects an
            # empty body + no inline comments for a COMMENT / REQUEST_CHANGES
            # event. (A clean APPROVE with an empty body is fine and stays
            # intentionally silent.) Reachable only if the model returns a
            # blocking verdict with zero findings.
            if not body and not inline_comments and (
                RAVEN_REVIEW_MODE == "advisory" or not approve
            ):
                sev = review.get("severity", "low")
                body = (
                    f"🦅 **Raven** — {SEVERITY_EMOJI.get(sev, default_emoji)} "
                    f"**{sev.upper()}** — {review.get('summary') or 'changes requested'}"
                )
        else:
            body = _format_comment(
                review,
                mode="advisory" if RAVEN_REVIEW_MODE == "advisory" else "review",
            )
        # Pass comment_only conditionally via dict-spread so out-of-tree
        # providers running in non-advisory modes never see the new kwarg.
        # (Default in the ABC doesn't propagate to overriders.)
        advisory_kwargs = (
            {"comment_only": True} if RAVEN_REVIEW_MODE == "advisory" else {}
        )
        try:
            new_review = provider.submit_review(repo_full_name, pr_number, body,
                                                approve=approve, inline_comments=inline_comments,
                                                commit_id=head_sha,
                                                **advisory_kwargs)
        except Exception as e:
            logger.error("Failed to submit review on PR #%d: %s", pr_number, e)
            notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
                   link=pr_url, action="review_submit_failed")
            inc("raven_errors_total", {"type": "review_submit_failed", "repo": repo_full_name})
            return

        # Resolve the platform threads of carried findings the model
        # dropped — mirrors _process_comment's AI-driven retraction.
        # Without this, the dropped finding's inline comment stays open
        # forever (its comment_id is about to leave the cache, so no
        # future flow can resolve it); on BB DC — where
        # dismiss_previous_reviews is a no-op and merge checks may
        # require all comments resolved — the drop would permanently
        # block the very merge it enables. Best-effort: a failed resolve
        # logs and continues (the drop itself stands). Runs only after
        # submit_review succeeded, alongside the other cache effects.
        for f in dropped_findings:
            cid = f.get("comment_id")
            if cid is None:
                continue
            try:
                ok = provider.retract_finding(repo_full_name, pr_number, cid)
            except Exception as e:
                ok = False
                logger.warning(
                    "retract_finding for re-validation-dropped finding "
                    "(comment %s) on PR #%d failed: %s", cid, pr_number, e,
                )
            inc("raven_retractions_total",
                {"repo": repo_full_name, "result": "ok" if ok else "fail"})

        # Dismiss previous Raven reviews — only after new review is safely posted
        new_review_id = new_review.get("id") if isinstance(new_review, dict) else None
        if new_review_id is None:
            logger.warning("submit_review returned no id — skipping dismiss to avoid self-dismissal")
        else:
            try:
                bot_user = provider.get_authenticated_user()
                provider.dismiss_previous_reviews(repo_full_name, pr_number, bot_user,
                                                  exclude_id=new_review_id)
            except Exception as e:
                logger.warning("Failed to dismiss old reviews on PR #%d: %s", pr_number, e)
                inc("raven_errors_total", {"type": "dismiss_failed", "repo": repo_full_name})

        # Tag findings with comment_id from the provider's submit_review
        # return BEFORE building findings_map. _findings_by_file groups by
        # reference (no copy), so tagging here propagates to the cached
        # dict. Retraction (comment-thread-context feature) later matches
        # by comment_id to drop the right entry on a successful retract.
        #
        # CRITICAL: iterate review["findings"] (post-carry-forward merge)
        # — NOT fresh_findings — to mirror the inline_comments filter at
        # line ~776. On incremental reviews, review["findings"] also
        # contains carried findings whose previous IDs were dismissed
        # alongside the prior review (server.py:794 dismiss_previous_reviews).
        # submit_review re-posts them with fresh IDs; the carried-finding
        # dicts in cache get retagged with those new IDs via shared
        # references (carried = list of dict refs from cached_findings,
        # which are the same dicts as _previous_diffs[pr_key].findings).
        # Only findings actually submitted as inline comments can be tagged
        # with a comment_id. When inline output is suppressed (summary mode),
        # nothing was posted inline, so this stays empty and the length-match
        # guard below trivially passes (0 == 0) rather than warning.
        submitted_findings = [
            f for f in review.get("findings", [])
            if _is_inline_postable(f)
        ] if post_inline else []
        posted_inline = (new_review or {}).get("inline_comments") or []
        # Defensive: providers MUST return inline_comments aligned by
        # index with input (None for failed posts). If lengths diverge —
        # filter drift in either side, or a provider returning a different
        # shape — silently zipping would land comment_ids on the wrong
        # findings, breaking retraction. Skip tagging and warn.
        if len(submitted_findings) != len(posted_inline):
            logger.warning(
                "comment_id propagation length mismatch on PR #%d: %d "
                "submitted findings vs %d posted inline_comments — "
                "skipping comment_id tagging this round; retraction-by-id "
                "is unavailable for these findings until the next "
                "push-driven re-review.",
                pr_number, len(submitted_findings), len(posted_inline),
            )
        else:
            for f, p in zip(submitted_findings, posted_inline):
                if p.get("comment_id") is not None:
                    f["comment_id"] = p["comment_id"]

        # Cache diff + per-file findings for incremental re-reviews.
        # Use fresh_findings (pre-merge, deduped) to avoid duplicating
        # carried findings. ALL carry-forward drops from this pass —
        # model-dropped (identity) and user-resolved (comment_id) — are
        # applied HERE, atomically under the lock and only now that
        # submit_review has succeeded: a failed submit leaves the cache
        # matching the standing platform review. Filtering reads the
        # LIVE entry's findings, not the pre-review snapshot — the
        # comment flow can retract a finding (under _previous_diffs_lock)
        # while the review is in flight, and writing back the stale
        # snapshot would silently resurrect it.
        verdict = "approve" if approve else "needs_work"
        # Store the formatted review body (with severity + findings list)
        # so the comment-driven `## Your Prior Verdict` block shows the
        # AI substantive context — not just the one-line `summary` field.
        # Matches the shape the comment-revision path writes
        # (revise["body"], also a multi-paragraph body).
        cache_summary = body
        dropped_identity = {id(f) for f in dropped_findings}

        def _keep_carried(f: dict) -> bool:
            return (id(f) not in dropped_identity
                    and f.get("comment_id") not in resolved_ids)

        with _previous_diffs_lock:
            live_entry = _previous_diffs.get(pr_key)
            live_findings = (
                live_entry.findings if live_entry is not None else cached_findings
            )
            if is_incremental and cached_findings:
                findings_map = {
                    fname: [f for f in live_findings.get(fname, []) if _keep_carried(f)]
                    for fname in current_hashes if fname not in changed_files
                }
                fresh = _findings_by_file(fresh_findings, changed_files)
                # Carry forward file-less findings from previous review + new file-less ones
                fresh.setdefault("", []).extend(
                    f for f in live_findings.get("", []) if _keep_carried(f))
                findings_map.update(fresh)
            else:
                findings_map = _findings_by_file(fresh_findings, set(current_hashes.keys()))
            _previous_diffs[pr_key] = CacheEntry(
                timestamp=time.time(),
                hashes=current_hashes,
                findings=findings_map,
                verdict=verdict,
                summary=cache_summary,
                coverage_gap_files=list(review.get("coverage_gap_files") or []),
            )
        _evict_cache()
        _save_cache()

        # Add label
        try:
            provider.add_label_to_pr(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("Failed to add label: %s", e)
            inc("raven_errors_total", {"type": "label_failed", "repo": repo_full_name})

        # Advisory mode is done after submit_review + cache + label.
        # No formal verdict was registered, so the auto-merge gates
        # (reviewer-state checks + merge dispatch) don't apply and the
        # "review=REQUEST_CHANGES — leaving open" log below would be
        # factually wrong. Notify per channel severity and return.
        if RAVEN_REVIEW_MODE == "advisory":
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        if not approve:
            logger.info("PR #%d review=REQUEST_CHANGES — leaving open", pr_number)
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        # Coverage-gap merge gate — defense-in-depth. The verdict force
        # above already turns any coverage-gap review into needs_work,
        # so this branch should be unreachable on the approve path; it
        # stays as a second, independent guard so a future refactor of
        # the approve computation can't silently reopen the
        # unreviewed-code-auto-merges hole. A human can still merge
        # manually.
        if review.get("coverage_gap"):
            logger.warning(
                "PR #%d approved but parts of the diff were not reviewed "
                "(coverage gap) — leaving open without auto-merge", pr_number)
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        # Errors inside the gate used to propagate to the outer
        # try/except as "internal error" and clear dedup; the helper
        # fails closed instead (returns False on any provider error).
        if not _is_sole_reviewer(provider, repo_full_name, pr_number):
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        # Raven is the only reviewer — merge. Dispatched to the
        # CI-wait pool so this review thread is freed immediately;
        # otherwise the worker would sit in time.sleep for up to
        # ``CI_WAIT_TIMEOUT`` and starve other incoming webhooks.
        merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
        fut = ci_wait_executor.submit(_safe_do_merge, provider, repo_full_name, pr_number,
                                       pr_title, pr_url, review, head_sha, merge_strategy)
        fut.add_done_callback(functools.partial(_log_future_exception, repo=repo_full_name))

    except Exception as e:
        logger.error("Unhandled error processing PR: %s", e, exc_info=True)
        reason = _review_failure_reason(e)
        repo_label = repo_full_name or "unknown"
        inc("raven_errors_total", {"type": "unhandled", "repo": repo_label})
        # Classified failure metric so operators can alert per cause
        # (e.g. a spike in timeout vs auth) without scraping logs.
        inc("raven_review_failures_total", {"reason": reason, "repo": repo_label})
        # Clear dedup entry so webhook retries can re-attempt this PR.
        # Key format must match _should_skip_duplicate's — same head_sha
        # fallback ("HEAD") as dispatch so the SHA-aware suffix aligns.
        if repo_full_name is not None and pr_number is not None:
            dedup_sha = payload.get("head_sha") or "HEAD"
            key = f"{provider.name}:{repo_full_name}#{pr_number}@{dedup_sha}"
            with _recent_prs_lock:
                _recent_prs.pop(key, None)
            try:
                # Classified, actionable comment — names the cause and the
                # next step. Static templates only (no str(e)), so no
                # credential-bearing exception text can leak.
                provider.post_pr_comment(repo_full_name, pr_number,
                    _failure_comment(reason))
            except Exception:
                pass
    finally:
        # Release the in-progress lock for this PR. The sentinel (pr_key
        # set back to None) means we never acquired it — skip the clear.
        #
        # Note: this discard happens BEFORE the auto-merge dispatched at
        # line ~1045 to ``ci_wait_executor`` completes. A new push during
        # CI wait can therefore trigger a fresh ``_process_pr`` while the
        # old merge is still polling — that's intentional. The new run
        # posts its own review + dismisses the old one + dispatches its
        # own merge; the OLD merge's ``head_sha`` recheck in
        # ``_do_merge`` (line ~1771) catches the SHA drift and skips
        # cleanly. The window is safe by design.
        if pr_key is not None:
            with _in_progress_lock:
                _in_progress_prs.discard(pr_key)


# ------------------------------------------------------------------ #
#  Comment response                                                    #
# ------------------------------------------------------------------ #

def _process_comment(provider: GitProvider, payload: dict) -> None:
    """Respond to a comment directed at Raven in a background thread.

    The webhook handler dispatches any comment that *could* be for Raven —
    either an @mention or a reply inside a thread. This worker confirms the
    thread case with a provider API call (kept off the webhook hot path),
    fetches active-thread + prior-verdict context for the AI, posts the
    reply, and optionally retracts invalidated findings and revises the
    overall verdict (which may dispatch auto-merge).
    """
    repo_full_name = payload["repo"]
    pr_number = payload.get("pr_number")
    comment_id = payload.get("comment_id")
    try:
        comment_body = payload.get("comment_body", "")
        file_path = payload.get("file_path", "") or ""
        line = payload.get("line") or 0
        parent_comment_id = payload.get("parent_comment_id")
        is_mention = bool(payload.get("_is_mention"))

        # Active thread fetched ONCE up-front, reused for:
        #   - Mention/author gating (non-mention path needs to verify Raven
        #     is in the thread before responding).
        #   - AI prompt context (`## Active Thread` block).
        #   - Retraction ID validation (we only retract IDs that appear in
        #     the fetched thread).
        # Seed selection: parent_comment_id when set (BB DC reply); else
        # the trigger comment's own id when the trigger is an inline-diff
        # comment (Gitea's group-by-(path,position) returns the same thread
        # from any member, so we use the trigger id as the seed). General
        # @mentions on flat issue comments have no thread → []. Failure is
        # best-effort: thread stays empty and the reply still goes out.
        thread: list[dict] = []
        seed_id = parent_comment_id or (comment_id if file_path else None)
        if seed_id:
            try:
                thread = provider.get_comment_thread(
                    repo_full_name, pr_number, seed_id,
                )
            except Exception as e:
                # Warning, not debug: an exception here means a real API
                # failure (auth, transport, server error). An empty
                # thread for a brand-new comment is a different code
                # path (no exception, just []) — that one stays silent.
                logger.warning("Thread fetch failed for PR #%s seed=%s: %s",
                               pr_number, seed_id, e)

        # Root-anchor fallback. A follow-up REPLY inside an inline thread
        # carries no anchor of its own — the file/line anchor lives on the
        # thread ROOT comment, not the reply. BB DC sends ``commentParentId``
        # but no ``anchor`` on the reply; on Gitea a reply can likewise
        # arrive without a populated path/line. Without this, ``file_path``/
        # ``line`` stay empty, the code-context fetch is skipped, and the
        # file-targeted diff truncation degrades to generic head-truncation
        # (potentially dropping the very file under discussion). Recover the
        # anchor from the root (``thread[0]``) — both providers populate
        # ``file_path``/``line`` on every rendered thread comment, including
        # the root. Degrade gracefully when the root also lacks an anchor.
        if (not file_path or line <= 0) and thread:
            root = thread[0]
            root_path = root.get("file_path") or ""
            root_line = root.get("line") or 0
            if root_path and root_line > 0:
                file_path = root_path
                line = root_line
                logger.debug(
                    "Recovered inline anchor for PR #%s reply from thread root: %s:%s",
                    pr_number, file_path, line,
                )

        # Fetch Raven's account name once up-front. Used by:
        #   - Non-mention thread verification (below) — only respond when
        #     Raven is already in the thread, so a reply-without-mention
        #     can be intended for us.
        #   - The AI prompt rendering — entries authored by ``raven_user``
        #     get a ``[YOU]`` marker so the model can identify which
        #     thread entries are its own findings (eligible for retract).
        #   - The retract authorship filter — only IDs authored by
        #     ``raven_user`` survive into ``to_retract``.
        # Failure (auth/transport) degrades gracefully: empty string, no
        # [YOU] markers, retract filter drops everything (safe — same as
        # the prior behavior).
        try:
            raven_user = provider.get_authenticated_user() or ""
        except Exception:
            raven_user = ""

        # Thread verification. @mentions dispatched by the handler are
        # authoritative and skip this step. Reply-with-no-mention events
        # land here with is_mention=False; use the thread we already
        # fetched to decide whether Raven should engage at all.
        if not is_mention:
            if not (raven_user and parent_comment_id):
                logger.debug("Comment on PR #%s not directed at Raven — skipping",
                             pr_number)
                return
            thread_authors = [c.get("user", {}).get("login", "") for c in thread]
            raven_lower = raven_user.lower()
            if not any(a and a.lower() == raven_lower for a in thread_authors):
                logger.debug("Raven not in thread rooted at %s — skipping",
                             parent_comment_id)
                return

        # Immediate 👀 ack so the user knows Raven saw the comment, long
        # before the Claude response lands. Best-effort — providers without
        # a reactions API (BB DC) no-op; swallow failures.
        if comment_id:
            try:
                provider.react_to_comment(repo_full_name, pr_number, comment_id)
            except Exception as e:
                logger.debug("react_to_comment failed: %s", e)

        # Fetch context (truncate diff to avoid token bloat on large PRs).
        # For diff comments on a specific file, bias the truncation so that
        # file's hunk is kept even when the rest of the diff doesn't fit.
        raw_diff = provider.fetch_pr_diff(repo_full_name, pr_number)
        diff = _strip_lockfiles_and_binaries(raw_diff)
        diff = _truncate_diff_for_comment(diff, file_path, line)
        # Fetch CLAUDE.md from the PR's BASE ref (matches _process_pr's
        # post-trust-tier behavior). CLAUDE.md is repo-policy content and
        # the reviewer renders it in the trusted ``<repo_policy_TAGID>``
        # block; using base ref means a PR can't sneak in policy-shaped
        # text that would bias its own re-review through the comment-reply
        # path. Code snippets later in this function still use head_sha
        # since they're showing the actual code under review.
        try:
            comment_base_ref = provider.get_pr_base_ref(repo_full_name, pr_number)
        except Exception as e:
            logger.debug("get_pr_base_ref for PR #%s CLAUDE.md fetch failed (falling back to HEAD): %s",
                         pr_number, e)
            comment_base_ref = "HEAD"
        # Fetch the PR head SHA up-front for the code-snippet block below.
        try:
            cmd_head_sha = provider.get_pr_head_sha(repo_full_name, pr_number)
        except Exception as e:
            logger.debug("get_pr_head_sha for PR #%s snippet fetch failed (falling back to HEAD): %s",
                         pr_number, e)
            cmd_head_sha = "HEAD"
        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref=comment_base_ref)
        except Exception as e:
            # 404 (file missing) returns "" without raising; reaching this
            # except means an auth/transport failure worth flagging.
            logger.warning("CLAUDE.md fetch for PR #%s reply failed (reply proceeds without repo context): %s",
                           pr_number, e)

        # Fetch conversation (keep last N to avoid prompt bloat). Dedupe
        # against thread IDs so the same comment doesn't appear twice.
        # Note: in both Gitea and BB DC, comment IDs are unique within a
        # repo namespace (single comment table with shared auto-increment
        # PK), so dedup by bare id is safe.
        thread_ids = {c.get("id") for c in thread}
        conversation = [
            c for c in provider.get_pr_comments(repo_full_name, pr_number)[-COMMENT_HISTORY:]
            if c.get("id") not in thread_ids
        ]

        # Fetch prior verdict from cache (best-effort; None disables
        # comment-driven verdict revision below).
        pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"
        prior_verdict: str | None = None
        prior_body: str | None = None
        with _previous_diffs_lock:
            cache_entry = _previous_diffs.get(pr_key)
        if cache_entry is not None:
            prior_verdict = cache_entry.verdict
            prior_body = cache_entry.summary

        # For inline diff comments, fetch the FULL modified file (capped at
        # MAX_FILE_LINES, mirroring the review flow's _fetch_changed_files)
        # and inject it into the prompt so a question about code OUTSIDE the
        # ±10-line snippet window is answerable. The narrow line-numbered
        # window is still extracted (it pinpoints the line under discussion),
        # but the full file is the substantive context.
        #
        # Three disclosure signals flow to the prompt so the model never
        # asserts code it wasn't shown:
        #   - file_content: full text (only when within the line cap)
        #   - file_truncated: True when the file exceeds MAX_FILE_LINES
        #     (full text withheld, omission disclosed)
        #   - context_fetch_failed: True when the fetch raised (auth/
        #     transport) — logged at WARNING, disclosed so the model flags
        #     uncertainty rather than guessing.
        code_snippet = ""
        file_content = ""
        file_truncated = False
        context_fetch_failed = False
        if file_path and line > 0:
            try:
                snippet_ref = cmd_head_sha
                if snippet_ref == "HEAD":
                    snippet_ref = provider.get_pr_head_sha(repo_full_name, pr_number)
                fetched = provider.fetch_file(repo_full_name, file_path, ref=snippet_ref)
                code_snippet = _extract_code_snippet(fetched, line)
                if fetched:
                    if fetched.count("\n") <= MAX_FILE_LINES:
                        file_content = fetched
                    else:
                        # Over the line cap — withhold the full text but
                        # disclose the omission (the snippet still localises
                        # the discussion).
                        file_truncated = True
            except Exception as e:
                # WARNING (not debug): a fetch failure here means the model
                # sees neither the full file nor the focused snippet, so it
                # must be told to flag uncertainty. The PR-reply still goes
                # out — degraded, but disclosed.
                context_fetch_failed = True
                logger.warning("Could not fetch code context for %s:%s on PR #%s — %s",
                               file_path, line, pr_number, e)

        # Fetch per-repo respond-prompt override from the PR base branch.
        # Reuse the base ref already fetched above for CLAUDE.md when
        # available; only re-call if that initial fetch failed.
        respond_prompt_override = None
        try:
            base_ref = (
                comment_base_ref
                if comment_base_ref != "HEAD"
                else provider.get_pr_base_ref(repo_full_name, pr_number)
            )
            respond_prompt_override = _fetch_prompt_override(
                provider, repo_full_name, base_ref, "respond",
            )
        except Exception as e:
            logger.debug("Could not resolve base ref / respond override for PR #%d: %s",
                         pr_number, e)

        # Generate response. respond_to_comment returns
        # {response, revise, retract_findings} since the comment-thread-context
        # feature; mocks may still return a plain string in older tests
        # (treated as response-only for back-compat).
        try:
            result = respond_to_comment(
                comment_body, conversation, diff, repo_full_name,
                claude_md=claude_md, file_path=file_path, line=line,
                code_snippet=code_snippet,
                file_content=file_content,
                file_truncated=file_truncated,
                context_fetch_failed=context_fetch_failed,
                prompt_override=respond_prompt_override,
                thread=thread,
                prior_verdict=prior_verdict,
                prior_body=prior_body,
                raven_user=raven_user,
            )
        except RespondParseError as e:
            logger.warning("Respond JSON parse error for PR #%d: %s", pr_number, e)
            inc("raven_response_parse_errors_total", {"repo": repo_full_name})
            provider.post_pr_comment(
                repo_full_name, pr_number,
                "\U0001f985 ⚠️ Couldn't generate a response — please try rephrasing.",
                parent_comment_id=comment_id,
            )
            return
        if isinstance(result, dict):
            response = result.get("response", "")
            revise = result.get("revise")
            retract_findings = result.get("retract_findings") or []
        else:
            response = result
            revise = None
            retract_findings = []

        # Post response (skip if Claude returned nothing)
        if not response:
            logger.warning("Empty response from Claude for comment on PR #%d — not posting", pr_number)
            provider.post_pr_comment(
                repo_full_name, pr_number,
                "\U0001f985 \u26a0\ufe0f Couldn't generate a response — please try rephrasing.",
                parent_comment_id=comment_id,
            )
            return
        # When the reply is threaded by the provider, the UI already shows
        # the file/line context of the thread — skip the redundant Re:
        # header. Keep it for flat-comment providers (Gitea).
        threaded = bool(comment_id) and getattr(provider, "supports_comment_threads", False)
        include_location = file_path and not threaded
        if include_location:
            location = f"`{file_path}`"
            if line:
                location += f" line {line}"
            body = f"\U0001f985 **Re: {location}**\n\n{response}"
        else:
            body = f"\U0001f985 {response}"
        provider.post_pr_comment(repo_full_name, pr_number, body,
                                 parent_comment_id=comment_id)
        inc("raven_responses_total", {"repo": repo_full_name})
        logger.info("Responded to comment on PR #%d in %s", pr_number, repo_full_name)

        # ── Retraction + verdict revision + auto-merge dispatch ────── #
        # Skip if nothing to do.
        if revise is None and not retract_findings:
            return

        # Atomic race guard: mirrors _process_pr's pattern at
        # server.py:522-529 — check-and-add under the lock. If a push-
        # driven re-review is in flight for this PR, skip mutations (the
        # in-flight review will write its own verdict and would race with
        # us on submit_review + the cache).
        #
        # Asymmetric semantics:
        #   - Check _in_progress_prs (push reviews) → push wins; bail.
        #   - Check _comment_mutating_prs → another comment-flow is
        #     already mutating this PR; bail to avoid both submitting
        #     opposing reviews + dismissing each other's. (TOCTOU re-
        #     check alone doesn't cover the window between cache read
        #     and cache write, where both flows can be in flight.)
        #   - Add ourselves to _comment_mutating_prs ONLY — pushing
        #     never waits on a comment-flow.
        with _in_progress_lock:
            if pr_key in _in_progress_prs:
                logger.debug("Skipping comment-driven mutations — push re-review in flight for %s", pr_key)
                return
            if pr_key in _comment_mutating_prs:
                logger.debug("Skipping comment-driven mutations — another comment-flow in flight for %s", pr_key)
                return
            _comment_mutating_prs.add(pr_key)

        mutated_cache = False
        try:
            # TOCTOU re-check: re-read prior_verdict from the cache under
            # the guard. If it moved (because a concurrent _process_pr
            # landed a fresh review while the AI was thinking), the AI's
            # decision is on stale context — skip mutations.
            with _previous_diffs_lock:
                current_entry = _previous_diffs.get(pr_key)
            current_verdict = current_entry.verdict if current_entry else None
            if current_verdict != prior_verdict:
                logger.debug(
                    "Cache verdict changed under us (%s -> %s) — skipping comment-driven mutations for %s",
                    prior_verdict, current_verdict, pr_key,
                )
                return

            # Server-side defense in depth: the prompt instructs the AI
            # not to revise without a prior verdict, but enforce here too.
            if prior_verdict is None:
                revise = None

            # PR state gate — fail closed.
            try:
                pr_state = provider.get_pr_state(repo_full_name, pr_number)
            except Exception as e:
                logger.warning("get_pr_state failed for %s: %s — skipping mutations (fail-closed)",
                               pr_key, e)
                return
            if pr_state != "open":
                logger.debug("PR %s state=%s — skipping revision/retraction", pr_key, pr_state)
                return

            # Filter retraction IDs against fetched thread, restricted to
            # comments Raven itself authored. Defense in depth: prevents
            # a hallucinating AI (or a prompt-injection vector) from
            # resolving a developer's comment via the platform API. We
            # only ever retract our OWN findings.
            raven_user_lc = raven_user.lower()
            raven_owned_ids = {
                c.get("id") for c in thread
                if c.get("id") is not None
                and ((c.get("user") or {}).get("login") or "").lower() == raven_user_lc
                and raven_user_lc
            }
            to_retract_seeds = [cid for cid in retract_findings if cid in raven_owned_ids]
            dropped = set(retract_findings) - set(to_retract_seeds)
            if dropped:
                # WARNING (not DEBUG): when the AI tries to retract IDs
                # we can't honor (not in thread, or not authored by us),
                # operators need to see it. A silent drop here on every
                # comment looks identical to "AI didn't try" from the
                # outside and is hard to diagnose.
                logger.warning("Dropped %d retract IDs not authored by Raven or absent from thread: %s",
                               len(dropped), dropped)

            # The AI may pick a reply id from the thread, but thread
            # resolution is semantically a thread-root operation —
            # ``threadResolved`` (the BB DC field the UI's "Resolve
            # thread" button maps to) belongs on the root comment, and
            # Gitea's ``/resolve`` endpoint treats every comment in a
            # ``(path, position)`` group the same anyway. Walk to root
            # in memory using the parent_id linkage filled in by
            # ``get_comment_thread`` (BB DC's GET shape has no parent
            # field, so the provider can't walk up via the single-
            # comment API).
            parent_map: dict[int, int | None] = {
                c.get("id"): c.get("parent_id")
                for c in thread if c.get("id") is not None
            }

            def _walk_to_root(start_id: int) -> int:
                """Walk parent_id chain in the in-memory thread; cycle-safe."""
                seen: set[int] = set()
                cur = start_id
                while (cur in parent_map
                        and parent_map[cur] is not None
                        and cur not in seen):
                    seen.add(cur)
                    cur = parent_map[cur]
                return cur

            # Resolve each retract candidate to its thread root. The
            # root may NOT be Raven-authored (e.g., the AI is replying
            # inside a developer-rooted thread); only resolve when the
            # root is also Raven-authored — we only resolve threads
            # Raven itself originated.
            to_retract: list[int] = []
            for cid in to_retract_seeds:
                root_cid = _walk_to_root(cid)
                if root_cid not in raven_owned_ids:
                    logger.warning(
                        "Dropped retract %s — thread root %s not authored by Raven",
                        cid, root_cid,
                    )
                    continue
                if root_cid not in to_retract:  # dedupe: multiple seeds may share a root
                    to_retract.append(root_cid)

            any_retraction_succeeded = False
            for cid in to_retract:
                try:
                    ok = provider.retract_finding(repo_full_name, pr_number, cid)
                except Exception as e:
                    logger.warning("retract_finding failed for comment %s on PR #%d: %s",
                                   cid, pr_number, e)
                    inc("raven_retractions_total", {"repo": repo_full_name, "result": "fail"})
                    continue
                inc("raven_retractions_total",
                    {"repo": repo_full_name, "result": "ok" if ok else "fail"})
                if not ok:
                    continue
                any_retraction_succeeded = True
                # Drop matching cached finding so the next push-driven
                # incremental review doesn't carry it forward and re-post
                # it as a new inline comment, effectively undoing the
                # retraction. Findings carry `comment_id` only when
                # provider.submit_review's extended return shape is wired
                # (deferred to a follow-up); legacy findings without
                # comment_id fall through this loop without a match.
                with _previous_diffs_lock:
                    entry = _previous_diffs.get(pr_key)
                    if entry is None:
                        continue
                    for fname, file_findings in list(entry.findings.items()):
                        kept = [f for f in file_findings if f.get("comment_id") != cid]
                        if len(kept) != len(file_findings):
                            entry.findings[fname] = kept
                            entry.timestamp = time.time()  # mark recently active so LRU doesn't evict
                            mutated_cache = True
                            logger.debug("Dropped retracted finding (comment_id=%s) from cache file %s",
                                         cid, fname)
                            break

            # Defense in depth: if the AI retracted finding(s) but didn't
            # set `revise`, and the cache now has no remaining findings,
            # and prior verdict was needs_work — synthesize a flip to
            # approve. The basis for blocking has been removed via the
            # conversation; the prompt asks the AI to revise in this
            # case but a conservative AI may not. Backstop here.
            if (
                any_retraction_succeeded
                and revise is None
                and prior_verdict == "needs_work"
            ):
                with _previous_diffs_lock:
                    entry_after_retract = _previous_diffs.get(pr_key)
                remaining_count = (
                    sum(len(fl) for fl in entry_after_retract.findings.values())
                    if entry_after_retract else 0
                )
                if remaining_count == 0:
                    logger.info(
                        "PR #%d: synthesizing revise→approve — all findings retracted via comment thread",
                        pr_number,
                    )
                    revise = {
                        "verdict": "approve",
                        "body": (
                            "🦅 **Raven Review (Revised)**\n\n"
                            "Revised to approve following the comment-thread discussion: "
                            "all previously-flagged findings have been retracted."
                        ),
                    }

            # Coverage-gap guard on the revision path: while the cached
            # review state records unreviewed files (oversized/failed
            # chunks), a comment-driven flip-to-approve — whether the AI
            # proposed it or the retraction backstop above synthesized
            # it — must be suppressed entirely. Posting a formal APPROVE
            # for code Raven never saw is the same invariant violation
            # as auto-merging it (the respond model can't see the
            # unreviewed files any better than the review did). The
            # conversational reply has already posted above; the cached
            # verdict stays needs_work, and the merge-dispatch gate
            # below remains as defense-in-depth. The gap clears via the
            # push flow once the named files re-review cleanly.
            #
            # Fail direction mirrors the merge-dispatch gate: a MISSING
            # entry means the gap state is unverifiable (the eviction
            # window between the TOCTOU re-check and here is real —
            # several provider HTTP round-trips sit in between, during
            # which a concurrent _process_pr's _evict_cache() can LRU-
            # evict this PR), so suppress the approve rather than
            # letting gap_files default to "no gap".
            if revise is not None and revise.get("verdict") == "approve":
                with _previous_diffs_lock:
                    gap_entry = _previous_diffs.get(pr_key)
                if gap_entry is None:
                    logger.warning(
                        "PR #%d: cache entry missing at flip-to-approve guard "
                        "— suppressing revision (fail-closed, mirrors the "
                        "merge-dispatch gate)", pr_number,
                    )
                    revise = None
                elif gap_entry.coverage_gap_files:
                    logger.warning(
                        "PR #%d: suppressing comment-driven flip-to-approve — "
                        "unreviewed files remain (%s)",
                        pr_number, ", ".join(gap_entry.coverage_gap_files),
                    )
                    revise = None

            # Verdict revision (only when verdict actually changes).
            do_revision = revise is not None and revise.get("verdict") != prior_verdict

            # Lift entry + remaining_findings ABOVE submit_review so the
            # advisory body wrap can render the synthetic review correctly
            # (needs severity + findings list at body-construction time).
            # The dispatch site below reads these same locals — single
            # source of truth, no duplicate flatten.
            with _previous_diffs_lock:
                entry = _previous_diffs.get(pr_key)
            remaining_findings: list[dict] = []
            if entry is not None:
                for fl in entry.findings.values():
                    remaining_findings.extend(fl)

            if not do_revision:
                new_verdict = prior_verdict
                new_body = prior_body or ""
                rev_head_sha = None
            else:
                new_verdict = revise["verdict"]
                # In advisory mode wrap revise.body via _format_comment so
                # the comment renders with the "Updated Recommendation"
                # header. In all/gap modes keep the raw revise.body.
                if RAVEN_REVIEW_MODE == "advisory":
                    new_body = _format_comment(
                        {
                            "severity": _max_severity_from_findings(remaining_findings),
                            "summary": revise["body"],
                            "findings": remaining_findings,
                        },
                        mode="advisory_update",
                    )
                else:
                    new_body = revise["body"]

            if do_revision:
                # Pin the revised review to the current head_sha for
                # force-push protection, same as _process_pr does at
                # server.py:739. Fail closed if we can't fetch it —
                # submitting with commit_id='' silently bypasses Gitea's
                # force-push guard so a push between the AI call and now
                # would land the revised verdict on un-inspected commits.
                # Next push will re-trigger _process_pr and the verdict
                # can be revised again from there.
                try:
                    rev_head_sha = provider.get_pr_head_sha(repo_full_name, pr_number)
                except Exception as e:
                    logger.warning(
                        "Could not fetch head_sha for revision on PR #%d: %s — "
                        "skipping revision (fail-closed force-push protection)",
                        pr_number, e,
                    )
                    return
                if not rev_head_sha:
                    logger.warning(
                        "Empty head_sha for revision on PR #%d — skipping (fail-closed)",
                        pr_number,
                    )
                    return
                # Pass comment_only conditionally so out-of-tree providers
                # running in all/gap mode never see the new kwarg.
                advisory_kwargs = (
                    {"comment_only": True} if RAVEN_REVIEW_MODE == "advisory" else {}
                )
                try:
                    new_review_dict = provider.submit_review(
                        repo_full_name, pr_number,
                        body=new_body,
                        approve=(new_verdict == "approve"),
                        inline_comments=None,
                        commit_id=rev_head_sha,
                        **advisory_kwargs,
                    )
                except Exception as e:
                    logger.warning("Verdict revision submit_review failed for PR #%d: %s",
                                   pr_number, e)
                    inc("raven_revision_submit_errors_total", {"repo": repo_full_name})
                    return

                # Dismiss the prior Raven review so the PR shows the
                # current verdict cleanly. Without this, two reviews of
                # opposite verdicts (the original needs_work + the new
                # approve) coexist on the PR — confusing for humans.
                # Best-effort; failure doesn't block.
                new_review_id = new_review_dict.get("id") if isinstance(new_review_dict, dict) else None
                if new_review_id is not None:
                    try:
                        bot_user = provider.get_authenticated_user()
                        provider.dismiss_previous_reviews(
                            repo_full_name, pr_number, bot_user,
                            exclude_id=new_review_id,
                        )
                    except Exception as e:
                        logger.warning("Failed to dismiss prior reviews on PR #%d after revision: %s",
                                       pr_number, e)

                # Update cache verdict + summary. Defensive .get() in case
                # the entry was evicted between the prior read and now.
                with _previous_diffs_lock:
                    existing = _previous_diffs.get(pr_key)
                    if existing is not None:
                        existing.verdict = new_verdict
                        existing.summary = new_body
                        existing.timestamp = time.time()  # mark recently active for LRU
                        mutated_cache = True
                    else:
                        logger.debug("Cache entry for %s evicted between read and write", pr_key)
                inc("raven_verdict_revisions_total",
                    {"repo": repo_full_name,
                     "from": prior_verdict or "none", "to": new_verdict})

            # Auto-merge dispatch:
            #   (a) verdict flipped needs_work → approve, OR
            #   (b) verdict was already approve AND a retraction succeeded
            #       (BB DC all-comments-resolved unblock).
            # Suppressed in advisory mode — no formal verdict was registered,
            # so there's nothing to gate the merge on.
            should_dispatch_merge = (
                RAVEN_REVIEW_MODE != "advisory"
                and (
                    (prior_verdict == "needs_work" and new_verdict == "approve")
                    or (prior_verdict == "approve" and new_verdict == "approve" and any_retraction_succeeded)
                )
            )
            # Same sole-reviewer gate as _process_pr: a comment-driven
            # verdict flip must not auto-merge past a pending human
            # reviewer. Checked only when the other conditions already
            # hold, to avoid two provider calls on every comment.
            if should_dispatch_merge and not _is_sole_reviewer(
                    provider, repo_full_name, pr_number):
                should_dispatch_merge = False
            # Sticky coverage-gap gate (same fail-closed style as
            # _is_sole_reviewer): if the cached review state says parts
            # of the diff were never reviewed (oversized/failed chunks),
            # a comment-driven flip-to-approve or retraction must not
            # auto-merge — the respond model never saw the unreviewed
            # files either, so it can't validly clear the gap. The
            # verdict revision itself still posts above; only the merge
            # is blocked. Entry missing (evicted between the TOCTOU
            # check and here) → gap state unverifiable → fail closed.
            if should_dispatch_merge:
                if entry is None:
                    logger.warning(
                        "PR #%d: cache entry missing at merge dispatch — "
                        "skipping auto-merge (fail-closed)", pr_number)
                    should_dispatch_merge = False
                elif entry.coverage_gap_files:
                    logger.warning(
                        "PR #%d: cached review has unreviewed files (%s) — "
                        "skipping comment-driven auto-merge",
                        pr_number, ", ".join(entry.coverage_gap_files))
                    should_dispatch_merge = False
            if should_dispatch_merge:
                # Fail-closed on metadata fetch failure: dispatching with
                # head_sha='' would defeat _do_merge's force-push
                # protection.
                try:
                    head_sha = rev_head_sha if rev_head_sha else provider.get_pr_head_sha(
                        repo_full_name, pr_number,
                    )
                except Exception as e:
                    logger.warning("Could not fetch head_sha for auto-merge of PR #%d: %s — skipping dispatch",
                                   pr_number, e)
                    return
                if not head_sha:
                    return
                try:
                    meta = provider.get_pr_metadata(repo_full_name, pr_number)
                except Exception as e:
                    logger.debug("get_pr_metadata for PR #%d failed: %s", pr_number, e)
                    meta = {}
                pr_title = meta.get("title") or f"PR #{pr_number}"
                pr_url = meta.get("html_url") or ""
                merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
                # Synthesize a review dict reflecting residual cache state.
                # entry + remaining_findings were already computed above
                # before submit_review (single source of truth).
                synthetic_review = {
                    "approve": True,
                    "severity": _max_severity_from_findings(remaining_findings),
                    "summary": new_body,
                    "findings": remaining_findings,
                }
                fut = ci_wait_executor.submit(
                    _safe_do_merge, provider, repo_full_name, pr_number,
                    pr_title, pr_url, synthetic_review,
                    head_sha, merge_strategy,
                )
                fut.add_done_callback(
                    functools.partial(_log_future_exception, repo=repo_full_name),
                )
        finally:
            # ``_save_cache()`` catches all exceptions internally (disk
            # full / permission denied are WARNING-logged + counted via
            # ``raven_cache_save_failures_total``) and never propagates,
            # so this call is non-raising. The lock release below would
            # still happen even if it did, but keeping the call bare
            # avoids dead defensive code.
            if mutated_cache:
                _save_cache()
            # Release the comment-mutation slot so a follow-up comment
            # on the same PR can proceed. The push-review set
            # (_in_progress_prs) is owned by _process_pr; we don't
            # touch it.
            with _in_progress_lock:
                _comment_mutating_prs.discard(pr_key)

    except Exception as e:
        logger.error("Failed to respond to comment: %s", e, exc_info=True)
        repo_label = repo_full_name or "unknown"
        inc("raven_errors_total", {"type": "comment_response_failed",
                                   "repo": repo_label})
        # Same classified failure metric as the review flow so timeout /
        # usage-cap / auth spikes on comment replies show on the same
        # dashboard. The reply UX keeps its own threaded message below
        # (a verdict-style review comment would be wrong here); only the
        # metric is unified. respond_to_comment already retried the
        # transient classes inside reviewer.py.
        inc("raven_review_failures_total",
            {"reason": _review_failure_reason(e), "repo": repo_label})
        if repo_full_name and pr_number:
            try:
                provider.post_pr_comment(
                    repo_full_name, pr_number,
                    "\U0001f985 \u26a0\ufe0f Couldn't respond — internal error while processing your comment.",
                    parent_comment_id=comment_id,
                )
            except Exception:
                pass


# ------------------------------------------------------------------ #
#  Merge orchestration                                                 #
# ------------------------------------------------------------------ #

def _safe_do_merge(provider: GitProvider, repo_full_name: str, pr_number: int,
                   pr_title: str, pr_url: str, review: dict,
                   head_sha: str, merge_strategy: str) -> None:
    """Wrap ``_do_merge`` with the outer error handler that used to live
    in ``_process_pr`` when the merge was synchronous.

    Dispatching to ``ci_wait_executor`` moved ``_do_merge`` out of
    ``_process_pr``'s try/except, which previously logged with the real
    repo label, cleared dedup so retries could reprocess, and posted a
    user-visible "internal error" PR comment. Without this wrapper,
    unexpected merge-phase failures (network error from ``merge_pr``,
    unexpected shape from ``_wait_for_ci``) would surface only in the
    metric — users would see the review posted but no indication that
    the merge never happened.

    ``_do_merge`` already handles *expected* failures inline (CI failed,
    CI timed out, head-SHA drift, merge_pr returning False). This
    wrapper is the safety net for truly unexpected exceptions.
    """
    try:
        _do_merge(provider, repo_full_name, pr_number, pr_title, pr_url,
                  review, head_sha, merge_strategy)
    except Exception as e:
        logger.error("Unhandled error in merge phase for PR #%d (%s): %s",
                     pr_number, repo_full_name, e, exc_info=True)
        inc("raven_errors_total", {"type": "merge_unhandled", "repo": repo_full_name})
        # Clear dedup so a webhook retry can re-attempt the review + merge.
        # head_sha here is the same value that _process_pr used for dispatch.
        key = f"{provider.name}:{repo_full_name}#{pr_number}@{head_sha or 'HEAD'}"
        with _recent_prs_lock:
            _recent_prs.pop(key, None)
        with contextlib.suppress(Exception):
            provider.post_pr_comment(repo_full_name, pr_number,
                "🦅 **Raven Review**\n\n⚠️ Internal error during merge phase — "
                "the review was posted but the merge could not be attempted.")


def _do_merge(provider: GitProvider, repo_full_name: str, pr_number: int,
              pr_title: str, pr_url: str, review: dict,
              head_sha: str, merge_strategy: str) -> None:
    """Wait for CI (or use Gitea auto-merge) then merge the PR.

    When RAVEN_GITEA_AUTO_MERGE is enabled and the provider is Gitea,
    delegates CI waiting to Gitea via merge_when_checks_succeed.
    Otherwise polls CI and merges manually with head_commit_id safety.
    """
    if _GITEA_AUTO_MERGE and provider.name == "gitea":
        merged = provider.merge_pr(repo_full_name, pr_number, commit_title=pr_title,
                                   strategy=merge_strategy, head_sha=head_sha,
                                   merge_when_checks_succeed=True)
        if merged:
            logger.info("PR #%d auto-merge queued via Gitea", pr_number)
            inc("raven_auto_merge_queued_total", {"repo": repo_full_name})
        else:
            notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
                   link=pr_url, action="merge_failed")
            inc("raven_errors_total", {"type": "merge_failed", "repo": repo_full_name})
        return

    ci_timeout = int(os.environ.get("CI_WAIT_TIMEOUT", "300"))
    ci_status = _wait_for_ci(provider, repo_full_name, head_sha, timeout=ci_timeout)

    if ci_status in ("failure", "error"):
        logger.info("PR #%d CI failed — not merging", pr_number)
        notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
               link=pr_url, action="ci_failed")
        inc("raven_ci_failures_total", {"repo": repo_full_name})
        return

    if ci_status == "pending":
        logger.info("PR #%d CI still pending after timeout — not merging", pr_number)
        notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
               link=pr_url, action="ci_timeout")
        return

    # Verify head SHA hasn't changed during CI wait (provider-agnostic safety net;
    # Gitea also enforces this via head_commit_id, but BB DC ignores head_sha)
    try:
        current_sha = provider.get_pr_head_sha(repo_full_name, pr_number)
        if current_sha != head_sha:
            logger.info("PR #%d head SHA changed during CI wait (%s -> %s) — skipping merge",
                        pr_number, head_sha[:8], current_sha[:8])
            return
    except Exception as e:
        logger.warning("Could not verify head SHA for PR #%d: %s — skipping merge (fail closed)", pr_number, e)
        return

    # CI passed or no CI — merge (head_sha provides additional atomic safety on Gitea)
    merged = provider.merge_pr(repo_full_name, pr_number, commit_title=pr_title,
                               strategy=merge_strategy, head_sha=head_sha)
    if merged:
        inc("raven_merges_total", {"repo": repo_full_name})
    else:
        notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
               link=pr_url, action="merge_failed")
        inc("raven_errors_total", {"type": "merge_failed", "repo": repo_full_name})


# ------------------------------------------------------------------ #
#  Review-approved handler                                             #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  CI status polling                                                   #
# ------------------------------------------------------------------ #

def _wait_for_ci(provider: GitProvider, repo_full_name: str, sha: str, timeout: int = 300) -> str:
    """Poll commit status until CI finishes or timeout. Returns final status.

    Returns 'success', 'failure', 'error', 'pending', or 'none'.
    'none' means no CI is configured for this repo.

    Fast path: probe once up front and short-circuit on any terminal
    state (``success``, ``failure``, ``error``, ``none``). By the time
    ``_do_merge`` is invoked the review has already taken 30-60s; if CI
    was going to register ``pending`` it has. The fast path saves the
    10-second initial delay for the no-CI and CI-already-done cases
    (re-reviews on pushed branches hit this often).

    Slow path: if the first probe returns ``pending``, sleep an initial
    10s and then poll at ``interval`` until ``timeout``. The delay
    exists as belt-and-braces in case ``pending`` flaps back to
    ``none`` momentarily on some providers.

    ``RAVEN_REQUIRE_CI`` (opt-in, off by default): a provider can report
    ``none`` both for "no CI configured" AND transiently right after a
    push, before any CI system has registered a status — and the cached-
    merge path can reach this within seconds of a webhook (no AI pass),
    widening that window. When the toggle is on, ``none`` is treated as
    ``pending`` (wait for CI to register, up to ``timeout``) on BOTH the
    fast-path probe and the slow-path poll, so a repo that DOES have CI
    can't be merged before CI even starts. When off, behaviour is exactly
    as before: ``none`` is terminal and no-CI repos merge immediately.
    Read per-call so tests can monkeypatch it.
    """
    require_ci = os.environ.get("RAVEN_REQUIRE_CI", "").lower() in ("1", "true", "yes")
    # When RAVEN_REQUIRE_CI is on, ``none`` is no longer terminal — it
    # means "CI hasn't registered yet", so we keep waiting for it.
    terminal = ("success", "failure", "error") if require_ci \
        else ("success", "failure", "error", "none")

    # Fast path — terminal state already known.
    initial = provider.get_commit_status(repo_full_name, sha)
    if initial in terminal:
        return initial

    # CI_WAIT_TIMEOUT=0 (or negative) means "don't wait" — config.example.env
    # documents this as the skip-CI-check switch. Without this short-circuit
    # we'd still hit the time.sleep(initial_delay) below before exiting.
    if timeout <= 0:
        return initial  # whatever it is (likely "pending"); caller decides

    initial_delay = 10
    interval = 15
    elapsed = 0

    # Give CI time to stabilise before the next probe
    time.sleep(initial_delay)
    elapsed += initial_delay

    while elapsed < timeout:
        status = provider.get_commit_status(repo_full_name, sha)
        if status in terminal:
            return status
        # Still pending (or, under RAVEN_REQUIRE_CI, an unregistered
        # ``none``) — wait and retry
        logger.info("CI pending for %s@%s — waiting (%ds/%ds)", repo_full_name, sha[:8], elapsed, timeout)
        time.sleep(interval)
        elapsed += interval

    return "pending"


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _findings_by_file(findings: list[dict], filenames: set[str]) -> dict[str, list[dict]]:
    """Group findings by their 'file' key. File-less findings go under key ''."""
    by_file: dict[str, list[dict]] = {"": [], **{f: [] for f in filenames}}
    for finding in findings:
        fname = finding.get("file", "")
        if fname in by_file:
            by_file[fname].append(finding)
        else:
            by_file[""].append(finding)
    return by_file


CODE_SNIPPET_CONTEXT_LINES = 10  # lines before/after the commented line


def _extract_code_snippet(file_content: str, line: int,
                           context: int = CODE_SNIPPET_CONTEXT_LINES) -> str:
    """Return a line-numbered window of ``file_content`` around ``line``.

    The target line is marked with ``→`` so Claude can't misidentify which
    line the comment is about. Returns an empty string when the content or
    line number is invalid.
    """
    if not file_content or line <= 0:
        return ""
    lines = file_content.splitlines()
    if not lines or line > len(lines):
        return ""
    start = max(1, line - context)
    end = min(len(lines), line + context)
    width = len(str(end))
    formatted: list[str] = []
    for n in range(start, end + 1):
        marker = "→" if n == line else " "
        formatted.append(f"{n:>{width}} {marker} {lines[n - 1]}")
    return "\n".join(formatted)


def _head_truncate(diff: str) -> str:
    """Plain head-truncation fallback used when no relevance bias applies."""
    total = diff.count("\n")
    if total <= MAX_DIFF_LINES:
        return diff
    lines = diff.splitlines(keepends=True)
    # Consume lines until we've included MAX_DIFF_LINES newlines.
    out_lines: list[str] = []
    seen = 0
    for ln in lines:
        if seen >= MAX_DIFF_LINES:
            break
        out_lines.append(ln)
        seen += ln.count("\n")
    return "".join(out_lines) + f"\n... (truncated, {total - seen} lines omitted)"


_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-\d+(?:,\d+)?\s+\+(?P<start>\d+)(?:,(?P<span>\d+))?\s+@@",
)


def _split_chunk_by_hunks(chunk: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Split a single-file diff chunk into (header, hunks).

    ``header`` is everything before the first ``@@`` line (diff --git,
    index, ---, +++). Each hunk is ``(dst_start, dst_end, text)`` where
    dst range is the inclusive destination line span. Returns an empty
    hunks list if the chunk has no parseable hunk headers.
    """
    lines = chunk.splitlines(keepends=True)
    header_lines: list[str] = []
    hunks: list[tuple[int, int, str]] = []
    current_start = 0
    current_span = 0
    current_text: list[str] = []

    def _flush() -> None:
        if current_text:
            end = current_start + max(current_span - 1, 0)
            hunks.append((current_start, end, "".join(current_text)))

    seen_hunk = False
    for ln in lines:
        m = _HUNK_HEADER_RE.match(ln)
        if m:
            _flush()
            current_start = int(m.group("start"))
            current_span = int(m.group("span") or "1")
            current_text = [ln]
            seen_hunk = True
        elif not seen_hunk:
            header_lines.append(ln)
        else:
            current_text.append(ln)
    _flush()
    return "".join(header_lines), hunks


def _window_chunk_around_line(chunk: str, line: int, budget: int) -> str | None:
    """Return a view of ``chunk`` containing the hunk that covers ``line``
    plus as many neighbouring hunks as fit in ``budget`` newlines.

    Returns ``None`` if the chunk has no parseable hunks or no hunk covers
    the target line — callers should fall back to head-truncation in that
    case. The diff --git / ---/+++ header is always preserved so the
    output is still a parseable unified-diff fragment.
    """
    header, hunks = _split_chunk_by_hunks(chunk)
    if not hunks:
        return None

    # Find the hunk covering the target line, or the closest one.
    covering_idx = None
    for i, (start, end, _text) in enumerate(hunks):
        if start <= line <= end:
            covering_idx = i
            break
    if covering_idx is None:
        return None

    header_lines = header.count("\n")
    budget -= header_lines
    if budget <= 0:
        return None

    kept = [covering_idx]
    covering_lines = hunks[covering_idx][2].count("\n")
    if covering_lines > budget:
        # Even the target hunk exceeds the budget — head-truncate it and
        # skip the rest.
        trunc = _head_truncate_chunk(hunks[covering_idx][2], budget)
        return header + trunc
    budget -= covering_lines

    # Expand outward alternately — one after, one before, until budget runs out.
    before = covering_idx - 1
    after = covering_idx + 1
    while before >= 0 or after < len(hunks):
        took = False
        if after < len(hunks):
            size = hunks[after][2].count("\n")
            if size <= budget:
                kept.append(after)
                budget -= size
                after += 1
                took = True
            else:
                after = len(hunks)
        if before >= 0:
            size = hunks[before][2].count("\n")
            if size <= budget:
                kept.insert(0, before)
                budget -= size
                before -= 1
                took = True
            else:
                before = -1
        if not took:
            break

    out_hunks = "".join(hunks[i][2] for i in sorted(kept))
    dropped = len(hunks) - len(kept)
    suffix = ""
    if dropped:
        suffix = f"\n... (truncated, {dropped} hunk(s) omitted to keep line {line} in context)"
    return header + out_hunks + suffix


def _head_truncate_chunk(chunk: str, budget: int) -> str:
    """Head-truncate a single chunk to ``budget`` newlines, with marker."""
    lines = chunk.splitlines(keepends=True)
    out: list[str] = []
    seen = 0
    for ln in lines:
        if seen >= budget:
            break
        out.append(ln)
        seen += ln.count("\n")
    total = chunk.count("\n")
    return "".join(out) + f"\n... (truncated, {total - seen} lines of the hunk omitted)"


def _truncate_diff_for_comment(diff: str, file_path: str = "", line: int = 0) -> str:
    """Truncate a diff to MAX_DIFF_LINES, keeping the relevant file first.

    For diff comments that name a file, split the diff per file, place that
    file's hunk first, then append other files in order until the limit is
    reached. If the relevant file's own chunk exceeds the budget:

    - When a ``line`` is provided and one of the hunks covers it, window
      around that hunk so the commented-on line stays in the output.
    - Otherwise head-truncate the chunk so the file is at least visible.

    Without a ``file_path`` — or when the named file isn't in the diff —
    falls back to plain head-truncation of the whole diff.
    """
    if diff.count("\n") <= MAX_DIFF_LINES:
        return diff

    if not file_path:
        return _head_truncate(diff)

    file_chunks = split_diff_by_file(diff)
    relevant = [(fn, ch) for fn, ch in file_chunks if fn == file_path]
    others = [(fn, ch) for fn, ch in file_chunks if fn != file_path]

    if not relevant:
        logger.debug(
            "Comment file_path %r not found among diff files %r — "
            "falling back to head-truncation",
            file_path, [fn for fn, _ in file_chunks],
        )
        return _head_truncate(diff)

    _relevant_fn, relevant_chunk = relevant[0]
    relevant_lines = relevant_chunk.count("\n")
    output_parts: list[str] = []
    budget = MAX_DIFF_LINES
    skipped_files = 0
    total_files = len(file_chunks)

    if relevant_lines > budget:
        # Target chunk exceeds the budget. If we know which line the user
        # commented on, window around the hunk that contains it so that
        # line stays visible. Falling back to head-truncation only when
        # no hunk covers the line.
        windowed = None
        if line > 0:
            windowed = _window_chunk_around_line(relevant_chunk, line, budget)
        if windowed is not None:
            output_parts.append(windowed)
        else:
            output_parts.append(_head_truncate(relevant_chunk))
        skipped_files = total_files - 1
    else:
        output_parts.append(relevant_chunk)
        budget -= relevant_lines
        for _fn, chunk in others:
            chunk_lines = chunk.count("\n")
            if chunk_lines <= budget:
                output_parts.append(chunk)
                budget -= chunk_lines
            else:
                skipped_files += 1

    out = "".join(output_parts)
    if skipped_files:
        out += (
            f"\n... (truncated, {skipped_files} of {total_files} file(s) "
            f"omitted to keep `{file_path}` in context)"
        )
    return out


def _resolve_file_context_caps() -> tuple[int, int]:
    """Read the file-context caps from the environment.

    Returns ``(max_file_lines, max_files)``:
      - ``RAVEN_MAX_FILE_LINES`` (default 500) — skip files larger than
        this (generated/minified, or just too big for the prompt).
      - ``RAVEN_MAX_FILES`` (default 10) — limit total files attached
        to avoid token bloat.
    """
    return (
        int(os.environ.get("RAVEN_MAX_FILE_LINES", "500")),
        int(os.environ.get("RAVEN_MAX_FILES", "10")),
    )


MAX_FILE_LINES, MAX_FILES = _resolve_file_context_caps()


def _fetch_changed_files(provider: GitProvider, repo_full_name: str, head_sha: str, clean_diff: str) -> tuple[dict[str, str], list[str]]:
    """Fetch full contents of changed files for review context.

    Returns ``(contents, omitted)``: ``contents`` maps filename → file
    text; ``omitted`` lists human-readable notes for files skipped by
    the context caps (over ``MAX_FILE_LINES``, or beyond ``MAX_FILES``)
    so the prompt can disclose the gap instead of letting the model
    assume the attached contents are exhaustive. Fetch failures are
    logged, not disclosed — they are not cap omissions.
    """
    file_chunks = split_diff_by_file(clean_diff)
    file_contents: dict[str, str] = {}
    omitted: list[str] = []
    for filename, _ in file_chunks[:MAX_FILES]:
        try:
            content = provider.fetch_file(repo_full_name, filename, ref=head_sha)
        except Exception as e:
            logger.debug("Could not fetch %s for context: %s", filename, e)
            continue
        if not content:
            continue
        lines = content.count("\n")
        if lines <= MAX_FILE_LINES:
            file_contents[filename] = content
        else:
            omitted.append(f"{filename} ({lines} lines, exceeds the {MAX_FILE_LINES}-line cap)")
    for filename, _ in file_chunks[MAX_FILES:]:
        omitted.append(f"{filename} (beyond the {MAX_FILES}-file cap)")
    return file_contents, omitted


# Directory whose ``*.md`` files are injected as "Repository Rules"
# context for every review. Override with RAVEN_RULES_DIR (set to empty
# string to disable entirely). Default matches the common Claude-Code
# convention of ``.claude/rules/``.
RULES_DIR = os.environ.get("RAVEN_RULES_DIR", ".claude/rules")


def _fetch_rules(provider: GitProvider, repo_full_name: str, ref: str) -> dict[str, str]:
    """Read ``*.md`` files from ``RULES_DIR`` at ``ref``, return
    ``{path: contents}`` sorted by path.

    Best-effort: a missing directory, listing error, or individual
    fetch failure returns an empty/partial map rather than raising —
    the review must proceed regardless.
    """
    if not RULES_DIR:
        return {}
    try:
        entries = provider.list_directory(repo_full_name, RULES_DIR, ref=ref)
    except Exception as e:
        # 404 (directory absent) returns [] from both providers without
        # raising; reaching this except means a real listing failure
        # (auth, transport) the operator should see.
        logger.warning("Could not list %s at %s (review proceeds without rule context): %s",
                       RULES_DIR, ref[:8], e)
        return {}
    if not entries:
        return {}
    # Sort so prompt ordering is deterministic (helps cache hits + test
    # reproducibility). Only *.md files per the product decision; other
    # file types under .claude/rules/ are ignored.
    md_paths = sorted(p for p in entries if p.lower().endswith(".md"))
    rules: dict[str, str] = {}
    for path in md_paths:
        try:
            content = provider.fetch_file(repo_full_name, path, ref=ref)
            if content:
                rules[path] = content
        except Exception as e:
            # fetch_file returns "" on 404; raising here means real
            # operational failure on a single rule file. Other rules
            # still get processed; warn so operator sees the gap.
            logger.warning("Could not fetch rule file %s: %s", path, e)
    if rules:
        logger.info("Loaded %d rule file(s) from %s for %s", len(rules), RULES_DIR, repo_full_name)
    return rules


def _fetch_prompt_override(provider: GitProvider, repo_full_name: str,
                            ref: str, name: str) -> str | None:
    """Fetch a per-repo prompt override from ``{RULES_DIR}/raven/prompts/{name}.md``.

    Returns the file contents on success, ``None`` on any failure mode
    (missing file, fetch error, empty / whitespace-only content, or
    ``RULES_DIR`` disabled). Callers must treat ``None`` as "no override,
    use the built-in default".

    ``name`` is ``"review"`` or ``"respond"`` — not validated, but only
    those two are currently wired into the reviewer.
    """
    if not RULES_DIR:
        return None
    path = f"{RULES_DIR}/raven/prompts/{name}.md"
    try:
        content = provider.fetch_file(repo_full_name, path, ref=ref)
    except Exception as e:
        # fetch_file returns "" on 404 (override absent — the common
        # case); reaching this except means an auth/transport failure
        # that prevents Raven from honoring a configured override.
        logger.warning("Prompt override fetch for %s@%s/%s failed (using built-in default): %s",
                       repo_full_name, ref[:8] if ref else "", path, e)
        return None
    if not content or not content.strip():
        return None
    logger.info("Loaded %s prompt override from %s for %s",
                name, path, repo_full_name)
    return content


def _notify_if_needed(repo_full_name: str, pr_number: int, pr_title: str, pr_url: str, review: dict) -> None:
    """Send notification — severity filtering is handled per-channel in notifier."""
    notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
           link=pr_url, action="needs_review")


SEVERITY_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _max_severity_from_findings(findings: list[dict]) -> str:
    """Highest severity name among findings; ``"low"`` when the list is
    empty or no severity matches ``SEVERITY_ORDER``. Safe against
    unknown severities — never raises ``StopIteration``."""
    if not findings:
        return "low"
    max_sev = max(SEVERITY_ORDER.get(f.get("severity", "low"), 0)
                  for f in findings)
    for name, ord_ in SEVERITY_ORDER.items():
        if ord_ == max_sev:
            return name
    return "low"


def _format_comment(review: dict, mode: str = "review") -> str:
    """Render the review summary body.

    ``mode`` selects the header / subtitle:
      * ``"review"``           — formal review (header: "Raven Review").
      * ``"advisory"``         — initial advisory recommendation.
      * ``"advisory_update"``  — advisory recommendation revised via comment thread.
    """
    severity = review.get("severity", "low")
    summary = review.get("summary", "")
    findings = review.get("findings", [])
    emoji = SEVERITY_EMOJI.get(severity, "🟡")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if mode == "advisory":
        header = "🦅 **Raven Recommendation**"
    elif mode == "advisory_update":
        header = "🦅 **Raven Updated Recommendation**"
    else:
        header = "🦅 **Raven Review**"

    lines = [header]
    if mode in ("advisory", "advisory_update"):
        lines.append("_Advisory only — Raven is not blocking this PR._")
    lines.append("")
    lines.append(f"**{emoji} {severity.upper()}** — {summary}")

    if findings:
        lines.append("")
        lines.append("**Findings:**")
        for f in findings:
            f_sev = f.get("severity", "low")
            f_emoji = SEVERITY_EMOJI.get(f_sev, "🟡")
            lines.append(f"- {f_emoji} [{f_sev}] {f.get('message', '')}")

    if review.get("chunked"):
        n = review.get("chunks_reviewed", "?")
        lines.append("")
        lines.append(f"*⚡ Large diff — reviewed {n} files separately*")

    if review.get("carried_count"):
        n = review["carried_count"]
        lines.append("")
        lines.append(f"*Includes {n} finding(s) carried from unchanged files*")

    lines.append("")
    lines.append(f"*Reviewed by Raven · {RAVEN_AI_MODEL} · effort {RAVEN_AI_EFFORT} · {timestamp}*")

    return "\n".join(lines)


def _format_inline_leftovers(findings: list[dict]) -> str:
    """Minimal body for ``RAVEN_REVIEW_OUTPUT=inline``.

    Inline mode posts no summary/recommendation comment. The only findings
    that can't ride on a diff line are those with no postable file/line —
    PR-wide notes and ⚠️ coverage-gap markers (filename, no line). Those are
    listed in a short body so they aren't silently dropped. Returns ``""``
    when every finding is inline-anchored, so a clean review posts no body.
    """
    if not findings:
        return ""
    lines = ["🦅 **Raven** — findings without an inline location:", ""]
    for f in findings:
        f_sev = f.get("severity", "low")
        f_emoji = SEVERITY_EMOJI.get(f_sev, "🟡")
        lines.append(f"- {f_emoji} **[{f_sev}]** {f.get('message', '')}")
    return "\n".join(lines)


def _is_skipped_repo(repo_full_name: str) -> bool:
    skip_list = os.environ.get("SKIP_REPOS", "")
    if not skip_list:
        return False
    skipped = {r.strip() for r in skip_list.split(",") if r.strip()}
    return repo_full_name in skipped


def _is_bot_author(*names: str) -> bool:
    """Return True if any name looks like a bot.

    Matches exact names in ``default_bots`` or ``SKIP_AUTHORS``, the
    GitHub ``user[bot]`` suffix, and clear ``-bot`` / ``bot-`` affixes.
    Deliberately conservative on the dash-segment check: an earlier
    version used ``"bot" in n.split("-")`` which also matched real
    human names like ``rob-bot`` or ``turbo-bot``. Anyone who actually
    uses such a name for a bot account should list it in
    ``SKIP_AUTHORS``.
    """
    skip_authors_raw = os.environ.get("SKIP_AUTHORS", "")
    default_bots = {"bot", "github-actions", "dependabot", "renovate", "gitea-actions"}
    skipped = default_bots | {a.strip().lower() for a in skip_authors_raw.split(",") if a.strip()}
    for name in names:
        if not name:
            continue
        n = name.lower()
        if n in skipped or n.endswith("[bot]"):
            return True
        if n == "bot" or n.endswith("-bot") or n.startswith("bot-"):
            logger.info("Skipping bot author %r (matched affix heuristic)", name)
            return True
    return False
