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
from typing import Callable, NamedTuple

from flask import Flask, abort, jsonify, request

from . import __version__
from .providers import GitProvider, DiffHeadMismatchError, DiffIdentityUnverifiableError, DiffTruncatedError, DiffUnverifiableError, IncompleteFileError, ThreadResolvedError, get_provider, register_provider, registered_providers
from .providers.gitea import GiteaProvider
from .metrics import add, inc, Timer, format_prometheus
from .notifier import notify
from .reviewer import _is_lockfile_name, _normalize_path, _rename_aliases
from .reviewer import review_diff, respond_to_comment, review_config_hash, strip_diff, _diff_lines, split_diff_by_file, diff_hash, hunk_positions, hunk_context_digests, MAX_DIFF_LINES, terminate_active_processes, RespondParseError, RAVEN_AI_MODEL, RAVEN_AI_EFFORT, RAVEN_AI_TIMEOUT, RAVEN_AI_RETRY
from .ai import get_backend
from .ai.base import AIError
from .severity import SeverityScale, default_scale, from_json, InvalidScale


class DiffHeadUnverifiedError(RuntimeError):
    """The PR diff could not be shown to describe the head under review.

    Gitea builds ``.diff`` from ``refs/pull/N/head``, which a background
    task moves after a push, so for a while the diff can describe an
    older commit than the PR head. Reviewing or merging on that pairing
    would put one commit's verdict on another (audit 2026-09-27 #21).
    """


class HeadUnverifiedError(RuntimeError):
    """The PR head could not be re-read before posting an APPROVE.

    An approval has to name the commit it covers, and branch protection
    may count a bot APPROVE, so nothing is posted or cached; the user
    re-triggers the review instead.
    """


def _review_failure_reason(exc: Exception) -> str:
    """Classify an unhandled review exception into a failure reason.

    An :class:`AIError` already carries the classified ``.reason`` (set by
    the backend). A :class:`DiffTruncatedError` (provider returned a
    truncated/partial diff — too large for the platform's diff limit) maps
    to ``"diff_truncated"``. Anything else (a non-AI bug in the flow, an
    out-of-tree backend raising plain ``RuntimeError``) is ``"unknown"``.
    A :class:`DiffHeadUnverifiedError` (the diff never came to describe
    the head under review) is ``"diff_head_unverified"``.
    A :class:`HeadUnverifiedError` (the head could not be re-read before
    an APPROVE) is ``"head_unverified"``.

    ``DiffUnverifiableError`` is checked FIRST because it subclasses
    ``DiffTruncatedError`` — the two need different operator advice
    ("split the PR" is wrong when the problem is a response format we
    can't inspect), and an isinstance test against the parent would
    swallow the subclass. Its own subclass ``DiffIdentityUnverifiableError``
    (a file's content identity unreadable) comes before it, for the same
    reason.
    """
    if isinstance(exc, AIError):
        return exc.reason
    if isinstance(exc, (DiffHeadUnverifiedError, DiffHeadMismatchError)):
        return "diff_head_unverified"
    if isinstance(exc, HeadUnverifiedError):
        return "head_unverified"
    if isinstance(exc, DiffIdentityUnverifiableError):
        return "diff_identity_unverified"
    if isinstance(exc, DiffUnverifiableError):
        return "diff_unverifiable"
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
        "to re-trigger. (Lines Bitbucket cuts for length don't cause this "
        "notice: they are reviewed, and their file is marked as not fully shown.)"
    ),
    "diff_unverifiable": (
        "🔍 Raven could not verify that it received the **complete** diff. "
        "Bitbucket returned it in a format that carries no truncation "
        "information, so a diff cut off at the server's size limit would be "
        "indistinguishable from a whole one — and Raven won't review a diff it "
        "can't confirm is complete (it could otherwise approve or merge code it "
        "never saw). This is a server-side configuration issue, not a problem "
        "with the PR: the Bitbucket instance needs to serve "
        "`/pull-requests/{id}/diff` as `application/json`. Splitting the PR "
        "will not help."
    ),
    "diff_identity_unverified": (
        "🔗 Raven could not read every changed file's content identity: "
        "Bitbucket's list of the PR's changed files was missing a file, or the "
        "commit it describes. Without it Raven can't tie a verdict to the exact "
        "files of this commit, so nothing was reviewed or merged. Push a commit "
        "or re-request the review to retry; if it keeps happening, check the "
        "service logs."
    ),
    "diff_head_unverified": (
        "🔄 Raven could not get a diff of the latest commit: the git host "
        "still served the diff of an older commit (or its head could not be "
        "read) after waiting, so reviewing it could put one commit's verdict "
        "on another. Nothing was reviewed or merged. Push a commit or "
        "re-request the review to retry."
    ),
    "head_unverified": (
        "🔁 Raven finished the review but could not confirm the PR's latest "
        "commit before posting its approval, so it posted nothing: an "
        "approval has to name the commit it covers. Push a commit or "
        "re-request the review to retry."
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

# How long _process_pr waits for the PR diff to describe the head it is
# about to review: Gitea moves refs/pull/N/head, which ``.diff`` reads, in a
# background task after a push (see _await_diff_head).
_DIFF_HEAD_POLLS = 30
_DIFF_HEAD_POLL_INTERVAL = 1.0  # seconds
# Tries for a single head read outside that wait (after the diff fetch, or
# for a payload with no SHA): enough to ride out one API blip.
_HEAD_READ_TRIES = 3

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
# same PR exits immediately and parks its payload in _rerun_requested, so
# the running review re-runs it when it finishes (latest push wins).
_in_progress_prs: set[str] = set()
# Separate set for comment-driven mutations: gives push priority (push
# webhooks check ONLY _in_progress_prs and never wait on a comment-flow)
# AND serializes concurrent comment-flows on the same PR (without it, two
# comments arriving within ~10s would both pass the TOCTOU re-check before
# either wrote the cache, then both submit_review + dismiss each other's
# review).
_comment_mutating_prs: set[str] = set()
_in_progress_lock = threading.Lock()
# A push that hits the in-progress guard is not dropped: its payload is
# parked here (latest wins — three pushes during one review collapse into
# one re-run of the newest) and _process_pr's finally resubmits it once the
# running review ends. Without this the newer commit stayed unreviewed
# until some later event, while the PR showed Raven's verdict on the older
# head (audit 2026-09-27 #7). Guarded by _in_progress_lock, so a push can't
# slip between "review finished" and "re-run scheduled".
_rerun_requested: dict[str, tuple[GitProvider, dict]] = {}
# The head each in-flight _process_pr run is reviewing. Lets the guard tell
# a stale event for that head (a redelivery, an out-of-order delivery)
# from a newer push, so the stale one can't displace the push it follows.
_in_progress_heads: dict[str, str] = {}

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
    hashes: dict[str, str]            # filename -> SHA256 of the RAW diff chunk
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
    # field load as no-gap. Lifecycle documented in
    # docs/design-notes.md ("Coverage-gap tracking"); pinned end-to-end by
    # tests/test_server.py::TestCoverageGapBlocksMerge.
    coverage_gap_files: list[str] = field(default_factory=list)
    # Per-repo review config this entry was computed under: the resolved
    # severity scale, the per-repo review prompt override, the base branch
    # and the CLAUDE.md and rules (audit 09-27 #8), hashed by
    # _entry_config_hash(). review_config_hash() (module-level, wipes the
    # WHOLE cache) can't express "this one repo's scale changed" — this
    # field is the per-entry complement. Defaults to "" for entries loaded
    # from cache files written before this field existed.
    #
    # The "" default behaves DIFFERENTLY on the two read paths, and that
    # asymmetry is deliberate, not an inconsistency:
    #   * _process_pr (a fresh AI review runs) — "" mismatches a real
    #     _entry_config_hash() once, the review runs anyway, and the
    #     post-submit write records a real hash. One-time cost, then
    #     re-warmed, the same as any other config change.
    #   * _maybe_dispatch_cached_merge (NO review runs — this is the
    #     cached-approve-only dispatch path) — "" is treated as an ACTIVE
    #     mismatch against a caller-supplied expected hash, not skipped.
    #     There is no write on this path to re-warm from, so "skip the
    #     comparison for a hash-less entry" would let it auto-merge under
    #     any future scale forever. See _maybe_dispatch_cached_merge's
    #     docstring for the gate itself.
    config_hash: str = ""
    # Rebase tolerance. ``hashes`` above is the RAW chunk hash and answers
    # "is this literally the same diff?" — it is what the no-changes skip
    # and, through it, _maybe_dispatch_cached_merge's hash gate compare,
    # so a cached approve still only short-circuits to a merge when
    # nothing moved at all. These two answer the narrower question the
    # incremental delta actually asks:
    #   * content_hashes — reviewer.diff_hash(chunk): the added/removed
    #     lines only. A rebase rewrites a chunk's ``index`` blob SHAs,
    #     ``@@`` line numbers and context lines without touching the PR's
    #     own edits, and re-reviewing on that basis re-posts findings the
    #     developer already resolved (resolution matches on comment_id,
    #     which a regenerated finding does not have). Files equal here are
    #     NOT re-reviewed.
    #   * hunks — reviewer.hunk_positions(chunk): new-side (start, length)
    #     per hunk. A file can be content-equal while every line number in
    #     it moved, and carried findings keep the ``line`` they were found
    #     at — so without this the cache would pin them (and post any
    #     that has no thread yet) to whatever code the rebase slid into
    #     that position. _remap_carried_lines
    #     shifts them by the per-hunk delta; a finding that cannot be
    #     mapped safely sends its file back into changed_files for a real
    #     re-review instead.
    # Both default empty. An entry with no content_hashes has nothing to
    # compare against, so the delta falls back to the raw hashes — the
    # pre-existing behaviour, NOT "every file changed" — and with no
    # hunks there is simply nothing to remap. DIFF_HASH_SCHEME folds into
    # review_config_hash(), so in practice the whole cache wipes once on
    # the upgrade and both fields are populated from the next review on.
    content_hashes: dict[str, str] = field(default_factory=dict)
    hunks: dict[str, list] = field(default_factory=dict)
    #   * hunk_context — reviewer.hunk_context_digests(chunk): a digest of
    #     each hunk's body, its edit lines reduced to their +/- markers.
    #     Positions alone cannot tell a rebase (same edit, moved by the
    #     base branch) from the author RELOCATING a byte-identical edit
    #     elsewhere in the file, or REORDERING it across a context line —
    #     all leave content_hashes equal and the hunk geometry intact, so
    #     the finding is carried onto code that was never reviewed where
    #     it now sits. The body is what separates them: a rebase keeps it
    #     byte-identical while its position moves. A mismatch makes
    #     _remap_carried_lines give up, or (same positions) _process_pr
    #     send the file back, for a re-review.
    #     Empty for entries written before this field, which simply skips
    #     the comparison — the same degrade-to-previous-behaviour rule the
    #     two fields above follow.
    hunk_context: dict[str, list] = field(default_factory=dict)
    # The raw chunks of a head the rebase-only shortcut recorded WITHOUT a
    # review (hashes keeps the reviewed head's). A later trigger for that
    # same head — a re-requested review, a reopen — is someone asking
    # again, so it gets a full review instead of the shortcut; otherwise
    # the head would stay unreviewed, and the comment flow unbound, until
    # a content push. Every review write starts a fresh entry, clearing it.
    unreviewed_hashes: dict[str, str] = field(default_factory=dict)


def _entry_config_hash(scale: SeverityScale, prompt_override: str | None, *,
                       base_ref: str, claude_md: str, rules: dict[str, str]) -> str:
    """Per-repo review config that the global review_config_hash() cannot
    express: the resolved severity scale, the per-repo prompt override
    (closes backlog #15 — the override was missing from the cache key
    entirely), the base branch, and the CLAUDE.md and rules the review was
    judged under (audit 09-27 #8: a retargeted PR, or one whose base
    changed its policy, reused an approve judged under other policy). A
    mismatch against ``CacheEntry.config_hash`` means the entry was
    computed under policy that no longer applies to this PR. The new parts
    are keyword-only with no default, so a caller can't leave one out.

    The base branch NAME, not its commit: every merge to the base would
    otherwise send every open PR back for a full review, while the
    CLAUDE.md and rules digest already catches the policy changing."""
    h = hashlib.sha256()

    def _part(text: str) -> None:
        # Length-prefixed, so no part's content can shift into the next
        # (Raven's review of BB PR #16: with bare separators, a CLAUDE.md
        # that absorbed the rules' text hashed like the rules).
        data = (text or "").encode("utf-8")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)

    for part in (scale.fingerprint(), prompt_override or "", base_ref, claude_md):
        _part(part)
    _part(str(len(rules or {})))
    for path in sorted(rules or {}):
        _part(path)
        _part(rules[path])
    return h.hexdigest()[:16]


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
                        config_hash=entry.get("config_hash") or "",
                        content_hashes=dict(entry.get("content_hashes") or {}),
                        # JSON has no tuples — restore the (start, length)
                        # pairs hunk_positions produced, and let a row that
                        # isn't a pair of ints fail into the guard below
                        # rather than reach _remap_carried_lines malformed.
                        hunks={
                            k: [(int(h[0]), int(h[1])) for h in v]
                            for k, v in (entry.get("hunks") or {}).items()
                        },
                        hunk_context={
                            k: [str(h) for h in v]
                            for k, v in (entry.get("hunk_context") or {}).items()
                        },
                        unreviewed_hashes=dict(entry.get("unreviewed_hashes") or {}),
                    )
                except (KeyError, TypeError, ValueError, IndexError) as e:
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


# ── Startup config validation ─────────────────────────────────────── #
_KNOWN_AI_EFFORTS = {"none", "low", "medium", "high", "max"}
# Match ``--workers N`` or ``-w N`` (space or =) inside GUNICORN_CMD_ARGS.
_WORKER_ARG_RE = re.compile(r"(?:--workers|-w)[=\s]+(\d+)")


def _warn_unknown_ai_effort() -> None:
    """WARN (not fail) on an unrecognized ``RAVEN_AI_EFFORT`` /
    ``RAVEN_AI_EFFORT_COMMENT``. An unknown effort is a silent footgun on the
    openai_compatible backend — ``_EFFORT_TO_REASONING`` maps it to ``None``,
    disabling reasoning entirely, indistinguishable from a deliberate
    ``none``. Only an explicitly-set unknown value warns; unset (the default)
    is fine. (audit 07-02 #4)"""
    for var in ("RAVEN_AI_EFFORT", "RAVEN_AI_EFFORT_COMMENT"):
        raw = os.environ.get(var, "")
        if raw.strip() and raw.strip().lower() not in _KNOWN_AI_EFFORTS:
            logger.warning(
                "%s=%r is not a recognized effort %s — the openai_compatible "
                "backend maps unknown values to no reasoning (silently "
                "disabling it); claude_cli passes it through to the CLI.",
                var, raw, sorted(_KNOWN_AI_EFFORTS),
            )


def _assert_single_worker() -> None:
    """Fail fast on a multi-worker gunicorn deploy. Raven's dedup /
    in-progress / findings-cache state is all process-local, so >1 worker
    silently disables the double-review and double-merge guards (each worker
    keeps its own copy). Detect the common signals — ``WEB_CONCURRENCY`` and a
    ``--workers``/``-w`` flag in ``GUNICORN_CMD_ARGS`` — and raise.

    NOTE: a ``--workers N`` flag baked directly into the gunicorn CMD is argv
    on the master process and NOT visible in the worker's environment, so this
    is best-effort; the README documents the single-worker requirement.
    (audit-06-13 #9)"""
    def _too_many(n: str) -> bool:
        return n.isdigit() and int(n) > 1

    msg = (
        "Raven requires a single gunicorn worker but {n} are configured "
        "({src}). All dedup / in-progress / findings-cache state is "
        "process-local, so >1 worker silently disables the double-review and "
        "double-merge guards. Set WEB_CONCURRENCY=1 and use --workers 1."
    )
    wc = os.environ.get("WEB_CONCURRENCY", "").strip()
    if _too_many(wc):
        raise RuntimeError(msg.format(n=wc, src="WEB_CONCURRENCY"))
    m = _WORKER_ARG_RE.search(os.environ.get("GUNICORN_CMD_ARGS", ""))
    if m and _too_many(m.group(1)):
        raise RuntimeError(msg.format(n=m.group(1), src="GUNICORN_CMD_ARGS"))


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
        bb_dc_username = os.environ.get("BITBUCKET_DC_USERNAME", "").strip()
        # Fail fast rather than at submit time. BB DC tokens have no whoami
        # endpoint, so get_authenticated_user() raises without this value —
        # and it is called ONLY on submit_review's needs-work path. The
        # resulting failure is asymmetric and confusing: approvals post
        # normally while every review WITH findings raises and the author
        # sees a generic internal-error comment instead of the findings.
        # The "no providers configured" message below already lists this
        # variable as required; this makes the code agree. (audit 07-30)
        if not bb_dc_username:
            raise RuntimeError(
                "BITBUCKET_DC_USERNAME is required when Bitbucket DC is "
                "configured. BB DC tokens expose no whoami endpoint, so Raven "
                "cannot resolve its own account without it — reviews that "
                "request changes would fail at submit time while approvals "
                "succeeded. Set it to the service account's BB DC slug."
            )
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

    # Enforce the single-worker invariant before anything else (all safety
    # state is process-local).
    _assert_single_worker()

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

    # Validate the AI backend at startup (fail fast, like the provider check
    # above) rather than surfacing an opaque per-PR error while /healthz still
    # reads healthy. get_backend() raises on missing creds / unknown backend
    # and caches the instance; the effort warning is advisory. (audit 07-02 #4)
    get_backend()
    _warn_unknown_ai_effort()

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


def _approve_from_severity(severity: str, scale: SeverityScale) -> bool:
    """Does this review severity permit an approve verdict?

    Replaces ``severity_gte(REVIEW_APPROVE_MAX_SEVERITY, severity)``, which
    reads the built-in three-tier vocabulary and therefore ranks every
    custom tier name at 0 — tying with the threshold and approving
    everything, including the repo's most severe tier.

    ``scale.blocks()`` normalizes unknown names to the most severe tier, so
    an unrecognised value fails closed here exactly as it does in
    ``_validate_review`` (PR #211).
    """
    # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
    return not scale.blocks(severity)


def _maybe_dispatch_cached_merge(provider: GitProvider, repo_full_name: str,
                                 pr_number: int, pr_title: str, pr_url: str,
                                 head_sha: str | None = None,
                                 current_hashes: dict[str, str] | None = None,
                                 expected_config_hash: str | None = None,
                                 scale: SeverityScale | None = None,
                                 source: str = "no_changes",
                                 *, scale_fetch_failed: bool,
                                 policy_unusable: bool) -> bool:
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
      * ``policy_unusable`` (keyword-only, no default, like the next
        one) — repo policy the caller couldn't read or validate (the
        no-changes skip: CLAUDE.md, the rules, the review override or an
        invalid severities.json; the comment flow also the respond override and an
        unresolvable base ref), so the cached verdict
        can't be checked against the policy that now applies (09-27 #12);
      * ``scale_fetch_failed`` (keyword-only, no default: a caller that
        forgot it would otherwise skip this gate silently) — the caller's
        severities.json read failed,
        so ``scale`` is a fallback (``default_scale()``, or a legacy-path
        scale) standing in for the repo's real one. The config-hash gate
        below can't catch that for an entry recorded under the same
        fallback (the hashes agree), while the repo may by now gate on a
        stricter scale, so it declines outright — the same fail-closed
        call as the review path;
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
      * when the caller supplies ``expected_config_hash`` (the per-repo
        policy this dispatch would be judged under, right now: scale,
        prompt override, base branch, CLAUDE.md and rules), it must equal
        ``entry.config_hash`` exactly — otherwise the cached approve was
        computed under policy that no longer applies, and re-dispatching it without a
        fresh AI pass could auto-merge under a stale gate (severity-scale
        cache-safety half of the per-entry config hash; see
        ``_entry_config_hash``). This DELIBERATELY includes a legacy
        entry with ``config_hash == ""`` (written before this check
        existed): unlike ``_process_pr``'s read path, where a "" mismatch
        triggers a fresh review that then records a real hash and
        re-warms the entry, this is the no-review path — there is no
        write to re-warm from, so treating "" as "skip the comparison"
        would let a hash-less entry auto-merge under any scale forever.
        Only a caller that omits the argument entirely (every call site
        predating this feature, and every existing test built around
        that default) skips the comparison — additive, not a new blanket
        requirement;
      * PR must be open (mirrors _process_comment's fail-closed state gate);
      * Raven must be the sole reviewer (shared ``_is_sole_reviewer`` gate).

    Outcomes land in ``raven_cached_merge_dispatch_total{outcome,repo,source}``
    as ``dispatched`` / ``declined_<reason>``; ``source`` names the caller
    (``no_changes`` — the push-flow skip — or ``comment``).

    ``scale`` is the repo's resolved severity scale, used only to render
    the synthesized notify payload's ``severity`` field correctly for a
    custom vocabulary — it plays no role in any of the gates above (the
    dispatch decision is entry.verdict == "approve", not a severity
    comparison). Omitting it falls back to ``default_scale()``, under
    which an unranked custom tier name silently collapses to the
    default scale's least-severe reading (task-14 defect class: found
    outside Task 14's assigned scope, fixed as part of it per team-lead
    ruling — same file, same shape).
    """
    # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
    def _decline(reason: str) -> bool:
        inc("raven_cached_merge_dispatch_total",
            {"outcome": f"declined_{reason}", "repo": repo_full_name,
             "source": source})
        return False

    scale = scale or default_scale()
    pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"

    if RAVEN_REVIEW_MODE == "advisory":
        logger.info("PR #%d: advisory mode — cached merge dispatch does not apply",
                    pr_number)
        return _decline("advisory_mode")

    if policy_unusable:
        logger.warning("PR #%d: a repo policy file could not be read or "
                       "validated — declining cached merge dispatch "
                       "(fail-closed)", pr_number)
        return _decline("policy_unusable")

    if scale_fetch_failed:
        logger.warning("PR #%d: severities.json could not be read — declining "
                       "cached merge dispatch (fail-closed)", pr_number)
        return _decline("scale_fetch_failed")

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
        # The diff fetched here must describe head_sha, before and after the
        # fetch — Gitea's diff ref lags the head after a push (09-27 #21).
        if _diff_head_mismatch(provider, repo_full_name, pr_number,
                               head_sha, "diff_ref_lag") is not None:
            logger.info("PR #%d: diff does not describe head %s — declining "
                        "cached merge dispatch (fail-closed)", pr_number, head_sha[:8])
            return _decline("diff_head_unbound")
        try:
            current_hashes = _diff_chunk_hashes(
                provider.fetch_pr_diff(repo_full_name, pr_number))
        except Exception as e:
            logger.warning("PR #%d: could not fetch diff for cached merge "
                           "dispatch: %s — declining (fail-closed)", pr_number, e)
            return _decline("diff_fetch_failed")
        if _diff_head_mismatch(provider, repo_full_name, pr_number,
                               head_sha, "head_moved") is not None:
            logger.info("PR #%d: diff ref moved during the fetch — declining "
                        "cached merge dispatch (fail-closed)", pr_number)
            return _decline("diff_head_unbound")

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
        if (expected_config_hash is not None
                and entry.config_hash != expected_config_hash):
            logger.info("PR #%d: cached approval's review config hash does not "
                        "match the repo's current severity scale / prompt "
                        "override — skipping cached merge dispatch", pr_number)
            return _decline("config_hash_mismatch")
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
        "severity": _max_severity_from_findings(remaining_findings, scale),
        "summary": cached_summary,
        "findings": remaining_findings,
        "severity_scale_names": scale.ordered(),
        "severity_blocks_at": scale.blocks_at_or_above,
    }
    merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
    logger.info("PR #%d: dispatching auto-merge from cached approve verdict "
                "(head %s)", pr_number, head_sha[:8])
    inc("raven_cached_merge_dispatch_total",
        {"outcome": "dispatched", "repo": repo_full_name, "source": source})
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
# Cap on prior findings offered to a re-review for keep-or-resolve,
# ranked by severity, then by whether the thread has replies. Overflow is
# kept verbatim on its threads (like the carried overflow): not offered,
# never resolved, never dropped.
RAVEN_PRIOR_FINDINGS_MAX = max(
    int(os.environ.get("RAVEN_PRIOR_FINDINGS_MAX", "30")), 1)


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


def _remap_carried_lines(
    old_hunks: list, new_hunks: list, findings: list[dict],
    old_context: list | None = None, new_context: list | None = None,
) -> dict[int, int] | None:
    """Map each finding's ``line`` from ``old_hunks`` onto ``new_hunks``.

    Called only for a file whose ``content_hashes`` entry is unchanged —
    the PR's own added/removed lines are byte-identical — but whose hunks
    sit at different absolute positions because the base branch moved
    under them. Findings carry the ``line`` the model reported at review
    time (per ``prompts/review.md``, a NEW-side line number), so re-posting
    them unshifted anchors each inline comment to whatever the rebase slid
    into that position.

    Returns ``{old_line: new_line}`` for the lines that need moving, or
    ``None`` when the mapping cannot be made safely. ``None`` is a
    fail-safe, not an error: the caller puts the file back into
    ``changed_files``, which restores exactly the pre-rebase-tolerance
    behaviour (re-review the file, regenerate its findings) for the cases
    this cannot prove. It happens when

      * the hunk count differs — a base edit landing within context range
        can merge two hunks into one, so ``hunks[i]`` no longer describes
        the same edit on both sides;
      * a matched hunk's new-side length differs, which means the same
        thing at a finer grain; or
      * a finding's line falls outside every recorded hunk (a stale or
        hand-written ``line``) and so has no delta to shift by.

    Findings with no usable ``line`` (PR-wide notes, ⚠️ coverage-gap
    markers) are skipped rather than rejected — they post no inline
    comment, so they have nothing to anchor.
    """
    if len(old_hunks) != len(new_hunks):
        return None
    # Geometry alone cannot tell a rebase from a relocation: the author
    # moving a byte-identical edit elsewhere in the file leaves the
    # content hash equal AND the hunk shape intact, so every check below
    # passes and the finding is carried onto code that was never
    # reviewed in its new position. Context content is the discriminator
    # — a rebase keeps it byte-identical while the position moves. Only
    # compared when both sides recorded it; an entry from before the
    # field degrades to the previous behaviour rather than re-reviewing
    # everything.
    if (old_context and new_context
            and list(old_context) != list(new_context)):
        return None
    pairs = [(tuple(o), tuple(n)) for o, n in zip(old_hunks, new_hunks)]
    mapping: dict[int, int] = {}
    for f in findings:
        line = f.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
            continue
        for (old_start, old_len), (new_start, new_len) in pairs:
            if old_start <= line < old_start + old_len:
                if old_len != new_len:
                    return None
                if new_start != old_start:
                    mapping[line] = line + (new_start - old_start)
                break
        else:
            return None
    return mapping


def _finding_at_remapped_line(finding: dict, mapping: dict[int, int] | None) -> dict:
    """``finding`` with its ``line`` shifted per ``mapping``, or unchanged.

    Returns a copy when it shifts — the cached finding dict is shared with
    the entry the comment flow reads under ``_previous_diffs_lock``, so
    the line move must not be an in-place mutation of it.
    """
    if not mapping:
        return finding
    new_line = mapping.get(finding.get("line"))
    if new_line is None:
        return finding
    return {**finding, "line": new_line}


# The inline comment Raven posts for a finding, and its reader. Kept side
# by side because get_review_threads hands back only an untracked
# thread's body: _parse_inline_body must read exactly what
# _format_inline_body wrote, so a change to one changes the other.
_INLINE_BODY_RE = re.compile(
    r"\A\S+ \*\*\[(?P<sev>[^\]\n]+)\]\*\* (?P<msg>.+)\Z", re.DOTALL)


def _format_inline_body(f: dict, scale: SeverityScale) -> str:
    """``{emoji} **[{tier}]** {message}``.

    The default tier is the SCALE's least severe one, not the literal
    'low': a finding missing its severity key (e.g. a malformed or legacy
    cache entry carried forward — CacheEntry.findings loads straight from
    JSON with no per-finding validation) must still render a name and a
    colour that exist in THIS repo's vocabulary.
    """
    sev = f.get("severity") or scale.least_severe
    return f"{scale.emoji(sev)} **[{sev}]** {f['message']}"


def _parse_inline_body(body: str | None, scale: SeverityScale) -> tuple[str, str] | None:
    """``(tier, message)`` from a body ``_format_inline_body`` wrote, or
    ``None`` when it isn't one or its tier isn't on ``scale`` (a scale
    change since it was posted): such a thread is left alone."""
    m = _INLINE_BODY_RE.match(body or "")
    if not m or not scale.is_known(m.group("sev")):
        return None
    return m.group("sev").strip().lower(), m.group("msg")


class _PriorSet(NamedTuple):
    findings: list[dict]   # offered to the model; prior_id = index
    replies: list[int]     # reply count per offered finding
    tracked: set[int]      # id() of offered findings that are cache dicts
    overflow: list[dict]   # over RAVEN_PRIOR_FINDINGS_MAX: kept verbatim
    moot: list[dict]       # on removed files: resolved after submit
    untracked_acted: int   # untracked open threads offered or moot
    moved: dict[int, str]  # id() of an offered finding from a renamed file -> its target


def _collect_prior_findings(threads: list[dict] | None,
                            cached_findings: dict[str, list[dict]],
                            scope: set[str], removed: set[str],
                            resolved_ids: set, gap_files: set[str],
                            scale: SeverityScale, cap: int,
                            renamed: dict[str, str] | None = None) -> _PriorSet:
    """The open prior findings on the files a re-review covers (``scope``).

    ``threads`` is ``provider.get_review_threads`` (``None``: unsupported
    or failed, so only the cache is read). A tracked finding — its
    ``comment_id`` in the cache — is offered as the cache's own dict,
    whose severity, message and remapped line win. A missing listing entry
    never means resolved: only an explicit ``resolved`` flag or
    ``resolved_ids`` does, so a partial listing can't drop a tracked
    blocker. An untracked open thread is parsed from its body and offered
    when it sits on a scope file; one that doesn't parse, or names a tier
    this scale lacks, is left alone. Findings on ``removed`` files (the
    PR no longer changes them) are moot — except a rename source
    (``renamed``: removed file -> the current file it was renamed to),
    whose code still ships under the target: its findings are offered as
    priors on the target when that is in scope (``moved`` records the
    target for a cache dict; an orphan is built on it), and are otherwise
    left alone, never resolved. Gap markers keep their own
    lifecycle and are never offered. ``untracked_acted`` counts only the
    untracked threads this pass acts on (offered or moot): one left alone
    would be recounted by every review of the PR.
    """
    listed = {t["comment_id"]: t for t in threads or []
              if isinstance(t.get("comment_id"), int)}
    closed = set(resolved_ids) | {cid for cid, t in listed.items() if t.get("resolved")}
    tracked_cids = {f.get("comment_id") for fl in cached_findings.values() for f in fl}

    def _open(f: dict) -> bool:
        return (not _is_coverage_gap_marker(f, gap_files)
                and f.get("comment_id") not in closed)

    renamed = renamed or {}
    gone = set(removed) - renamed.keys()
    offered = [f for fname in sorted(scope)
               for f in cached_findings.get(fname, []) if _open(f)]
    moved: dict[int, str] = {}
    for src, dst in sorted(renamed.items()):
        if dst in scope:
            for f in cached_findings.get(src, []):
                if _open(f):
                    offered.append(f)
                    moved[id(f)] = dst
    tracked = {id(f) for f in offered}
    moot = [f for fname in sorted(gone) for f in cached_findings.get(fname, [])
            if _open(f) and f.get("comment_id") is not None]
    untracked_acted = 0
    for cid, t in listed.items():
        if cid in tracked_cids or cid in closed:
            continue
        parsed = _parse_inline_body(t.get("body"), scale)
        home = renamed.get(t.get("file"), t.get("file"))
        if parsed is None or not (home in scope or t.get("file") in gone):
            continue
        untracked_acted += 1
        orphan = {"severity": parsed[0], "file": home, "line": t.get("line"),
                  "message": parsed[1], "comment_id": cid}
        (offered if home in scope else moot).append(orphan)
    replies = [int(listed.get(f.get("comment_id"), {}).get("replies") or 0)
               for f in offered]
    overflow: list[dict] = []
    if len(offered) > cap:
        # scale.rank() fails closed (unknown -> most severe): a tier the
        # current scale lost ranks first, so it is re-judged under this
        # scale instead of sitting in the overflow. Either way the kept copy counts as the most severe
        # tier (see _process_pr).
        order = sorted(range(len(offered)), key=lambda i: (
            -scale.rank(offered[i].get("severity")), -(replies[i] > 0)))
        top = set(order[:cap])
        overflow = [f for i, f in enumerate(offered) if i not in top]
        replies = [r for i, r in enumerate(replies) if i in top]
        offered = [f for i, f in enumerate(offered) if i in top]
    return _PriorSet(offered, replies, tracked, overflow, moot, untracked_acted, moved)


def _resolve_finding_threads(provider: GitProvider, repo_full_name: str,
                             pr_number: int, findings: list[dict], reason: str,
                             skip: set | frozenset = frozenset()) -> None:
    """Resolve each finding's platform thread, best-effort and after a
    successful submit only. A failed resolve logs and moves on."""
    for f in findings:
        cid = f.get("comment_id")
        if cid is None or cid in skip:
            continue
        try:
            ok = provider.retract_finding(repo_full_name, pr_number, cid)
        except Exception as e:
            ok = False
            logger.warning("retract_finding for %s finding (comment %s) on PR #%d "
                           "failed: %s", reason, cid, pr_number, e)
        inc("raven_retractions_total",
            {"repo": repo_full_name, "result": "ok" if ok else "fail"})


def _process_pr(provider: GitProvider, payload: dict) -> None:
    """Review a PR in a background thread. All exceptions caught here."""
    repo_full_name = None
    pr_number = None
    pr_key = None
    posted_head = None   # the head this run posted a review for, if any
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
                # Latest push wins — except that an event for the head
                # already under review never displaces a parked NEWER head:
                # it would be dropped as "just reviewed" and the newer push
                # would never re-run.
                parked = _rerun_requested.get(pr_key)
                running = _in_progress_heads.get(pr_key)
                if not (parked is not None and head_sha == running
                        and parked[1].get("head_sha") != running):
                    _rerun_requested[pr_key] = (provider, payload)
                logger.info("PR %s already being reviewed — re-running for %s "
                            "when it finishes", pr_key, head_sha[:8])
                inc("raven_reviews_skipped_total", {"reason": "in_progress", "repo": repo_full_name})
                pr_key = None  # don't clear the entry in finally
                return
            _in_progress_prs.add(pr_key)
            _in_progress_heads[pr_key] = head_sha

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

        # Fetch diff — only once it describes head_sha, and check it still
        # did after the fetch. On Gitea the diff is built from a ref that
        # lags the head for a moment after a push; paired with head B, a
        # diff of A put A's verdict on B, and with a cached approve of A
        # the no-changes skip merged B unreviewed (audit 2026-09-27 #21).
        # A head that moved on is a newer push: park a re-run for it (its
        # own event may never come — a bot author, a lost webhook) and stop
        # without a failure; a re-run needs a real head change each time,
        # so this can't spin. Anything else fails closed as a classified
        # failure. A payload with no SHA is bound to the head read here.
        # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
        if head_sha == "HEAD":
            read_head = None
            for attempt in range(_HEAD_READ_TRIES):
                if attempt:
                    time.sleep(_DIFF_HEAD_POLL_INTERVAL)
                try:
                    read_head = provider.get_pr_head_sha(repo_full_name, pr_number)
                except Exception as e:
                    logger.warning("get_pr_head_sha for PR #%s failed: %s", pr_number, e)
                    read_head = None
                if isinstance(read_head, str) and read_head:
                    break
            if not (isinstance(read_head, str) and read_head):
                raise DiffHeadUnverifiedError(f"PR #{pr_number}: no head SHA to bind the diff to")
            head_sha = read_head
            with _in_progress_lock:
                _in_progress_heads[pr_key] = head_sha
        unbound, moved_to = _await_diff_head(provider, repo_full_name, pr_number, head_sha)
        if unbound is None:
            diff = provider.fetch_pr_diff(repo_full_name, pr_number)
            # A failed read is retried briefly; a mismatch never is — the
            # diff may then describe either commit. A check that never ran
            # counts as unverified.
            unbound = "head_unknown"
            for attempt in range(_HEAD_READ_TRIES):
                if attempt:
                    time.sleep(_DIFF_HEAD_POLL_INTERVAL)
                unbound = _diff_head_mismatch(provider, repo_full_name, pr_number,
                                              head_sha, "diff_ref_lag")
                if unbound in (None, "diff_ref_lag"):
                    break
            if unbound is not None:
                moved_to = _moved_head(provider, repo_full_name, pr_number, head_sha)
                if moved_to is not None:
                    unbound = "head_moved"
        if unbound == "head_moved":
            logger.info("PR #%s: head moved %s -> %s before its diff could be "
                        "read — re-running for the new head",
                        pr_number, head_sha[:8], moved_to[:8])
            inc("raven_reviews_skipped_total",
                {"reason": "head_moved", "repo": repo_full_name})
            _park_rerun(pr_key, provider, payload, moved_to)
            return
        if unbound is not None:
            raise DiffHeadUnverifiedError(
                f"PR #{pr_number}: diff does not describe {head_sha[:8]} ({unbound})")

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
        # Policy sources this pass couldn't read or validate (audit 09-27
        # #12). Any of them forces needs_work: the review still posts, and
        # its body names what couldn't be read, but it can't approve or
        # merge on policy Raven never saw. Static labels only.
        policy_unusable: list[str] = []
        # Whether any of them is something other than an unreadable
        # severities.json, which the dispatcher reports on its own
        # (declined_scale_fetch_failed). Set here, at the source, rather
        # than re-derived from the display labels.
        other_policy_unusable = False

        def _policy_unusable(label: str, *, scale_read: bool = False) -> None:
            nonlocal other_policy_unusable
            if label not in policy_unusable:
                policy_unusable.append(label)
            if not scale_read:
                other_policy_unusable = True

        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref=base_ref)
        except Exception as e:
            # 404 (missing file) returns "" without raising; reaching this
            # except means an auth/transport/server-side failure that the
            # operator probably wants to see. Warn instead of debug.
            logger.warning("CLAUDE.md fetch for %s@%s failed (review proceeds without repo context, "
                           "and can't approve): %s", repo_full_name, base_ref, e)
            _policy_unusable("`CLAUDE.md`")
        # The rules, review override and scale are read further down,
        # before the no-changes skip: the cache entry is bound to them
        # (audit 09-27 #8), so even an unchanged diff needs them.

        # Guard: empty diff after stripping lockfiles/binaries
        stripped = strip_diff(diff)
        clean_diff = stripped.clean
        # Stripped files' gaps (a changed lockfile, D2 (b)) come from the
        # whole diff on every pass, so they are never carried. A lockfile a
        # rename brought into the clean diff is carried like any reviewed
        # file: its marker dedupes against the fresh one, and it stops
        # being a gap only by leaving the diff, which forces a full review.
        stripped_now = set(stripped.stripped)
        # Every lockfile the PR changes, held here and unioned into the
        # effective gap below, so the merge block doesn't depend on every
        # review_diff return path echoing it back.
        lockfile_gaps = _lockfile_gaps(diff)
        if not clean_diff.strip():
            provider.post_pr_comment(repo_full_name, pr_number,
                "🦅 **Raven Review**\n\nEmpty diff after stripping lockfiles/binaries — skipping review.")
            inc("raven_reviews_skipped_total", {"reason": "empty_diff", "repo": repo_full_name})
            return

        # Incremental review: only review files that changed since last review
        pr_key = f"{provider.name}:{repo_full_name}#{pr_number}"
        file_chunks = {f: c for f, c in split_diff_by_file(clean_diff)}
        # Two hashes per file, answering two different questions — see
        # CacheEntry.content_hashes. ``current_hashes`` (raw chunk) is
        # "is this literally the same diff?" and gates the no-changes
        # skip and the cached-merge dispatch under it; it covers every
        # file the PR changes, stripped ones included (see
        # _diff_chunk_hashes). ``current_content_hashes`` is "did the
        # PR's own edits to this file change?" and picks the incremental
        # delta, over the files the model can review.
        current_hashes = _diff_chunk_hashes(diff)
        current_content_hashes = {f: diff_hash(c) for f, c in file_chunks.items()}
        current_hunks = {f: hunk_positions(c) for f, c in file_chunks.items()}
        current_hunk_context = {f: hunk_context_digests(c)
                                for f, c in file_chunks.items()}
        now = time.time()
        with _previous_diffs_lock:
            cached = _previous_diffs.get(pr_key)
            if cached:
                previous_hashes = cached.hashes
                previous_content_hashes = cached.content_hashes
                previous_hunks = cached.hunks
                previous_hunk_context = cached.hunk_context
                cached_findings = cached.findings
            else:
                previous_hashes = {}
                previous_content_hashes = {}
                previous_hunks = {}
                previous_hunk_context = {}
                cached_findings = {}

        # Raw-chunk delta: the literal "did anything at all move?" set.
        # Only this one may reach the no-changes skip, because the cached
        # merge dispatched from there approves the head WITHOUT a fresh
        # review — that shortcut stays limited to a byte-identical diff.
        raw_changed_files = {f for f, h in current_hashes.items()
                             if previous_hashes.get(f) != h}
        # Content delta: the set actually worth re-reviewing. A file the
        # rebase only slid around is not in here, so it is not re-reviewed
        # and its findings (with their comment_ids, hence the developer's
        # resolutions) carry forward instead of being regenerated.
        # An entry from a cache file written before content_hashes existed
        # has nothing to compare against, so it falls back to the raw
        # delta — exactly the pre-existing behaviour, not "everything
        # changed". DIFF_HASH_SCHEME wipes the cache once on upgrade
        # anyway; this only has to be sane in the meantime.
        changed_files = (
            {f for f, h in current_content_hashes.items()
             if previous_content_hashes.get(f) != h}
            if previous_content_hashes else set(raw_changed_files)
        )
        # The repo policy this pass is judged under, read before the
        # no-changes skip: the cache entry is bound to it (audit 09-27 #8),
        # so a change since the cached review sends the PR to a full
        # review instead of the skip or an incremental pass.
        rules = _fetch_rules(provider, repo_full_name, base_ref,
                             on_fetch_failed=lambda: _policy_unusable("the review rules"))

        # Deprecated-config-path nag. Same local-closure shape as
        # scale_fetch_failed below and for the same reason: _process_pr
        # runs one PR to completion per call, so a closure carries this
        # without any cross-repo module state.
        legacy_config_paths: list[str] = []

        def _note_legacy_config_path(relpath: str) -> None:
            if relpath not in legacy_config_paths:
                legacy_config_paths.append(relpath)

        review_prompt_override = _fetch_prompt_override(
            provider, repo_full_name, base_ref, "review",
            on_legacy_path=_note_legacy_config_path,
            on_fetch_failed=lambda: _policy_unusable("the review prompt override"),
        )
        # An unreadable severities.json, or an invalid one, joins
        # policy_unusable (fail the merge gate closed below — see
        # coverage_gap for the same pattern); only "no file" is a silent
        # default_scale() fall-back. Local state, not module state:
        # _process_pr runs one PR to completion per call, so a closure is
        # enough and stays free of the cross-repo leak a shared "current
        # scale" would risk (see CLAUDE.md's "Severity scale rules" on why
        # scale is always threaded as a parameter, never global).
        scale_fetch_failed = False

        def _mark_scale_fetch_failed() -> None:
            nonlocal scale_fetch_failed
            scale_fetch_failed = True
            _policy_unusable("`severities.json`", scale_read=True)

        scale = _fetch_severity_scale(provider, repo_full_name, base_ref,
                                      on_fetch_failed=_mark_scale_fetch_failed,
                                      on_invalid=lambda: _policy_unusable("`severities.json` (invalid)"),
                                      on_legacy_path=_note_legacy_config_path)
        entry_config_hash = _entry_config_hash(
            scale, review_prompt_override,
            base_ref=base_ref, claude_md=claude_md, rules=rules)
        # An empty hash (an entry from before the hash existed) counts as
        # a change too, as on the cached-merge path.
        config_changed = cached is not None and cached.config_hash != entry_config_hash

        removed_files = previous_hashes.keys() - current_hashes.keys()
        # Policy Raven couldn't read hashes as if it were absent, so it
        # reads as a change; the skip declines the merge on it instead of
        # running a full review that the next healthy trigger would repeat.
        if (previous_hashes and not raw_changed_files and not removed_files
                and (not config_changed or policy_unusable)):
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
            #
            # The cached approve is reused WITHOUT a fresh AI pass, so it
            # must be checked against the repo's config right now — a
            # severities.json, prompt-override, CLAUDE.md or rules edit
            # landing on base_ref since the cached review must not silently
            # keep auto-merging under policy that no longer applies. A
            # changed policy never reaches here (config_changed sends it to
            # a full review below); the helper's own entry.config_hash
            # comparison stays as the gate, refusing a hash-less entry like
            # any other mismatch (see CacheEntry.config_hash's and
            # _maybe_dispatch_cached_merge's docstrings). Policy Raven
            # couldn't read or validate blocks the merge as it blocks a
            # fresh review.
            _maybe_dispatch_cached_merge(
                provider, repo_full_name, pr_number,
                pr_title, pr_url, head_sha=head_sha,
                current_hashes=current_hashes,
                expected_config_hash=entry_config_hash,
                scale=scale,
                scale_fetch_failed=scale_fetch_failed,
                policy_unusable=other_policy_unusable)
            return

        # Line remap for the files the rebase only slid around: content
        # equal (so not in changed_files, so not re-reviewed) but sitting
        # at new absolute positions. Their carried findings still hold the
        # line numbers from the previous review, which now point at
        # whatever the base branch moved into that place. Runs BEFORE the
        # delta is finalised because a finding that cannot be mapped
        # safely sends its whole file back into changed_files.
        remapped_lines: dict[str, dict[int, int]] = {}
        for fname, new_h in current_hunks.items():
            if fname in changed_files or fname in removed_files:
                continue
            old_h = previous_hunks.get(fname)
            if not old_h:
                continue
            if [tuple(x) for x in old_h] == [tuple(x) for x in new_h]:
                # Same positions, so nothing to remap — but the body can
                # still differ: an added line moved across a context line
                # inside its hunk keeps both the content hash and the
                # geometry (audit 09-27 #5). Compared only when both sides
                # recorded it, like the remap below.
                old_ctx = previous_hunk_context.get(fname)
                new_ctx = current_hunk_context.get(fname)
                if old_ctx and new_ctx and list(old_ctx) != list(new_ctx):
                    logger.info("PR #%d: %s kept its positions but its hunk body "
                                "changed — re-reviewing the file", pr_number, fname)
                    changed_files.add(fname)
                continue
            mapping = _remap_carried_lines(
                old_h, new_h, cached_findings.get(fname, []),
                previous_hunk_context.get(fname),
                current_hunk_context.get(fname))
            if mapping is None:
                logger.info(
                    "PR #%d: %s moved but its findings can't be remapped safely "
                    "— re-reviewing the file", pr_number, fname)
                changed_files.add(fname)
            elif mapping:
                logger.info("PR #%d: %s shifted — remapping %d carried finding line(s)",
                            pr_number, fname, len(mapping))
                remapped_lines[fname] = mapping

        # Apply the shift to the cached entry itself rather than to copies
        # taken further down. Everything downstream — the drop-or-keep
        # prompt, the carried/fresh dedupe, the inline comments, the
        # post-submit write — reads these same dicts, and the write
        # matches the model's drops by object identity, so a copy would
        # silently un-drop them. Recording a corrected line number is not
        # a finding-set change (the "no cache effects before submit" rule
        # guards drops, which alter the standing review's content);
        # retraction matches on comment_id, never on line.
        if remapped_lines:
            with _previous_diffs_lock:
                live_entry = _previous_diffs.get(pr_key)
                if live_entry is not None:
                    live_entry.findings = {
                        fname: [_finding_at_remapped_line(f, remapped_lines.get(fname))
                                for f in fl]
                        for fname, fl in live_entry.findings.items()
                    }
                    # Only the remapped files' geometry moves with their
                    # shifted lines. Writing every file's new geometry here
                    # laundered a relocation this pass had just found: if
                    # the review then failed, the next pass compared the
                    # relocated file against its own geometry and carried
                    # it unreviewed (audit 09-27 #5). The rest is written
                    # at the post-submit write, after a review covered it.
                    live_entry.hunks = {**live_entry.hunks, **{
                        f: current_hunks[f] for f in remapped_lines}}
                    live_entry.hunk_context = {**live_entry.hunk_context, **{
                        f: current_hunk_context[f] for f in remapped_lines}}
                    cached = live_entry
                    cached_findings = live_entry.findings

        is_incremental = False
        unchanged_files: list[str] = []
        if previous_hashes and removed_files:
            # Files were removed — do a full review to clear stale findings;
            # their threads resolve as moot (_collect_prior_findings)
            logger.info("PR #%d files removed since last review — full re-review", pr_number)
            review_diff_text = clean_diff
        elif previous_hashes and config_changed:
            # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
            # The cached review was judged under other policy: the scale,
            # the prompt override, the base branch, CLAUDE.md or the rules
            # (audit 09-27 #8). Carrying its findings and verdict into an
            # incremental pass, or keeping it through the rebase-only
            # shortcut, would apply the old policy to this head. Review the
            # whole head; open findings ride in as prior findings, each tier
            # read through scale.normalize(), so still-valid ones stay on
            # their threads.
            logger.info("PR #%d: review policy changed since the cached review — "
                        "full re-review", pr_number)
            inc("raven_config_change_full_reviews_total", {"repo": repo_full_name})
            review_diff_text = clean_diff
        elif previous_hashes and changed_files:
            # Incremental: rebuild diff from only changed file chunks.
            # The unchanged files are declared to review_diff so the
            # prompt discloses the delta scope — without it the model
            # judges PR-level claims ("the implementation is missing
            # from this PR") from files it was never shown.
            is_incremental = True
            unchanged_files = sorted(set(file_chunks) - changed_files)
            review_diff_text = "".join(file_chunks[f] for f in sorted(changed_files))
            logger.info("PR #%d incremental review: %d/%d files changed", pr_number, len(changed_files), len(file_chunks))
        elif (previous_hashes and cached is not None and cached.verdict == "approve"
              and RAVEN_REVIEW_MODE != "advisory"):
            # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
            # An APPROVED PR whose diff moved without changing its own edits
            # (a rebase or base merge) gets a full review of the rebased
            # head. The shortcut below would leave the approve standing
            # against a head no review saw (audit 2026-09-27 #6). Costs one
            # review per rebase of an approved-but-unmerged PR. Its open
            # findings ride in as prior findings, so the still-valid ones
            # stay on their threads; a finding the comment flow retracted
            # isn't offered and can come back, as can an approve the
            # comment flow granted. Advisory mode never merges, so it keeps
            # the shortcut.
            logger.info("PR #%d: approved PR rebased — full review of the new head",
                        pr_number)
            inc("raven_rebase_full_reviews_total",
                {"repo": repo_full_name, "reason": "approved"})
            review_diff_text = clean_diff
        elif (previous_hashes and cached is not None
              and cached.unreviewed_hashes == current_hashes):
            # The shortcut below already skipped this exact head once; a
            # second trigger for it is someone asking again. Review it —
            # nothing else would until a content push.
            logger.info("PR #%d: re-triggered on a rebased head the shortcut "
                        "skipped — full review", pr_number)
            inc("raven_rebase_full_reviews_total",
                {"repo": repo_full_name, "reason": "retrigger"})
            review_diff_text = clean_diff
        elif previous_hashes:
            # The diff moved but the PR's own edits did not — a rebase or
            # a merge from the base branch, or a change to stripped files
            # only (a lockfile, a skip-listed binary). There is no new
            # authored code to review, so re-reviewing would only
            # regenerate the standing findings and strand the developer's
            # resolutions (which match on comment_id). Record the new
            # positions, shift the carried findings onto them, and stop.
            #
            # entry.hashes is deliberately NOT advanced: it means "the raw
            # chunks of the head a review covered", and both the comment
            # flow's binding and the cached merge rely on that. Writing the
            # rebased hashes here let a comment flip approve and merge a
            # head no review saw, even from needs_work (audit 2026-09-27
            # #6). Positions and context are advanced so the next push
            # remaps from the right place; the next content push reviews.
            logger.info(
                "PR #%d: diff moved without changing the PR's own edits "
                "(rebase or base merge) — no re-review, no merge dispatch",
                pr_number)
            inc("raven_reviews_skipped_total",
                {"reason": "rebase_only", "repo": repo_full_name})
            # The finding lines were already shifted above; record the new
            # positions and context so the next push remaps from this head.
            # entry.hashes keeps the reviewed head's raw chunks (see above).
            with _previous_diffs_lock:
                live_entry = _previous_diffs.get(pr_key)
                if live_entry is not None:
                    live_entry.content_hashes = current_content_hashes
                    live_entry.hunks = current_hunks
                    live_entry.hunk_context = current_hunk_context
                    live_entry.unreviewed_hashes = current_hashes
            _save_cache()
            return
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


        # User-resolved-comment filter, pass 1 (pre-review). Findings
        # whose backing inline comment the developer marked resolved via
        # the platform UI (Gitea "Resolve conversation"; BB DC "Resolve thread"
        # → threadResolved=true) are excluded from the carry-forward set
        # — the model must never be asked to drop-or-keep a finding the
        # developer already dismissed. Symmetric to the AI-driven
        # retraction in _process_comment: that flow is Raven resolving
        # on the AI's behalf; this one is Raven respecting the user's
        # direct dismissal. A second fetch after the AI call (below)
        # catches resolutions landing during the multi-minute review.
        # Every re-review with a cache fetches it: the carried set
        # (incremental) and the prior findings (every re-review) both skip
        # resolved threads.
        #
        # NOTE: nothing is written to the cache here. ALL cache effects
        # of this pass — user-resolved drops, model drops — are applied
        # atomically at the post-submit cache write, so a failed
        # submit_review leaves the cache matching the standing platform
        # review (the comment-flow all-retracted backstop counts cached
        # findings; a premature wipe could synthesize flip-to-approve).
        resolved_ids: set = set()
        if cached_findings:
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
                # A stripped file's only findings are its lockfile gap
                # marker, recomputed from the whole diff on every pass:
                # carrying it would duplicate it, and outlive a fix.
                if fname not in changed_files and fname not in stripped_now:
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
            # Deliberately NOT scale.rank() — same reasoning as
            # reviewer._cap_findings: rank()/normalize() fail CLOSED
            # (unknown -> most severe) for MODEL-EMITTED severities, but
            # these candidates are already-validated cache entries, so an
            # unknown/missing severity here is a malformed-data bug. The
            # safe direction for capping is to drop it first rather than
            # let it crowd out a well-formed high finding for cap space —
            # falling back to the least-severe tier's rank matches the
            # pre-scale SEVERITY_ORDER.get(name, 0) behavior exactly.
            least_rank = scale.ranks[scale.least_severe]
            by_sev = sorted(
                range(len(carried_candidates)),
                key=lambda i: -scale.ranks.get(
                    str(carried_candidates[i].get("severity", "") or "")
                    .strip().lower(),
                    least_rank),
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

        # Prior findings on the files this pass re-reviews — Raven's open
        # threads (platform) joined with the cache — offered for
        # keep-or-resolve, so a still-valid finding stays on its thread
        # instead of being regenerated into a second one.
        # Only files whose code is in this pass's diff: changed_files is a
        # subset of file_chunks by construction today, and the
        # intersection keeps it that way if that ever changes — a prior
        # judged without its code in the prompt must never be superseded.
        rereviewed = ((set(changed_files) & set(file_chunks)) if is_incremental
                      else set(file_chunks))
        review_threads = None
        if bot_user:
            try:
                review_threads = provider.get_review_threads(repo_full_name, pr_number, bot_user)
            except Exception as e:
                logger.warning("get_review_threads for PR #%d failed: %s — prior "
                               "findings from the cache only", pr_number, e)
        if not isinstance(review_threads, list):
            review_threads = None
        # A removed file that a rename carried to a current file is not
        # gone: its findings are offered on the target (review 2575).
        renamed: dict[str, str] = {}
        if removed_files:
            aliases = _rename_aliases(diff)
            by_norm = {_normalize_path(k): k for k in current_hashes}
            for src in removed_files:
                target = by_norm.get(aliases.get(_normalize_path(src), ""))
                if target:
                    renamed[src] = target
        priors = _collect_prior_findings(
            review_threads, cached_findings, rereviewed, set(removed_files),
            resolved_ids, set(cached.coverage_gap_files or []) if cached else set(),
            scale, RAVEN_PRIOR_FINDINGS_MAX, renamed=renamed)
        if priors.untracked_acted:
            add("raven_untracked_open_threads_total", priors.untracked_acted,
                {"repo": repo_full_name})

        # Run review
        with Timer("raven_review_duration_seconds", {"repo": repo_full_name}):
            review = review_diff(
                review_diff_text, repo_full_name,
                claude_md=claude_md, file_contents=file_contents,
                omitted_files=omitted_files,
                stripped_files=stripped.stripped,
                lockfile_gaps=lockfile_gaps,
                pr_title=pr_title, pr_description=pr_description,
                pr_comments=pr_comments, bot_user=bot_user,
                rules=rules,
                prompt_override=review_prompt_override,
                is_incremental=is_incremental,
                unchanged_files=unchanged_files,
                carried_findings=revalidation_candidates or None,
                prior_findings=[
                    {**{k: f[k] for k in ("line", "message") if k in f},
                     "file": priors.moved.get(id(f), f.get("file")),
                     "severity": scale.normalize(f.get("severity")), "replies": r}
                    for f, r in zip(priors.findings, priors.replies)] or None,
                scale=scale,
            )
        # Ride the migration nag on the review dict, the same channel
        # unknown_severities uses to reach both body renderers. Set here
        # rather than at each _format_* call so the parse-error and
        # advisory bodies carry it too.
        if legacy_config_paths:
            review["legacy_config_paths"] = list(legacy_config_paths)
        if policy_unusable:
            review["policy_unusable"] = list(policy_unusable)

        # Save original findings before merging carried ones (used for cache write)
        fresh_findings = list(review.get("findings", []))

        # The keep-or-resolve answer. Superseding is the explicit outcome,
        # keeping the fail-safe: the prompt tells the model not to restate
        # what it keeps, so a prior is superseded (resolved + dropped)
        # ONLY when a well-formed answer from its call omits it. An
        # unanswered prior (missing or voided answer, failed chunk) is
        # kept verbatim, like the over-cap overflow — the same rule as
        # dropped_carried. A verbatim restatement — the same severity
        # too, since a new tier is the "raise it fresh" case — counts as
        # a keep, and the prior copy wins with its thread. A prior's tier
        # is read through scale.normalize() here and on every kept copy:
        # a cached tier the current scale lost fails closed to the most
        # severe, like model output, so a kept prior never counts less.
        prior_answer = review.pop("prior_answer", None) or {}
        answered: set[int] = set(prior_answer.get("answered") or ())
        kept_lines: dict[int, int | None] = dict(prior_answer.get("kept") or {})

        def _prior_key(f: dict) -> tuple:
            return (scale.normalize(f.get("severity")),
                    priors.moved.get(id(f), f.get("file")), f.get("line"),
                    f.get("message"))

        def _kept_copy(p: dict, line: int | None) -> dict:
            c = {**p, "severity": scale.normalize(p.get("severity")),
                 "file": priors.moved.get(id(p), p.get("file"))}
            if line:
                c["line"] = line
            return c

        # The fresh copy of a verbatim restatement is removed at the merge
        # (step 8), against only the kept priors that survive the
        # post-review resolved filter: if the prior's thread was resolved
        # mid-review, the model's restatement must still post.
        if priors.findings:
            prior_keys = {_prior_key(p): i for i, p in enumerate(priors.findings)}
            for f in fresh_findings:
                i = prior_keys.get(_prior_key(f))
                if i is not None:
                    kept_lines.setdefault(i, None)
        unanswered = [i for i in range(len(priors.findings)) if i not in answered]
        for i in unanswered:
            kept_lines.setdefault(i, None)
        # (copy, origin): the new line rides on a copy, never on the
        # shared cache dict; origin is how the cache write spots a
        # concurrent retraction.
        kept_pairs = [
            (_kept_copy(priors.findings[i], line), priors.findings[i])
            for i, line in sorted(kept_lines.items())
        ] + [(_kept_copy(f, None), f) for f in priors.overflow]
        superseded = [p for i, p in enumerate(priors.findings) if i not in kept_lines]

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
        # would otherwise stay in this pass's verdict and summary body
        # until the next push. One extra provider GET per incremental
        # review.
        if (is_incremental and kept_candidates) or kept_pairs or superseded or priors.moot:
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
                before_prior = len(kept_pairs)
                kept_pairs = [(k, o) for k, o in kept_pairs
                              if k.get("comment_id") not in resolved_ids]
                user_resolved_count += before_prior - len(kept_pairs)
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
        kept_priors = [k for k, _ in kept_pairs]
        if kept_pairs:
            # Verbatim restatements of the kept priors that survived pass
            # 2 lose to the prior copy (keyed on both the prior and its
            # line-updated copy).
            kept_keys = ({_prior_key(o) for _, o in kept_pairs}
                         | {_prior_key(k) for k, _ in kept_pairs})
            fresh_findings = [f for f in fresh_findings
                              if _prior_key(f) not in kept_keys]
        if carried:
            carried_keys = {
                (f.get("file"), f.get("line"), f.get("message")) for f in carried
            }
            fresh_findings = [
                f for f in fresh_findings
                if (f.get("file"), f.get("line"), f.get("message")) not in carried_keys
            ]
            review["carried_count"] = len(carried)
        if kept_priors:
            review["kept_prior_count"] = len(kept_priors)
        review["findings"] = fresh_findings + carried + kept_priors
        if carried or kept_priors:
            # Recompute severity across all findings
            review["severity"] = _max_severity_from_findings(
                [{"severity": review["severity"]}] + carried + kept_priors, scale)

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
            {f for f in (cached.coverage_gap_files or [])
             if f not in changed_files and f not in stripped_now}
            if is_incremental and cached is not None else set()
        )
        effective_gap_files = sorted(fresh_gap_files | carried_gap_files | set(lockfile_gaps))
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

        # A carried finding with a comment_id already has its inline
        # thread, from the pass that found it, so it is not posted again.
        # A second copy would open a second thread for the same finding on
        # every push and orphan the first one's conversation: resolution
        # and retraction match only the id the cache tracks. Nothing would
        # clear the old copy either — dismiss_previous_reviews is a no-op
        # on Bitbucket DC, and dismissing a review doesn't delete its
        # inline comments. The finding still counts toward the verdict and
        # still appears in the summary body. Matched by identity against
        # the carried dicts, so only the cache's own findings qualify.
        # The post below and the comment_id tagging after submit share
        # this one predicate, which keeps their zip aligned. A kept prior
        # finding likewise stays on its thread.
        on_thread = {
            id(f) for f in carried + kept_priors if f.get("comment_id") is not None}

        def _posts_inline(f: dict) -> bool:
            return _is_inline_postable(f) and id(f) not in on_thread

        # Submit formal review — must succeed before dismissing old reviews.
        # Verdict + inline comments are computed first; the inline-mode body
        # (below) needs to know whether anything was posted inline.
        # Reads the repo's resolved scale (fetched above), NOT
        # severity_gte(REVIEW_APPROVE_MAX_SEVERITY, ...) — that compares
        # against the built-in three-tier SEVERITY_ORDER, so every custom
        # tier name would rank 0, tie with the threshold, and approve —
        # including the repo's most severe tier. scale.blocks() (via
        # _approve_from_severity) normalizes unknown names to the most
        # severe tier instead, failing closed like _validate_review
        # (PR #211).
        # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
        approve = _approve_from_severity(review["severity"], scale)
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
        # Same fail-closed treatment for repo policy this pass couldn't read
        # or validate (an unreadable or invalid severities.json, CLAUDE.md,
        # the rules, the review override): the review still ran and still
        # posts (the author gets feedback, and the body names the source),
        # but we don't know the repo's real policy, so we must not approve
        # or auto-merge as if the built-in default applied. Unlike
        # coverage_gap this needs no persisted per-entry field — every
        # review pass (incremental or full) re-fetches the policy from
        # scratch, so a forced needs_work verdict this pass is enough: it
        # lands in CacheEntry.verdict, which already makes
        # _maybe_dispatch_cached_merge refuse a later no-op-diff retrigger
        # (entry.verdict != "approve"), and the next real push simply
        # re-fetches.
        if approve and policy_unusable:
            logger.warning(
                "PR #%d: could not read or validate repo policy (%s) — "
                "forcing needs_work verdict (fail-closed; refusing to "
                "approve or auto-merge on policy Raven never saw)",
                pr_number, ", ".join(policy_unusable))
            approve = False
        # Display only: the headline badge (SeverityScale.badge) must not
        # read "no issues" on a review whose verdict blocks, and the forced
        # needs_work above happens outside the severity it would otherwise go by.
        review["blocking"] = not approve

        inline_comments = [
            {
                "file": f["file"],
                "line": f["line"],
                "body": _format_inline_body(f, scale),
            }
            for f in review.get("findings", [])
            if _posts_inline(f)
        ] if post_inline else []

        # Build the summary body per output channel:
        #   both / summary → full review summary.
        #   inline         → NO recommendation comment. Inline-able findings
        #                    live on their lines; only findings with no
        #                    postable file/line (PR-wide notes and ⚠️
        #                    coverage-gap markers) get a MINIMAL body so they
        #                    aren't silently dropped. Empty body when there
        #                    are none — a clean PR posts just the verdict.
        #                    A carried finding with a thread isn't posted
        #                    inline, so the body lists it too: a pass whose
        #                    only blocker is carried must still name it.
        if RAVEN_REVIEW_OUTPUT == "inline":
            leftover = [f for f in review.get("findings", [])
                        if not _is_inline_postable(f)]
            on_threads = [f for f in review.get("findings", [])
                          if _is_inline_postable(f) and not _posts_inline(f)]
            body = _format_inline_leftovers(leftover, scale, review,
                                            on_threads=on_threads)
            # Never submit a content-less non-approve review: Gitea rejects an
            # empty body + no inline comments for a COMMENT / REQUEST_CHANGES
            # event. (A clean APPROVE with an empty body is fine and stays
            # intentionally silent.) Reachable only if the model returns a
            # blocking verdict with zero findings.
            if not body and not inline_comments and (
                RAVEN_REVIEW_MODE == "advisory" or not approve
            ):
                sev = review.get("severity", scale.least_severe)
                emoji, label = scale.badge(sev, review.get("findings"),
                                           blocking=not approve)
                # A clean review's headline says it all; the model's
                # sentence only repeated it. Anything else keeps it, to
                # say why (the user, 2026-10-01).
                reason = ("" if scale.no_issues(sev, review.get("findings"),
                                                blocking=not approve)
                          else f" — {review.get('summary') or 'changes requested'}")
                body = (
                    f"🦅 **Raven** — {emoji} **{label}**{reason}"
                    f"\n\n{_review_footer()}"
                )
        else:
            body = _format_comment(
                review,
                mode="advisory" if RAVEN_REVIEW_MODE == "advisory" else "review",
                scale=scale,
            )
        # Pass comment_only conditionally via dict-spread so out-of-tree
        # providers running in non-advisory modes never see the new kwarg.
        # (Default in the ABC doesn't propagate to overriders.)
        advisory_kwargs = (
            {"comment_only": True} if RAVEN_REVIEW_MODE == "advisory" else {}
        )
        # Never post APPROVE for a head that moved while this review ran:
        # the verdict describes the code this run fetched, not the new
        # commit. Park a re-run for the current head instead (it also
        # replaces any parked webhook payload for the same push). A head
        # that can't be re-read — an API error, or anything but a SHA —
        # posts NOTHING: branch protection may count a bot APPROVE, which
        # must never land on a head that may have moved, and a stand-in
        # comment would still cache an approve that a later trigger merges.
        # It parks nothing either (re-running on an API error could loop
        # paid reviews); the classified failure comment asks for a
        # re-trigger. The read itself is retried briefly first: the review
        # is already paid for, and one API blip shouldn't discard it.
        if approve and head_sha != "HEAD":
            current_head = None
            for attempt in range(_HEAD_READ_TRIES):
                if attempt:
                    time.sleep(_DIFF_HEAD_POLL_INTERVAL)
                try:
                    current_head = provider.get_pr_head_sha(repo_full_name, pr_number)
                except Exception as e:
                    logger.warning("PR #%d: could not re-check the head before "
                                   "approving: %s", pr_number, e)
                    current_head = None
                if isinstance(current_head, str) and current_head:
                    break
            if not (isinstance(current_head, str) and current_head):
                raise HeadUnverifiedError(
                    f"PR #{pr_number}: head could not be re-read before approving")
            if current_head != head_sha:
                logger.info("PR #%d: head moved %s -> %s during the review — "
                            "not posting; re-running for the new head",
                            pr_number, head_sha[:8], current_head[:8])
                inc("raven_reviews_skipped_total",
                    {"reason": "head_moved", "repo": repo_full_name})
                # Park for the CURRENT head — replacing, e.g., a stale event
                # for this run's own head.
                _park_rerun(pr_key, provider, payload, current_head)
                return
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
        posted_head = head_sha

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
        _resolve_finding_threads(provider, repo_full_name, pr_number,
                                 dropped_findings, "re-validation-dropped")
        # Superseded prior findings (a well-formed answer omitted them) and moot ones
        # (their file left the PR): resolved so no open thread is left
        # untracked. Ones already resolved are skipped.
        _resolve_finding_threads(provider, repo_full_name, pr_number,
                                 superseded + priors.moot, "superseded prior",
                                 skip=resolved_ids)
        outcomes = (
            ("kept", sum(1 for i in kept_lines if i in answered)),
            ("superseded", len(superseded)),
            ("unanswered", len(unanswered) + len(priors.overflow)),
            ("moot", len(priors.moot)),
        )
        for outcome, n in outcomes:
            if n:
                add("raven_prior_findings_total", n,
                    {"repo": repo_full_name, "outcome": outcome})

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
        # — NOT fresh_findings — with the same _posts_inline predicate as
        # the inline_comments build, so this list and posted_inline line
        # up by index. On incremental reviews that list includes carried
        # findings that had no thread yet (first seen in summary mode, or
        # their post failed); they are posted now, and their carried
        # dicts get the new ids through shared references (carried = dict
        # refs from cached_findings, the same dicts as
        # _previous_diffs[pr_key].findings). A carried finding that
        # already had a thread wasn't posted, so it keeps its comment_id.
        # Only findings actually submitted as inline comments can be tagged
        # with a comment_id. When inline output is suppressed (summary mode),
        # nothing was posted inline, so this stays empty and the length-match
        # guard below trivially passes (0 == 0) rather than warning.
        submitted_findings = [
            f for f in review.get("findings", [])
            if _posts_inline(f)
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
            # A tracked prior finding that left the live entry was retracted
            # by the comment flow mid-review: never resurrect it.
            live_ids = {id(f) for fl in live_findings.values() for f in fl}
            kept_live = [k for k, origin in kept_pairs
                         if id(origin) not in priors.tracked or id(origin) in live_ids]
            if is_incremental and cached_findings:
                findings_map = {
                    fname: [f for f in live_findings.get(fname, []) if _keep_carried(f)]
                    for fname in current_hashes if fname not in changed_files
                }
                # A stripped file's gap marker is fresh on every pass (see
                # stripped_now): keyed under its file, the fresh bucket
                # replaces the cached one (empty once the PR stops changing
                # the lockfile), and it never falls into the file-less
                # bucket, which the next pass would carry back to the model
                # as a finding.
                fresh = _findings_by_file(fresh_findings + kept_live, changed_files | stripped_now)
                # Carry forward file-less findings from previous review + new file-less ones
                fresh.setdefault("", []).extend(
                    f for f in live_findings.get("", []) if _keep_carried(f))
                findings_map.update(fresh)
            else:
                findings_map = _findings_by_file(fresh_findings + kept_live, set(current_hashes.keys()))
            _previous_diffs[pr_key] = CacheEntry(
                timestamp=time.time(),
                hashes=current_hashes,
                findings=findings_map,
                verdict=verdict,
                summary=cache_summary,
                coverage_gap_files=list(review.get("coverage_gap_files") or []),
                config_hash=entry_config_hash,
                content_hashes=current_content_hashes,
                hunks=current_hunks,
                hunk_context=current_hunk_context,
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

        # Same defense-in-depth for unreadable or invalid repo policy — the
        # verdict force above should already make this unreachable.
        if policy_unusable:
            logger.warning(
                "PR #%d approved but repo policy could not be read or "
                "validated — leaving open without auto-merge", pr_number)
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
        reason = _review_failure_reason(e)
        repo_label = repo_full_name or "unknown"
        # A classified reason (diff_truncated, timeout, …) is an expected
        # fail-closed condition with its own per-reason metric and an
        # actionable PR comment — one WARNING without a traceback, and no
        # raven_errors_total increment, so alerts on that metric only fire
        # for genuinely unexplained failures.
        if reason == "unknown":
            logger.error("Unhandled error processing PR: %s", e, exc_info=True)
            inc("raven_errors_total", {"type": "unhandled", "repo": repo_label})
        else:
            logger.warning("Review failed (reason=%s) for %s#%s: %s",
                           reason, repo_label, pr_number, e)
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
                _in_progress_heads.pop(pr_key, None)
                rerun = _rerun_requested.pop(pr_key, None)
            # A parked event for the head this run just reviewed and posted
            # (a re-requested review, a late redelivery) has nothing left to
            # do; re-running it would take the no-changes skip and dispatch
            # the merge a second time, whose failure alerts on a merged PR.
            # Re-read the head for ANY parked event: delivery order is not
            # push order (a redelivered older push can displace a newer one),
            # so the re-run targets the current head. Drop it when the
            # current head is the one this run just reviewed and posted — a
            # re-run would only take the no-changes skip and dispatch the
            # merge a second time, whose failure alerts on a merged PR. An
            # unreadable head leaves the event as parked, unless the event
            # is for the head just posted (same reason).
            if rerun is not None:
                try:
                    current_head = provider.get_pr_head_sha(repo_full_name, pr_number)
                except Exception:
                    current_head = None
                if isinstance(current_head, str) and current_head:
                    if posted_head is not None and current_head == posted_head:
                        logger.info("PR %s: the current head is the one just "
                                    "reviewed (%s) — dropping the parked event "
                                    "for %s", pr_key, posted_head[:8],
                                    (rerun[1].get("head_sha") or "?")[:8])
                        rerun = None
                    elif rerun[1].get("head_sha") != current_head:
                        rerun = (rerun[0], {**rerun[1], "head_sha": current_head})
                elif (posted_head is not None
                      and rerun[1].get("head_sha") == posted_head):
                    rerun = None
            if rerun is not None:
                logger.info("PR %s: re-running review for a push that landed "
                            "during the previous one", pr_key)
                try:
                    # _process_pr catches its own exceptions, like the webhook
                    # submits. After shutdown the pool refuses new work; the
                    # re-run is lost then, which is unavoidable, but it must
                    # not escape this finally as an unhandled error.
                    executor.submit(_process_pr, *rerun)
                except RuntimeError as e:
                    logger.warning("PR %s: not re-running (%s)", pr_key, e)


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
        #
        # The head is pinned BEFORE the diff is fetched, and checked against
        # the commit the diff is actually built from (below), so the pinned
        # SHA is the head these hashes describe. Every state change below
        # (retraction, verdict revision, merge) is bound to that one head:
        # it proceeds only while the cached review's per-file hashes equal
        # pinned_hashes. Without the binding, a push
        # whose review failed or was dropped left the cache describing an
        # older head, and an ordinary reply approved and merged code no
        # review ever saw (audit 2026-09-27 #1).
        #
        # "The diff" is only as current as the ref it is built from: on
        # Gitea that is refs/pull/N/head, which lags the head branch for a
        # moment after a push. So the pin must also be the commit the diff
        # describes, read on both sides of the fetch — otherwise a stale
        # diff of the reviewed head A would bind a comment to head B.
        # Why the head ends up unbound, when it does — one metric reason
        # per cause, so API errors and the Gitea ref lag are visible apart
        # from a genuinely unreviewed head.
        unbound_reason: str | None = None
        try:
            pinned_head = provider.get_pr_head_sha(repo_full_name, pr_number) or None
        except Exception as e:
            logger.warning("get_pr_head_sha for PR #%s failed — reply only, "
                           "no verdict or finding changes: %s", pr_number, e)
            pinned_head = None
        # The head as read, independent of whether the binding holds — the
        # code snippet shows the PR head either way.
        read_head = pinned_head
        if pinned_head is None:
            unbound_reason = "head_unknown"
        else:
            # Anything but the pinned SHA itself — a lagging ref, None, any
            # non-SHA from a provider — fails closed.
            unbound_reason = _diff_head_mismatch(
                provider, repo_full_name, pr_number, pinned_head, "diff_ref_lag")
            if unbound_reason is not None:
                logger.info("PR #%s: diff does not describe head %s (%s) — reply "
                            "only, no verdict or finding changes",
                            pr_number, pinned_head[:8], unbound_reason)
                pinned_head = None
        raw_diff = provider.fetch_pr_diff(repo_full_name, pr_number)
        if pinned_head is not None:
            # The ref matched before the fetch, so a change now is a push
            # landing mid-fetch, not ref lag.
            unbound_reason = _diff_head_mismatch(
                provider, repo_full_name, pr_number, pinned_head, "head_moved")
            if unbound_reason is not None:
                pinned_head = None
        clean_diff = strip_diff(raw_diff).clean
        pinned_hashes = _diff_chunk_hashes(raw_diff)
        diff = _truncate_diff_for_comment(clean_diff, file_path, line)
        # Fetch CLAUDE.md from the PR's BASE ref (matches _process_pr's
        # post-trust-tier behavior). CLAUDE.md is repo-policy content and
        # the reviewer renders it in the trusted ``<repo_policy_TAGID>``
        # block; using base ref means a PR can't sneak in policy-shaped
        # text that would bias its own re-review through the comment-reply
        # path. Code snippets later in this function still use head_sha
        # since they're showing the actual code under review.
        comment_base_ref_unresolved = False
        try:
            comment_base_ref = provider.get_pr_base_ref(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("get_pr_base_ref for PR #%s failed (policy read at HEAD; the reply "
                           "can't approve): %s", pr_number, e)
            comment_base_ref = "HEAD"
            comment_base_ref_unresolved = True
        # Repo severity scale — same base-ref provenance as CLAUDE.md above
        # (a scale change must land through its own review cycle, reviewed
        # under the OLD scale). Without this, a repo's own tier names are
        # unranked against the built-in low/medium/high vocabulary in the
        # two _max_severity_from_findings call sites below, silently
        # mis-rendering (and mis-notifying) severity — the comment-reply
        # flow was the last path still doing that (Task 14; see
        # docs/design-notes.md "Severity scale"). _fetch_severity_scale
        # never raises — it already falls back to default_scale()
        # internally on any failure.
        #
        # But "does not raise" is not "does not loosen the gate": that
        # fallback silently substitutes the built-in vocabulary for a
        # scale we simply failed to READ. For a repo that reuses the
        # low/medium/high names with a tighter blocks_at_or_above, the
        # default is the more permissive gate, so a flip-to-approve could
        # auto-merge past the repo's own blocking tier. _process_pr
        # already fails the merge closed on this state (see
        # scale_fetch_failed at the fresh-review call site); mirror it
        # here, which was the last merge-capable path without the guard.
        comment_scale_fetch_failed = False
        # Any other policy source this reply couldn't read or validate
        # (CLAUDE.md, a prompt override, an invalid severities.json): the
        # reply still posts, but it can't approve or merge (09-27 #12).
        comment_policy_unusable = False

        def _mark_comment_scale_fetch_failed() -> None:
            nonlocal comment_scale_fetch_failed
            comment_scale_fetch_failed = True

        def _mark_comment_policy_unusable() -> None:
            nonlocal comment_policy_unusable
            comment_policy_unusable = True

        # Policy read at HEAD isn't the policy the PR is held to.
        if comment_base_ref_unresolved:
            _mark_comment_policy_unusable()

        # Deprecated-config-path nag, as in _process_pr. The reply is the
        # only body this flow posts, so it carries the note for both the
        # scale and the respond override.
        comment_legacy_config_paths: list[str] = []

        def _note_comment_legacy_config_path(relpath: str) -> None:
            if relpath not in comment_legacy_config_paths:
                comment_legacy_config_paths.append(relpath)

        comment_scale = _fetch_severity_scale(
            provider, repo_full_name, comment_base_ref,
            on_fetch_failed=_mark_comment_scale_fetch_failed,
            on_invalid=_mark_comment_policy_unusable,
            on_legacy_path=_note_comment_legacy_config_path)
        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref=comment_base_ref)
        except Exception as e:
            # 404 (file missing) returns "" without raising; reaching this
            # except means an auth/transport failure worth flagging.
            logger.warning("CLAUDE.md fetch for PR #%s reply failed (reply proceeds without repo context, "
                           "and can't approve): %s", pr_number, e)
            _mark_comment_policy_unusable()
        # Read for the cache entry's hash (audit 09-27 #8): a merge from
        # this reply must match the policy the cached review was judged
        # under.
        comment_rules = _fetch_rules(provider, repo_full_name, comment_base_ref,
                                     on_fetch_failed=_mark_comment_policy_unusable)

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
        head_bound = (pinned_head is not None and cache_entry is not None
                      and cache_entry.hashes == pinned_hashes)
        if not head_bound and unbound_reason is None:
            unbound_reason = ("no_cache_entry" if cache_entry is None
                              else "head_not_reviewed")

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
                snippet_ref = read_head or provider.get_pr_head_sha(
                    repo_full_name, pr_number)
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
        review_override = None
        override_base_ref = comment_base_ref
        try:
            override_base_ref = (
                comment_base_ref
                if comment_base_ref != "HEAD"
                else provider.get_pr_base_ref(repo_full_name, pr_number)
            )
            respond_prompt_override = _fetch_prompt_override(
                provider, repo_full_name, override_base_ref, "respond",
                on_legacy_path=_note_comment_legacy_config_path,
                on_fetch_failed=_mark_comment_policy_unusable,
            )
            # The review override is part of the cached entry's hash, which
            # both the flip to approve and the merge are checked against.
            review_override = _fetch_prompt_override(
                provider, repo_full_name, override_base_ref, "review",
                on_fetch_failed=_mark_comment_policy_unusable)
        except Exception as e:
            # The overrides were never read: unknown, not "none".
            logger.warning("Could not resolve base ref / prompt overrides for PR #%d: %s",
                           pr_number, e)
            _mark_comment_policy_unusable()
        # The policy this reply runs under, as the cache entry records it
        # (audit 09-27 #8): an entry judged under other policy (a retarget,
        # a CLAUDE.md or rules change) can neither approve nor merge from
        # here. Computed when an approve or a merge is on the table.
        def _comment_entry_hash() -> str:
            return _entry_config_hash(
                comment_scale, review_override, base_ref=comment_base_ref,
                claude_md=claude_md, rules=comment_rules)

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
        wants_change = bool(retract_findings) or (
            revise is not None and revise.get("verdict") != prior_verdict)
        if wants_change and not head_bound:
            body += ("\n\n_No verdict or finding changes made: Raven's last "
                     "review doesn't cover the latest commit. Push a commit "
                     "or re-request review, and it will be re-evaluated._")
        legacy_lines = _legacy_config_path_lines(
            {"legacy_config_paths": comment_legacy_config_paths})
        if legacy_lines:
            body += "\n\n" + "\n".join(legacy_lines)
        provider.post_pr_comment(repo_full_name, pr_number, body,
                                 parent_comment_id=comment_id)
        inc("raven_responses_total", {"repo": repo_full_name})
        logger.info("Responded to comment on PR #%d in %s", pr_number, repo_full_name)

        # ── Retraction + verdict revision + auto-merge dispatch ────── #
        # Skip if nothing to do.
        if revise is None and not retract_findings:
            return
        if not head_bound:
            logger.info(
                "PR #%s: cached review does not cover head %s (%s) — reply "
                "only, no retraction / revision / merge", pr_number,
                (pinned_head or "<unknown>")[:8], unbound_reason)
            if wants_change:
                inc("raven_comment_mutations_skipped_total",
                    {"reason": unbound_reason, "repo": repo_full_name})
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
            # TOCTOU re-check: the entry the model reasoned about must still
            # be the live one AND still cover the pinned head. Comparing the
            # verdict string alone missed a concurrent review that landed
            # with the same verdict for a different head: the stale
            # decision was applied to it (audit 2026-09-27 #1). Identity
            # catches a replacement (_process_pr's post-submit write builds
            # a new CacheEntry); the hash comparison catches an in-place
            # rewrite of the hashes (the rebase-only path); the verdict
            # comparison catches an in-place revision by a concurrent
            # comment flow that finished its AI call first.
            with _previous_diffs_lock:
                current_entry = _previous_diffs.get(pr_key)
                still_bound = (current_entry is not None
                               and current_entry is cache_entry
                               and current_entry.hashes == pinned_hashes
                               and current_entry.verdict == prior_verdict)
            if not still_bound:
                logger.info(
                    "Cached review for %s changed during the AI call — "
                    "skipping comment-driven mutations", pr_key)
                inc("raven_comment_mutations_skipped_total",
                    {"reason": "state_changed", "repo": repo_full_name})
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
                # Drop matching cached finding so the next push-driven
                # incremental review doesn't carry it forward into its
                # verdict and summary, effectively undoing the
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
                            # Only a retraction that removed a cached
                            # FINDING counts toward the backstop and the
                            # retraction-on-approve merge trigger below.
                            # Resolving Raven's summary or a failure notice
                            # is a platform action, not a verdict change
                            # (audit 2026-09-27 #1).
                            any_retraction_succeeded = True
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

            # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
            # The server derives a comment-driven approve, as the review flow
            # does (audit 09-27 #3a): it stands only if the findings left
            # after the retractions don't block under the repo's scale, and
            # only if that scale was read. The model's revise.verdict alone
            # approved over a live blocker; and a failed read, whose fallback
            # scale can be looser than the repo's, posted and cached the
            # formal APPROVE although only that pass's merge was blocked.
            #
            # entry + remaining_findings are read once, here, for this gate,
            # the advisory body wrap (severity + findings list at body time)
            # and the dispatch site below: a single source of truth, so a
            # filter added to the findings applies to the approve gate too.
            with _previous_diffs_lock:
                entry = _previous_diffs.get(pr_key)
            remaining_findings: list[dict] = []
            if entry is not None:
                for fl in entry.findings.values():
                    remaining_findings.extend(fl)
            if revise is not None and revise.get("verdict") == "approve":
                if entry is None:
                    refusal = "no_cache_entry"
                elif comment_policy_unusable:
                    refusal = "policy_unusable"
                elif comment_scale_fetch_failed:
                    refusal = "scale_fetch_failed"
                elif entry.config_hash != _comment_entry_hash():
                    # Judged under other policy: the formal APPROVE could
                    # count for branch protection, so refuse it, not only
                    # the merge the dispatcher would decline.
                    refusal = "policy_changed"
                elif (any(comment_scale.blocks(f.get("severity")) for f in remaining_findings)
                      or not _approve_from_severity(
                          _max_severity_from_findings(remaining_findings, comment_scale),
                          comment_scale)):
                    # Each finding through blocks(), which reads a name this
                    # scale doesn't know (a finding cached under another
                    # scale) as the most severe tier; the max-severity check
                    # keeps a scale whose least tier blocks from approving.
                    refusal = "blocking_findings_remain"
                else:
                    refusal = None
                if refusal:
                    logger.warning(
                        "PR #%d: suppressing comment-driven flip-to-approve (%s)",
                        pr_number, refusal)
                    inc("raven_comment_mutations_skipped_total",
                        {"reason": refusal, "repo": repo_full_name})
                    revise = None

            # Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.
            # Verdict revision (only when verdict actually changes).
            do_revision = revise is not None and revise.get("verdict") != prior_verdict

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
                            "severity": _max_severity_from_findings(remaining_findings, comment_scale),
                            "summary": revise["body"],
                            "findings": remaining_findings,
                            "blocking": new_verdict != "approve",
                        },
                        mode="advisory_update",
                        scale=comment_scale,
                    )
                else:
                    new_body = revise["body"]

            if do_revision:
                # The revision is bound to the pinned head: re-check it is
                # still the PR head right before submitting, and submit
                # against it. A push during the AI call means the head the
                # verdict would land on is not the head the cached review
                # (and this reply) covered — skip, and let that push's own
                # review decide. Fail closed if the head can't be read.
                try:
                    current_head = provider.get_pr_head_sha(repo_full_name, pr_number)
                except Exception as e:
                    logger.warning(
                        "Could not re-check head_sha for revision on PR #%d: %s — "
                        "skipping revision (fail-closed)", pr_number, e)
                    return
                if current_head != pinned_head:
                    logger.info(
                        "PR #%d: head moved %s -> %s during the AI call — "
                        "skipping revision", pr_number, (pinned_head or "")[:8],
                        (current_head or "<none>")[:8])
                    inc("raven_comment_mutations_skipped_total",
                        {"reason": "head_moved", "repo": repo_full_name})
                    return
                rev_head_sha = pinned_head
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
                # Same binding as the TOCTOU re-check: a review that replaced
                # the entry after it (even for the same head) must not have
                # this comment's stale verdict written over it.
                with _previous_diffs_lock:
                    existing = _previous_diffs.get(pr_key)
                    if (existing is not None and existing is cache_entry
                            and existing.hashes == pinned_hashes):
                        existing.verdict = new_verdict
                        existing.summary = new_body
                        existing.timestamp = time.time()  # mark recently active for LRU
                        mutated_cache = True
                    else:
                        logger.info("Cache entry for %s changed or was evicted "
                                    "before the revision write — not recorded", pr_key)
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
            if should_dispatch_merge:
                # One gate set for every merge that reuses a cached verdict:
                # _maybe_dispatch_cached_merge re-checks advisory mode, a
                # failed severities.json read (the verdict above was then
                # computed under a fallback scale that may be looser than
                # the repo's real one; the revision still posts, only the
                # merge is declined), the entry and its verdict, the
                # coverage gap, that the cached hashes describe the pinned
                # head's diff, the per-entry
                # review-config hash, PR state and sole reviewer — then
                # dispatches the same _safe_do_merge (CI wait + head
                # recheck). The comment path used to carry its own partial
                # copy of these gates, without the hash or config checks
                # (audit 2026-09-27 #1).
                try:
                    meta = provider.get_pr_metadata(repo_full_name, pr_number)
                except Exception as e:
                    logger.debug("get_pr_metadata for PR #%d failed: %s", pr_number, e)
                    meta = {}
                _maybe_dispatch_cached_merge(
                    provider, repo_full_name, pr_number,
                    meta.get("title") or f"PR #{pr_number}",
                    meta.get("html_url") or "",
                    head_sha=pinned_head, current_hashes=pinned_hashes,
                    expected_config_hash=_comment_entry_hash(),
                    scale=comment_scale, source="comment",
                    scale_fetch_failed=comment_scale_fetch_failed,
                    policy_unusable=comment_policy_unusable)
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

    except ThreadResolvedError as e:
        # Someone resolved the thread while Raven was answering: they closed
        # the conversation, so the answer is dropped, as is the failure
        # reply, which the same resolved thread would refuse. Nothing after
        # the reply ran, so no state changed.
        logger.warning("Reply skipped for %s#%s: %s", repo_full_name, pr_number, e)
        inc("raven_comment_replies_skipped_total",
            {"reason": "thread_resolved", "repo": repo_full_name or "unknown"})

    except Exception as e:
        reason = _review_failure_reason(e)
        repo_label = repo_full_name or "unknown"
        # Same ERROR-vs-WARNING split as _process_pr: classified reasons
        # are expected conditions and must not pollute raven_errors_total
        # or read as crashes in the logs.
        if reason == "unknown":
            logger.error("Failed to respond to comment: %s", e, exc_info=True)
            inc("raven_errors_total", {"type": "comment_response_failed",
                                       "repo": repo_label})
        else:
            logger.warning("Comment response failed (reason=%s) for %s#%s: %s",
                           reason, repo_label, pr_number, e)
        # Same classified failure metric as the review flow so timeout /
        # usage-cap / auth spikes on comment replies show on the same
        # dashboard. The reply UX keeps its own threaded message below
        # (a verdict-style review comment would be wrong here). respond_to_comment
        # already retried the transient classes inside reviewer.py.
        inc("raven_review_failures_total",
            {"reason": reason, "repo": repo_label})
        if repo_full_name and pr_number:
            # A diff Raven refused (truncated, unverifiable or not bound to
            # one head) gets the same actionable notice as the review flow
            # (07-02 #6). Anything else keeps the generic text: an
            # unclassified exception's message must never reach the PR.
            reply = (_failure_comment(reason)
                     if reason in ("diff_truncated", "diff_unverifiable",
                                   "diff_identity_unverified", "diff_head_unverified")
                     else "\U0001f985 \u26a0\ufe0f Couldn't respond — internal error while processing your comment.")
            try:
                provider.post_pr_comment(
                    repo_full_name, pr_number, reply,
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

def _diff_head_mismatch(provider: GitProvider, repo_full_name: str,
                        pr_number: int, pinned_head: str,
                        mismatch_reason: str) -> str | None:
    """Why the PR diff can't be bound to ``pinned_head``, or None when it can.

    ``mismatch_reason`` names a real SHA that isn't the pin; a read that
    fails or returns no SHA at all is ``head_unknown`` / ``diff_head_unknown``
    instead, so an API or provider fault never shows up as ref lag or a push.
    """
    try:
        diff_head = provider.get_pr_diff_head_sha(repo_full_name, pr_number)
    except Exception as e:
        logger.warning("get_pr_diff_head_sha for PR #%s failed: %s", pr_number, e)
        return "head_unknown"
    if not isinstance(diff_head, str) or not diff_head:
        return "diff_head_unknown"
    return None if diff_head == pinned_head else mismatch_reason


def _moved_head(provider: GitProvider, repo_full_name: str, pr_number: int,
                head_sha: str) -> str | None:
    """The PR head, when it is a real SHA other than ``head_sha``; else None."""
    try:
        current = provider.get_pr_head_sha(repo_full_name, pr_number)
    except Exception as e:
        logger.debug("get_pr_head_sha for PR #%s failed: %s", pr_number, e)
        return None
    if isinstance(current, str) and current and current != head_sha:
        return current
    return None


def _await_diff_head(provider: GitProvider, repo_full_name: str,
                     pr_number: int, head_sha: str) -> tuple[str | None, str | None]:
    """Wait until the PR diff describes ``head_sha``.

    Returns ``(None, None)`` once it does, and ``("head_moved", new_head)``
    as soon as the PR head itself has moved on. Otherwise it keeps polling
    — a read fault is retried like a lagging ref — for up to
    ``_DIFF_HEAD_POLLS`` reads and about as many seconds, then returns the
    last reason (``diff_ref_lag`` / ``head_unknown`` / ``diff_head_unknown``,
    see ``_diff_head_mismatch``) with None.
    """
    deadline = time.monotonic() + _DIFF_HEAD_POLLS * _DIFF_HEAD_POLL_INTERVAL
    reason: str | None = None
    for attempt in range(_DIFF_HEAD_POLLS):
        reason = _diff_head_mismatch(provider, repo_full_name, pr_number,
                                     head_sha, "diff_ref_lag")
        if reason is None:
            return None, None
        moved = _moved_head(provider, repo_full_name, pr_number, head_sha)
        if moved is not None:
            return "head_moved", moved
        if attempt == _DIFF_HEAD_POLLS - 1 or time.monotonic() >= deadline:
            break
        time.sleep(_DIFF_HEAD_POLL_INTERVAL)
    return reason, None


def _park_rerun(pr_key: str, provider: GitProvider, payload: dict, head: str) -> None:
    """Park a re-run of ``payload`` for ``head``, started when the review
    in progress ends. A webhook already parked for that same head is kept
    (it carries its own title/base_ref); one for any other head is replaced."""
    with _in_progress_lock:
        parked = _rerun_requested.get(pr_key)
        if parked is None or parked[1].get("head_sha") != head:
            _rerun_requested[pr_key] = (provider, {**payload, "head_sha": head})


def _lockfile_gaps(diff: str) -> list[str]:
    """Every lockfile this PR changes, deleted ones included (D2 (b), as
    amended 2026-09-28). Lockfiles are stripped from the diff the model
    reviews, so a swapped package source or an added package would merge
    unseen; a human merges instead. A deletion counts too: the next plain
    install re-resolves the whole tree within the manifest's ranges, drops
    the pins, and can resolve a package the lockfile pinned to a private
    host from the public registry. So does a rename off a lockfile name
    (``git mv package-lock.json package-lock.json.png``): it removes the
    live lockfile just the same. Read from every section of the whole diff
    by both its names, not from the stripped paths: a rename from a
    non-lockfile name isn't stripped (the model must see the source
    leave), yet it makes a live lockfile. Each gap is keyed by the
    section's new name, the key the diff has."""
    renamed_off_a_lockfile = {target for source, target in _rename_aliases(diff).items()
                              if _is_lockfile_name(source)}
    paths: list[str] = []
    for path, _chunk in split_diff_by_file(diff):
        if ((_is_lockfile_name(path) or _normalize_path(path) in renamed_off_a_lockfile)
                and path not in paths):
            paths.append(path)
    return paths


def _diff_chunk_hashes(diff: str) -> dict[str, str]:
    """Per-file SHA256 of the raw diff chunks: the "is this literally the
    same diff?" identity ``CacheEntry.hashes`` records. One definition
    shared by every path that compares against it, so the push flow, the
    cached-merge gate and the comment flow can't drift.

    Verdict-logic trigger: changing this bumps reviewer._VERDICT_LOGIC_VERSION.

    Callers pass the UNSTRIPPED diff: the identity covers every file the
    PR changes, lockfiles and skip-listed binaries included. Over the
    stripped diff, a push that changed only a stripped file hashed the
    same as the approved head and merged from the cache with no review
    (audit 09-27 #4)."""
    return {f: hashlib.sha256(c.encode()).hexdigest()
            for f, c in split_diff_by_file(diff)}


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
    lines = _diff_lines(file_content)
    if not lines or line > len(lines):
        return ""
    start = max(1, line - context)
    end = min(len(lines), line + context)
    width = len(str(end))
    formatted: list[str] = []
    for n in range(start, end + 1):
        marker = "→" if n == line else " "
        # Numbered as git numbers lines (\n only); a CRLF file's \r is
        # dropped from the display.
        formatted.append(f"{n:>{width}} {marker} {lines[n - 1].rstrip(chr(13))}")
    return "\n".join(formatted)


def _head_truncate(diff: str) -> str:
    """Plain head-truncation fallback used when no relevance bias applies."""
    total = diff.count("\n")
    if total <= MAX_DIFF_LINES:
        return diff
    lines = _diff_lines(diff, keepends=True)
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
    lines = _diff_lines(chunk, keepends=True)
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
    lines = _diff_lines(chunk, keepends=True)
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
    # A source file git diffed as binary is a coverage gap (strip_diff)
    # whose content can't be shown meaningfully, and one with few
    # newlines would pass the line cap whole: don't fetch it. Nor a file
    # Bitbucket cut lines in: it is a gap too, and its contents would add
    # the same over-long lines to the prompt a second time.
    stripped = strip_diff(clean_diff)
    unfetched = set(stripped.binary_gaps) | set(stripped.cut_gaps)
    file_chunks = [(f, c) for f, c in split_diff_by_file(clean_diff) if f not in unfetched]
    file_contents: dict[str, str] = {}
    omitted: list[str] = []
    for filename, _ in file_chunks[:MAX_FILES]:
        try:
            content = provider.fetch_file(repo_full_name, filename, ref=head_sha)
        except IncompleteFileError:
            omitted.append(f"{filename} (the platform couldn't return it whole)")
            continue
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
#
# These are the repo's OWN rules, shared with every agent working in it —
# Raven reading them is the point. Contrast CONFIG_DIR below, which holds
# config that is Raven's business alone.
RULES_DIR = os.environ.get("RAVEN_RULES_DIR", ".claude/rules")

# Directory holding Raven's per-repo configuration: prompt overrides
# (``prompts/{review,respond}.md``) and the severity scale
# (``severities.json``). Override with RAVEN_CONFIG_DIR (empty string
# disables per-repo Raven config entirely).
#
# Deliberately OUTSIDE ``.claude/``, where all three lived until
# 2026-08-17. Agents working in a repo sweep ``.claude/`` into their
# context wholesale, so a Raven prompt override — thousands of tokens of
# instructions addressed to a code-review bot, relevant to no other
# tool — was being loaded by every one of them. Raven's own rule listing
# is flat (``providers/*.list_directory``), so it never re-read that
# subtree itself: the leak was entirely outbound, which is exactly why
# nothing here caught it.
CONFIG_DIR = os.environ.get("RAVEN_CONFIG_DIR", ".raven")


def _legacy_config_path(relpath: str) -> str | None:
    """The pre-2026-08-17 home of ``{CONFIG_DIR}/{relpath}``, under
    ``{RULES_DIR}/raven/``.

    ``None`` when ``RULES_DIR`` is disabled — there is no legacy path to
    read, and no legacy path to name in the migration note.
    """
    if not RULES_DIR:
        return None
    return f"{RULES_DIR}/raven/{relpath}"


class _RepoConfigFile(NamedTuple):
    """Outcome of resolving one per-repo config file across both homes.

    ``content`` is ``""`` when the file is absent from both. ``legacy``
    marks a hit on the deprecated path (drives the migration note).
    ``fetch_failed`` marks that at least one attempt RAISED — distinct
    from "absent", and the distinction is load-bearing for the severity
    scale's fail-closed merge gate.
    """
    content: str
    legacy: bool
    fetch_failed: bool


def _fetch_repo_config_file(provider: GitProvider, repo_full_name: str,
                            ref: str, relpath: str,
                            on_legacy_path: Callable[[str], None] | None = None,
                            ) -> _RepoConfigFile:
    """Read ``{CONFIG_DIR}/{relpath}`` at ``ref``, falling back to the
    legacy ``{RULES_DIR}/raven/{relpath}``.

    The new path wins whenever it holds content, and the legacy path is
    not even probed in that case — so a repo mid-migration that left the
    old file behind gets the new one, and pays no extra API call.

    ``on_legacy_path`` fires with ``relpath`` (not the full path — the
    renderer reconstructs both ends from the current dirs) only when the
    content actually came from the deprecated location. Callers use it to
    nag in-band; see ``_legacy_config_path_lines``.

    A whitespace-only file counts as absent on BOTH paths: blanking the
    new file is how someone disables an override, and having that
    silently resurrect the legacy one would be a nasty surprise.

    Never raises — a config file that can't be read must not stop the
    review. ``fetch_failed`` reports the failure to callers that need to
    act on it.
    """
    fetch_failed = False
    candidates: list[tuple[str, bool]] = []
    if CONFIG_DIR:
        candidates.append((f"{CONFIG_DIR}/{relpath}", False))
    legacy_path = _legacy_config_path(relpath)
    if legacy_path:
        candidates.append((legacy_path, True))

    for path, is_legacy in candidates:
        try:
            content = provider.fetch_file(repo_full_name, path, ref=ref)
        except Exception as e:
            # Both providers return "" for a 404 (the common "no such
            # file" case), so reaching this means a real auth/transport
            # failure. Keep trying the other path — one location being
            # unreachable says nothing about the other.
            logger.warning("Could not read %s at %s for %s: %s",
                           path, ref[:8] if ref else "", repo_full_name, e)
            fetch_failed = True
            continue
        if content and content.strip():
            if is_legacy:
                logger.warning(
                    "%s read Raven config from the deprecated path %s — "
                    "move it to %s/%s (the old path still works, but "
                    "everything under %s is loaded into every other "
                    "agent's context in this repo)",
                    repo_full_name, path, CONFIG_DIR, relpath, RULES_DIR)
                if on_legacy_path is not None:
                    on_legacy_path(relpath)
            return _RepoConfigFile(content, is_legacy, fetch_failed)

    return _RepoConfigFile("", False, fetch_failed)


def _fetch_rules(provider: GitProvider, repo_full_name: str, ref: str,
                 on_fetch_failed: Callable[[], None] | None = None) -> dict[str, str]:
    """Read ``*.md`` files from ``RULES_DIR`` at ``ref``, return
    ``{path: contents}`` sorted by path.

    Best-effort: a missing directory, listing error, or individual
    fetch failure returns an empty/partial map rather than raising —
    the review must proceed regardless. A listing error or a rule file
    that can't be read fires ``on_fetch_failed``, so the caller can keep
    the review from approving on rules it never saw (audit 09-27 #12); a
    missing directory or file doesn't.
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
        if on_fetch_failed is not None:
            on_fetch_failed()
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
            if on_fetch_failed is not None:
                on_fetch_failed()
    if rules:
        logger.info("Loaded %d rule file(s) from %s for %s", len(rules), RULES_DIR, repo_full_name)
    return rules


def _fetch_prompt_override(provider: GitProvider, repo_full_name: str,
                            ref: str, name: str,
                            on_legacy_path: Callable[[str], None] | None = None,
                            on_fetch_failed: Callable[[], None] | None = None,
                            ) -> str | None:
    """Fetch a per-repo prompt override from ``{CONFIG_DIR}/prompts/{name}.md``,
    falling back to the legacy ``{RULES_DIR}/raven/prompts/{name}.md``.

    Returns the file contents on success, ``None`` on any failure mode
    (missing file, fetch error, empty / whitespace-only content, or both
    config dirs disabled). Callers must treat ``None`` as "no override,
    use the built-in default".

    ``name`` is ``"review"`` or ``"respond"`` — not validated, but only
    those two are currently wired into the reviewer.

    ``on_legacy_path`` — see ``_fetch_repo_config_file``. Optional and
    keyword-friendly so every existing call site is unaffected.

    ``on_fetch_failed`` fires when either path couldn't be read: ``None``
    then means "unknown", not "no override", and a caller deciding an
    approve or a merge must not treat the two alike (audit 09-27 #12).
    """
    relpath = f"prompts/{name}.md"
    result = _fetch_repo_config_file(provider, repo_full_name, ref, relpath,
                                     on_legacy_path=on_legacy_path)
    if result.fetch_failed and on_fetch_failed is not None:
        on_fetch_failed()
    if not result.content:
        return None
    logger.info("Loaded %s prompt override for %s", name, repo_full_name)
    return result.content


def _fetch_severity_scale(provider: GitProvider, repo_full_name: str,
                          ref: str,
                          on_fetch_failed: Callable[[], None] | None = None,
                          on_legacy_path: Callable[[str], None] | None = None,
                          on_invalid: Callable[[], None] | None = None,
                          ) -> SeverityScale:
    """Load ``{CONFIG_DIR}/severities.json`` at ``ref``, falling back to
    the legacy ``{RULES_DIR}/raven/severities.json``.

    Always returns a usable scale. A missing file, a fetch failure, or an
    invalid file falls back to the built-in default — a defined
    conservative gate beats an undefined one, and one repo's typo must not
    stop that repo being reviewed.

    ``ref`` is the PR's BASE ref, like rules and prompt overrides: a
    change to the scale must land through its own review cycle, reviewed
    under the OLD scale, so a hostile PR cannot widen its own merge gate.

    ``on_fetch_failed`` fires ONLY for the "could not read the file"
    branch — never for "file absent" (the normal case for most repos) or
    "file present but invalid" (already counted separately via
    ``raven_severity_scale_invalid_total``). The distinction matters: a
    fetch failure means the repo may well *have* a stricter scale than the
    built-in default and we simply failed to read it, so the caller can
    fail the MERGE GATE closed (refuse auto-merge for this pass) without
    refusing to review — unlike a missing file, which is silently and
    correctly the default scale. Keyword-only and optional so every
    existing caller (and every test that calls or mocks this function
    positionally / by return value) is unaffected. ``_process_pr``'s
    review path and no-changes skip and ``_process_comment`` all wire it
    up. See CLAUDE.md "Severity scale rules".

    ``on_legacy_path`` — see ``_fetch_repo_config_file``.

    ``on_invalid`` fires for a file that is present but invalid. The
    review still runs under the default scale, but the repo meant a
    different one, so the caller keeps it from approving or merging
    (audit 09-27 #12).

    A read failure on EITHER path fires ``on_fetch_failed``, even when the
    other path yielded a usable scale: an unreadable ``{CONFIG_DIR}``
    file may be a scale stricter than whatever we managed to fall back
    to, and "we could not see the current config" is the condition the
    merge gate exists to refuse on.
    """
    result = _fetch_repo_config_file(provider, repo_full_name, ref,
                                     "severities.json",
                                     on_legacy_path=on_legacy_path)
    if result.fetch_failed:
        inc("raven_severity_scale_fetch_failed_total", {"repo": repo_full_name})
        if on_fetch_failed is not None:
            on_fetch_failed()

    if not result.content:
        return default_scale()

    path = (_legacy_config_path("severities.json") if result.legacy
            else f"{CONFIG_DIR}/severities.json")
    try:
        scale = from_json(result.content)
    except InvalidScale as e:
        logger.warning("Invalid %s in %s — using the default severity scale: %s",
                       path, repo_full_name, e)
        inc("raven_severity_scale_invalid_total", {"repo": repo_full_name})
        if on_invalid is not None:
            on_invalid()
        return default_scale()

    logger.info("Using repo severity scale for %s: %s (blocks at %s)",
                repo_full_name, ", ".join(scale.ordered()),
                scale.blocks_at_or_above or "nothing")
    return scale


def _notify_if_needed(repo_full_name: str, pr_number: int, pr_title: str, pr_url: str, review: dict) -> None:
    """Send notification — severity filtering is handled per-channel in notifier."""
    notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
           link=pr_url, action="needs_review")


def _max_severity_from_findings(findings: list[dict],
                                scale: SeverityScale | None = None) -> str:
    """Highest severity name among findings; the scale's least severe tier
    when the list is empty.

    Each name goes through ``scale.rank()``, which reads a name the scale
    doesn't know (or a missing one) as its MOST severe tier, as
    ``scale.normalize()`` does for model-emitted severities. That was the
    deferred "Phase B decision": an unrecognised severity used to rank
    lowest, so a finding cached under another scale could let a review
    approve (audit 09-27 #8). Names are stripped and lowercased before
    lookup, as ``reviewer._validate_review`` does (#211).
    """
    scale = scale or default_scale()
    if not findings:
        return scale.least_severe
    least_rank = scale.ranks[scale.least_severe]
    best = max((scale.rank(f.get("severity", "")) for f in findings), default=least_rank)
    for name, rank in scale.ranks.items():
        if rank == best:
            return name
    return scale.least_severe


def _policy_unusable_lines(review: dict) -> list[str]:
    """Why a review that would otherwise approve can't: repo policy this
    pass couldn't read or validate (audit 09-27 #12). The labels are
    static (set by ``_process_pr``), never an exception's text."""
    labels = review.get("policy_unusable") or []
    if not labels or not isinstance(labels, list):
        return []
    return [f"⚠️ **Repository policy unavailable:** Raven couldn't read or "
            f"validate {', '.join(labels)} at the base branch, so this review "
            f"can't approve or auto-merge. Re-trigger the review once that's fixed."]


def _legacy_config_path_lines(review: dict) -> list[str]:
    """Migration nag for a review that read Raven config from the
    deprecated ``{RULES_DIR}/raven/`` home instead of ``{CONFIG_DIR}/``.

    Rendered on EVERY affected review rather than counted on a dashboard:
    the person who can move the file is the one reading the PR, and the
    nag stops by itself the moment they do.

    Reconstructs both ends of each move from the relative path the fetch
    reported, so the note always names the paths this instance actually
    resolves. Returns ``[]`` when nothing legacy was read (the common
    case), when the field is absent (cached / pre-move review dicts), or
    when ``RULES_DIR`` is disabled — with no legacy dir configured there
    is no old path to name, and a stale field must not render a note
    pointing at nothing.
    """
    relpaths = review.get("legacy_config_paths") or []
    if not relpaths or not isinstance(relpaths, list):
        return []
    moves = []
    for rp in relpaths:
        rp = str(rp)
        # Defence in depth. This field is set only by _process_pr from
        # Raven's own literals, and _validate_review rebuilds the review
        # dict from a whitelist so a model-emitted key can't reach here —
        # but the dict is otherwise model-shaped, and these paths render
        # inside code spans in a PR comment. A backtick or newline would
        # break out of the span; drop rather than escape.
        if "`" in rp or "\n" in rp:
            continue
        old = _legacy_config_path(rp)
        if old:
            moves.append(f"`{old}` → `{CONFIG_DIR}/{rp}`")
    if not moves:
        return []
    return [
        "> ⚠️ **Deprecated Raven config path.** This review read config "
        f"from Raven's pre-`{CONFIG_DIR}` location: {'; '.join(moves)}. "
        "The old path still works but will stop being read in a future "
        f"release. Raven's per-repo config moved out of `{RULES_DIR}/` "
        "because agents working in this repo load that directory into "
        "their context wholesale, and Raven's config is no use to any of "
        "them."
    ]


def _severity_mismatch_lines(review: dict) -> list[str]:
    """Lines reporting a vocabulary mismatch between the model's emitted
    severities and the scale actually in effect — the empirical detection
    signal for the whole feature (see raven/severity.py's ``normalize()``
    and docs/archive/specs/2026-08-03-configurable-severity-scale-design.md).
    Returns ``[]`` when there's no mismatch (the common case) or the review
    predates this field.

    Shared by ``_format_comment`` and ``_format_inline_leftovers`` so
    ``RAVEN_REVIEW_OUTPUT=inline`` — which never calls ``_format_comment``
    — doesn't silently drop this line. Each caller is responsible for its
    own leading blank-line spacing before the returned lines.

    Names the severities.json path ONLY when a repo scale is
    actually governing this review — and names the path the scale was
    really read from (``legacy_config_paths`` says whether that was the
    deprecated home), because this line's whole job is to send the
    operator to a file to go edit. ``unknown_severities`` fires just as
    often for the ~100% of repos with no severities.json at all (built-in
    low/medium/high in effect) — pointing at a file that was never read,
    and likely doesn't exist, sent the operator chasing the wrong cause;
    there the real one is usually a prompt override emitting non-standard
    names, or the model simply not honouring the vocabulary. Detected by
    comparing the review's tier names against ``default_scale().ordered()``
    — stable regardless of ``REVIEW_APPROVE_MAX_SEVERITY`` (that env var
    only shifts ``blocks_at_or_above``, never the three tier names) — not
    a new field: every review dict a producer builds already carries
    ``severity_scale_names``.
    """
    unknown_severities = review.get("unknown_severities") or []
    if not unknown_severities:
        return []
    names = review.get("severity_scale_names") or []
    known = ", ".join(f"`{n}`" for n in names)
    offending = ", ".join(f"`{n}`" for n in unknown_severities)
    if names == default_scale().ordered():
        scale_ref = "the active severity scale (this repo's built-in default)"
        fix_hint = (
            "This repo has no custom severity scale configured, so the "
            "likely cause is a prompt override emitting non-standard "
            "severity names, or the model not honouring the vocabulary."
        )
    else:
        legacy = "severities.json" in (review.get("legacy_config_paths") or [])
        scale_path = (_legacy_config_path("severities.json") if legacy
                      else f"{CONFIG_DIR}/severities.json")
        scale_ref = f"`{scale_path}`"
        fix_hint = "Fix the scale or the prompt override so they use the same names."
    return [
        f"> ⚠️ **Severity config mismatch.** This review emitted severity "
        f"names not in {scale_ref}: {offending}. "
        f"Known tiers: {known}. Unrecognised findings were treated as the "
        f"most severe tier and the merge was blocked. {fix_hint}"
    ]


def _review_footer() -> str:
    """The provenance line that closes every review body Raven composes:
    the full summary, and inline mode's short body whenever one is posted."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"*Reviewed by Raven v{__version__} · {RAVEN_AI_MODEL} · "
            f"effort {RAVEN_AI_EFFORT} · {timestamp}*")


def _format_comment(review: dict, mode: str = "review",
                    scale: SeverityScale | None = None) -> str:
    """Render the review summary body.

    ``mode`` selects the header / subtitle:
      * ``"review"``           — formal review (header: "Raven Review").
      * ``"advisory"``         — initial advisory recommendation.
      * ``"advisory_update"``  — advisory recommendation revised via comment thread.

    ``scale`` should be the repo's resolved scale (``_fetch_severity_scale``)
    whenever the caller has one — omitting it silently falls back to
    ``default_scale()``, under which every custom tier name is unknown and
    ``emoji()`` fails closed to most-severe for ALL of them: a real bug
    found in review, where a nit/bug/blocker repo rendered every finding
    🔴 regardless of tier, defeating Phase A's position-based colour work
    entirely. ``scale=None`` stays the default so any call site that
    hasn't been threaded a scale yet keeps behaving exactly as before.
    """
    scale = scale or default_scale()
    severity = review.get("severity", scale.least_severe)
    summary = review.get("summary", "")
    findings = review.get("findings", [])
    emoji, label = scale.badge(severity, review.get("findings"),
                               blocking=bool(review.get("blocking")))

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
    lines.append(f"**{emoji} {label}** — {summary}")

    # Empirical vocabulary-mismatch signal — an override that contradicts
    # the scale is not prevented (impossible against opaque override
    # text), it fails closed and is reported here. Missing key must be
    # safe (cached/legacy review dicts predate this field).
    mismatch_lines = _severity_mismatch_lines(review)
    if mismatch_lines:
        lines.append("")
        lines.extend(mismatch_lines)

    legacy_lines = _legacy_config_path_lines(review)
    if legacy_lines:
        lines.append("")
        lines.extend(legacy_lines)

    policy_lines = _policy_unusable_lines(review)
    if policy_lines:
        lines.append("")
        lines.extend(policy_lines)

    if findings:
        lines.append("")
        lines.append("**Findings:**")
        for f in findings:
            f_sev = f.get("severity", scale.least_severe)
            f_emoji = scale.emoji(f_sev)
            lines.append(f"- {f_emoji} [{f_sev}] {f.get('message', '')}")

    if review.get("chunked"):
        n = review.get("chunks_reviewed", "?")
        lines.append("")
        lines.append(f"*⚡ Large diff — reviewed {n} files separately*")

    if review.get("carried_count"):
        n = review["carried_count"]
        lines.append("")
        lines.append(f"*Includes {n} finding(s) carried from unchanged files*")

    if review.get("kept_prior_count"):
        n = review["kept_prior_count"]
        lines.append("")
        lines.append(f"*Includes {n} earlier finding(s) that still apply, kept on their threads*")

    lines.append("")
    lines.append(_review_footer())

    return "\n".join(lines)


def _format_inline_leftovers(findings: list[dict],
                             scale: SeverityScale | None = None,
                             review: dict | None = None,
                             on_threads: list[dict] | None = None) -> str:
    """Minimal body for ``RAVEN_REVIEW_OUTPUT=inline``.

    Inline mode posts no summary/recommendation comment. The only findings
    that can't ride on a diff line are those with no postable file/line —
    PR-wide notes and ⚠️ coverage-gap markers (filename, no line). Those are
    listed in a short body so they aren't silently dropped. Returns ``""``
    when every finding is inline-anchored AND there's neither a
    severity-mismatch note nor a deprecated-config-path note (see below),
    so a clean review posts no body. A body that is posted ends with the
    same footer as the full summary (``_review_footer``); a footer alone
    would be a comment on every clean review, so an empty body stays empty.

    ``on_threads`` — carried findings that already have an inline thread
    from an earlier review, so this review doesn't post them. They are
    listed in their own section, or a pass whose only blocker is carried
    would request changes without naming it.

    ``scale`` should be the repo's resolved scale — see ``_format_comment``
    for why omitting it is unsafe for a custom vocabulary (every tier
    renders 🔴). Defaults to ``default_scale()`` for callers not yet
    threaded through.

    ``review`` — when supplied — surfaces the same "Severity config
    mismatch" and "Deprecated Raven config path" notes ``_format_comment``
    renders (``_severity_mismatch_lines``, ``_legacy_config_path_lines``).
    Inline mode never calls ``_format_comment``, so without this the
    feature's whole detection story (which names were unrecognised, what
    the scale actually allows) was invisible under
    ``RAVEN_REVIEW_OUTPUT=inline`` — the merge still failed closed, but
    silently. Rendered even when ``findings`` is empty, since a mismatch
    can occur with zero non-postable findings.
    """
    scale = scale or default_scale()
    mismatch_lines = _severity_mismatch_lines(review) if review else []
    legacy_lines = _legacy_config_path_lines(review) if review else []
    policy_lines = _policy_unusable_lines(review) if review else []
    if (not findings and not on_threads and not mismatch_lines and not legacy_lines
            and not policy_lines):
        return ""
    lines = ["🦅 **Raven**"]
    for heading, group in (("Findings without an inline location:", findings),
                           ("Still open from earlier reviews, on their existing threads:",
                            on_threads)):
        if not group:
            continue
        lines.append("")
        lines.append(heading)
        lines.append("")
        for f in group:
            f_sev = f.get("severity", scale.least_severe)
            f_emoji = scale.emoji(f_sev)
            lines.append(f"- {f_emoji} **[{f_sev}]** {f.get('message', '')}")
    if mismatch_lines:
        lines.append("")
        lines.extend(mismatch_lines)
    if legacy_lines:
        lines.append("")
        lines.extend(legacy_lines)
    if policy_lines:
        lines.append("")
        lines.extend(policy_lines)
    lines.append("")
    lines.append(_review_footer())
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
