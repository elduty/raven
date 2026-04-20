"""server.py — Flask app with webhook endpoints for git platform providers."""

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request

from .providers import GitProvider, get_provider, register_provider, registered_providers
from .providers.gitea import GiteaProvider
from .metrics import inc, Timer, format_prometheus
from .notifier import notify
from .reviewer import review_diff, respond_to_comment, severity_gte, SEVERITY_ORDER, review_config_hash, _strip_lockfiles_and_binaries, split_diff_by_file, MAX_DIFF_LINES

_SEVERITY_NAME = {v: k for k, v in SEVERITY_ORDER.items()}

logger = logging.getLogger(__name__)

_GITEA_AUTO_MERGE = os.environ.get("RAVEN_GITEA_AUTO_MERGE", "").lower() in ("1", "true", "yes")

executor = ThreadPoolExecutor(max_workers=int(os.environ.get("RAVEN_MAX_WORKERS", "16")))

# ── PR dedup: prevent concurrent reviews for the same PR ──────────── #
_recent_prs: dict[str, float] = {}
_recent_prs_lock = threading.Lock()
DEDUP_WINDOW = 30  # seconds

# ── Comment response cooldown per PR ─────────────────────────────── #
_comment_cooldowns: dict[str, float] = {}
_comment_cooldowns_lock = threading.Lock()
COMMENT_COOLDOWN = 30  # seconds between responses on the same PR

def _check_comment_cooldown(repo: str, pr_number: int) -> bool:
    """Return True if a comment response is on cooldown for this PR."""
    key = f"{repo}#comment#{pr_number}"
    now = time.time()
    with _comment_cooldowns_lock:
        if key in _comment_cooldowns and now - _comment_cooldowns[key] < COMMENT_COOLDOWN:
            return True
        _comment_cooldowns[key] = now
        # Prune stale entries
        stale = [k for k, t in _comment_cooldowns.items() if now - t > COMMENT_COOLDOWN * 2]
        for k in stale:
            del _comment_cooldowns[k]
    return False

# ── Previous diff cache for incremental reviews ──────────────────── #
_previous_diffs: dict[str, tuple[float, dict[str, str], dict[str, list[dict]]]] = {}
# key -> (timestamp, {filename: diff_hash}, {filename: [findings]})
_previous_diffs_lock = threading.Lock()
_MAX_CACHED_PRS = int(os.environ.get("RAVEN_MAX_CACHED_PRS", "200"))
_CACHE_DIR = Path(os.environ.get("RAVEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "raven")))
_CACHE_FILE = _CACHE_DIR / "findings_cache.json"


def _load_cache() -> None:
    """Load findings cache from disk on startup. Wipes cache if config changed."""
    global _previous_diffs
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
        with _previous_diffs_lock:
            for key, entry in entries.items():
                if isinstance(entry, list) and len(entry) >= 3:
                    _previous_diffs[key] = (entry[0], entry[1], entry[2])
        logger.info("Loaded %d cached PR reviews from %s", len(_previous_diffs), _CACHE_FILE)
    except Exception as e:
        logger.warning("Could not load findings cache from %s: %s", _CACHE_FILE, e)


def _save_cache() -> None:
    """Persist findings cache to disk (atomic write)."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _previous_diffs_lock:
            data = {
                "_config_hash": review_config_hash(),
                "entries": {k: list(v) for k, v in _previous_diffs.items()},
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


def _evict_cache() -> None:
    """Evict oldest entries if cache exceeds _MAX_CACHED_PRS."""
    with _previous_diffs_lock:
        if len(_previous_diffs) <= _MAX_CACHED_PRS:
            return
        sorted_keys = sorted(_previous_diffs, key=lambda k: _previous_diffs[k][0])
        to_remove = len(_previous_diffs) - _MAX_CACHED_PRS
        for key in sorted_keys[:to_remove]:
            del _previous_diffs[key]


def _should_skip_duplicate(repo: str, pr_number: int) -> bool:
    """Return True if this PR was already dispatched within DEDUP_WINDOW."""
    key = f"{repo}#{pr_number}"
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

# ------------------------------------------------------------------ #
#  App factory                                                         #
# ------------------------------------------------------------------ #

def create_app() -> Flask:
    # Register providers based on available env vars
    gitea_url = os.environ.get("GITEA_URL")
    gitea_token = os.environ.get("GITEA_TOKEN")
    gitea_secret = os.environ.get("GITEA_WEBHOOK_SECRET") or os.environ.get("RAVEN_WEBHOOK_SECRET")
    if os.environ.get("RAVEN_WEBHOOK_SECRET") and not os.environ.get("GITEA_WEBHOOK_SECRET"):
        logger.warning("RAVEN_WEBHOOK_SECRET is deprecated — use GITEA_WEBHOOK_SECRET")
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    _load_cache()

    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.route("/metrics")
    def metrics():
        return format_prometheus(), 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.route("/hook", methods=["POST"])
    def hook_legacy():
        """Backward compat — route to first registered provider."""
        first = next(iter(registered_providers()))
        return hook(first)

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
            if _should_skip_duplicate(f"{provider.name}:{repo}", pr_number):
                logger.info("Push to PR branch %s — skipping duplicate for PR #%s", branch, pr_number)
                return jsonify({"status": "skipped", "reason": "duplicate"})
            logger.info("Push to PR branch %s — triggering re-review for PR #%s", branch, pr_number)
            # Enrich payload with PR details from the lookup
            payload["pr_number"] = pr_number
            payload["pr_title"] = pr.get("title", f"PR #{pr_number}")
            payload["pr_url"] = pr.get("html_url", "")
            payload["head_sha"] = pr.get("head", {}).get("sha", "HEAD")
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
            if _should_skip_duplicate(f"{provider.name}:{repo}", pr_number):
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
            if _should_skip_duplicate(f"{provider.name}:{repo}", f"review-{pr_number}"):
                return jsonify({"status": "skipped", "reason": "duplicate"})
            executor.submit(_process_pr, provider, payload)
            return jsonify({"status": "accepted", "reason": "review requested"})

        elif event_type in ("review_approved", "review_rejected"):
            if event_type == "review_rejected":
                return jsonify({"status": "ignored", "reason": "review rejected — no action"})
            pr_number = payload.get("pr_number")
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            if _is_bot_author(sender):
                return jsonify({"status": "skipped"})
            if _is_skipped_repo(repo):
                return jsonify({"status": "skipped"})
            if _should_skip_duplicate(f"{provider.name}:{repo}", f"review-approved-{pr_number}"):
                return jsonify({"status": "skipped", "reason": "duplicate"})
            executor.submit(_process_review_approved, provider, payload)
            return jsonify({"status": "accepted", "reason": "checking merge eligibility"})

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
            # Mention check
            is_mention = raven_user and bool(
                re.search(rf"@{re.escape(raven_user)}\b", comment_body, re.IGNORECASE)
            )
            if not is_mention:
                return jsonify({"status": "ignored", "reason": "not directed at Raven"})
            if not pr_number:
                return jsonify({"status": "skipped", "reason": "missing PR number"})
            if comment_id and _should_skip_duplicate(f"{provider.name}:{repo}", f"comment-{comment_id}"):
                return jsonify({"status": "skipped", "reason": "duplicate"})
            if _check_comment_cooldown(repo, pr_number):
                return jsonify({"status": "skipped", "reason": "comment cooldown"})
            executor.submit(_process_comment, provider, payload)
            return jsonify({"status": "accepted", "reason": "responding to comment"})

        else:
            logger.debug("Ignoring unsupported event: %s", event_type)
            return jsonify({"status": "ignored", "event": event_type})

    return app


# ------------------------------------------------------------------ #
#  Background PR review                                               #
# ------------------------------------------------------------------ #

def _process_pr(provider: GitProvider, payload: dict) -> None:
    """Review a PR in a background thread. All exceptions caught here."""
    repo_full_name = None
    pr_number = None
    try:
        repo_full_name = payload["repo"]
        pr_number = payload["pr_number"]
        pr_title = payload.get("pr_title") or f"PR #{pr_number}"
        pr_url = payload.get("pr_url", "")
        head_sha = payload.get("head_sha") or "HEAD"

        # Make the pending review visible immediately by adding Raven as a reviewer
        # before fetching the diff. Best-effort — failures do not block the review.
        try:
            provider.add_self_as_reviewer(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("Failed to add Raven as reviewer on PR #%d: %s", pr_number, e)

        # Fetch diff
        diff = provider.fetch_pr_diff(repo_full_name, pr_number)

        # Fetch repo context
        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref=head_sha)
        except Exception:
            pass

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
                previous_hashes = cached[1]
                cached_findings = cached[2] if len(cached) > 2 else {}
            else:
                previous_hashes = {}
                cached_findings = {}

        changed_files = {f for f, h in current_hashes.items() if previous_hashes.get(f) != h}
        removed_files = previous_hashes.keys() - current_hashes.keys()
        if previous_hashes and not changed_files and not removed_files:
            logger.info("PR #%d re-review: no files changed since last review — skipping", pr_number)
            inc("raven_reviews_skipped_total", {"reason": "no_changes", "repo": repo_full_name})
            return

        is_incremental = False
        if previous_hashes and removed_files:
            # Files were removed — do a full review to clear stale findings
            logger.info("PR #%d files removed since last review — full re-review", pr_number)
            review_diff_text = clean_diff
        elif previous_hashes and changed_files:
            # Incremental: rebuild diff from only changed file chunks
            is_incremental = True
            review_diff_text = "".join(file_chunks[f] for f in sorted(changed_files))
            logger.info("PR #%d incremental review: %d/%d files changed", pr_number, len(changed_files), len(current_hashes))
        else:
            review_diff_text = clean_diff

        # Fetch full file contents for all PR files (cross-file context even on incremental)
        file_contents = _fetch_changed_files(provider, repo_full_name, head_sha, clean_diff)

        # Run review
        with Timer("raven_review_duration_seconds", {"repo": repo_full_name}):
            review = review_diff(review_diff_text, repo_full_name, claude_md=claude_md, file_contents=file_contents)
        # Save original findings before merging carried ones (used for cache write)
        fresh_findings = list(review.get("findings", []))
        # Merge carried findings from unchanged files into the review
        if is_incremental and cached_findings:
            carried = []
            for fname in current_hashes:
                if fname not in changed_files:
                    carried.extend(cached_findings.get(fname, []))
            # Also carry forward file-less findings
            carried.extend(cached_findings.get("", []))
            if carried:
                review["findings"] = review.get("findings", []) + carried
                review["carried_count"] = len(carried)
                # Recompute severity across all findings
                max_sev = SEVERITY_ORDER.get(review["severity"], 0)
                for finding in carried:
                    max_sev = max(max_sev, SEVERITY_ORDER.get(finding.get("severity", "low"), 0))
                review["severity"] = _SEVERITY_NAME.get(max_sev, "low")

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

        # Submit formal review — must succeed before dismissing old reviews
        body = _format_comment(review)
        approve_sev = os.environ.get("REVIEW_APPROVE_MAX_SEVERITY", "low")
        approve = severity_gte(approve_sev, review["severity"])
        default_emoji = "\U0001f7e2"
        inline_comments = [
            {
                "file": f["file"],
                "line": f["line"],
                "body": f"{SEVERITY_EMOJI.get(f.get('severity', 'low'), default_emoji)} **[{f.get('severity', 'low')}]** {f['message']}",
            }
            for f in review.get("findings", [])
            if f.get("file") and isinstance(f.get("line"), int) and f["line"] > 0
        ]
        try:
            new_review = provider.submit_review(repo_full_name, pr_number, body,
                                                approve=approve, inline_comments=inline_comments,
                                                commit_id=head_sha)
        except Exception as e:
            logger.error("Failed to submit review on PR #%d: %s", pr_number, e)
            notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
                   link=pr_url, action="review_submit_failed")
            inc("raven_errors_total", {"type": "review_submit_failed", "repo": repo_full_name})
            return

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

        # Cache diff + per-file findings for incremental re-reviews
        # Use fresh_findings (pre-merge) to avoid duplicating carried findings
        if is_incremental and cached_findings:
            findings_map = {fname: cached_findings.get(fname, []) for fname in current_hashes if fname not in changed_files}
            fresh = _findings_by_file(fresh_findings, changed_files)
            # Carry forward file-less findings from previous review + new file-less ones
            fresh.setdefault("", []).extend(cached_findings.get("", []))
            findings_map.update(fresh)
        else:
            findings_map = _findings_by_file(fresh_findings, set(current_hashes.keys()))
        with _previous_diffs_lock:
            _previous_diffs[pr_key] = (time.time(), current_hashes, findings_map)
        _evict_cache()
        _save_cache()

        # Add label
        try:
            provider.add_label_to_pr(repo_full_name, pr_number)
        except Exception as e:
            logger.warning("Failed to add label: %s", e)

        if not approve:
            logger.info("PR #%d review=REQUEST_CHANGES — leaving open", pr_number)
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        raven_user = provider.get_authenticated_user()
        other_reviews = [r for r in provider.get_pr_reviews(repo_full_name, pr_number)
                         if r.get("user", {}).get("login") != raven_user]
        requested = provider.get_pr_requested_reviewers(repo_full_name, pr_number)
        if other_reviews or requested:
            logger.info("PR #%d has other reviewers — leaving open for human review", pr_number)
            _notify_if_needed(repo_full_name, pr_number, pr_title, pr_url, review)
            return

        # Raven is the only reviewer — merge
        merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
        _do_merge(provider, repo_full_name, pr_number, pr_title, pr_url, review, head_sha, merge_strategy)

    except Exception as e:
        logger.error("Unhandled error processing PR: %s", e, exc_info=True)
        inc("raven_errors_total", {"type": "unhandled", "repo": repo_full_name or "unknown"})
        # Clear dedup entry so webhook retries can re-attempt this PR
        if repo_full_name is not None and pr_number is not None:
            key = f"{provider.name}:{repo_full_name}#{pr_number}"
            with _recent_prs_lock:
                _recent_prs.pop(key, None)
            try:
                provider.post_pr_comment(repo_full_name, pr_number,
                    "🦅 **Raven Review**\n\n⚠️ Internal error — review could not be completed.")
            except Exception:
                pass


# ------------------------------------------------------------------ #
#  Comment response                                                    #
# ------------------------------------------------------------------ #

def _process_comment(provider: GitProvider, payload: dict) -> None:
    """Respond to a comment directed at Raven in a background thread."""
    try:
        repo_full_name = payload["repo"]
        pr_number = payload.get("pr_number")
        comment_body = payload.get("comment_body", "")
        file_path = payload.get("file_path", "") or ""
        line = payload.get("line") or 0

        # Fetch context (truncate diff to avoid token bloat on large PRs)
        raw_diff = provider.fetch_pr_diff(repo_full_name, pr_number)
        diff = _strip_lockfiles_and_binaries(raw_diff)
        diff_lines = diff.splitlines(keepends=True)
        if len(diff_lines) > MAX_DIFF_LINES:
            diff = "".join(diff_lines[:MAX_DIFF_LINES]) + f"\n... (truncated, {len(diff_lines) - MAX_DIFF_LINES} lines omitted)"
        claude_md = ""
        try:
            claude_md = provider.fetch_file(repo_full_name, "CLAUDE.md", ref="HEAD")
        except Exception:
            pass

        # Fetch conversation (keep last 20 to avoid prompt bloat)
        conversation = provider.get_pr_comments(repo_full_name, pr_number)[-20:]

        # Generate response
        response = respond_to_comment(
            comment_body, conversation, diff, repo_full_name,
            claude_md=claude_md, file_path=file_path, line=line,
        )

        # Post response (skip if Claude returned nothing)
        if not response:
            logger.warning("Empty response from Claude for comment on PR #%d — not posting", pr_number)
            return
        if file_path:
            location = f"`{file_path}`"
            if line:
                location += f" line {line}"
            body = f"\U0001f985 **Re: {location}**\n\n{response}"
        else:
            body = f"\U0001f985 {response}"
        provider.post_pr_comment(repo_full_name, pr_number, body)
        inc("raven_responses_total", {"repo": repo_full_name})
        logger.info("Responded to comment on PR #%d in %s", pr_number, repo_full_name)

    except Exception as e:
        logger.error("Failed to respond to comment: %s", e, exc_info=True)


# ------------------------------------------------------------------ #
#  Merge orchestration                                                 #
# ------------------------------------------------------------------ #

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

_AUTO_MERGE_ON_APPROVAL = os.environ.get("RAVEN_AUTO_MERGE_ON_APPROVAL", "").lower() in ("1", "true", "yes")


def _latest_review_per_user(reviews: list[dict]) -> dict[str, str]:
    """Resolve review list to {login: latest_state}. Later entries win."""
    latest: dict[str, str] = {}
    for r in reviews:
        login = r.get("user", {}).get("login", "")
        state = r.get("state", "")
        if login and state in ("APPROVED", "REQUEST_CHANGES"):
            latest[login] = state
    return latest


def _process_review_approved(provider: GitProvider, payload: dict) -> None:
    """Check if a human approval means we can now auto-merge a Raven-approved PR."""
    try:
        if not _AUTO_MERGE_ON_APPROVAL:
            logger.debug("RAVEN_AUTO_MERGE_ON_APPROVAL not enabled — ignoring review_approved event")
            return

        repo_full_name = payload["repo"]
        pr_number = payload["pr_number"]
        pr_title = payload.get("pr_title") or f"PR #{pr_number}"
        pr_url = payload.get("pr_url", "")

        raven_user = provider.get_authenticated_user()
        reviews = provider.get_pr_reviews(repo_full_name, pr_number)

        # Resolve to latest review state per user (handles superseded reviews)
        latest = _latest_review_per_user(reviews)

        if latest.get(raven_user) != "APPROVED":
            logger.info("PR #%d: Raven's latest review is not APPROVED — skipping merge", pr_number)
            return

        # Check no reviewer has outstanding REQUEST_CHANGES
        has_rejections = any(
            state == "REQUEST_CHANGES" for login, state in latest.items() if login != raven_user
        )
        if has_rejections:
            logger.info("PR #%d: outstanding REQUEST_CHANGES from another reviewer — skipping merge", pr_number)
            return

        # Check no outstanding requested reviewers remain
        requested = provider.get_pr_requested_reviewers(repo_full_name, pr_number)
        if requested:
            logger.info("PR #%d: still has requested reviewers — skipping merge", pr_number)
            return

        current_sha = provider.get_pr_head_sha(repo_full_name, pr_number)
        merge_strategy = os.environ.get("MERGE_STRATEGY", "squash")
        # Construct a minimal review dict for notifications
        review = {"severity": "low", "summary": "Previously approved by Raven", "findings": []}

        _do_merge(provider, repo_full_name, pr_number, pr_title, pr_url, review,
                  current_sha, merge_strategy)

    except Exception as e:
        logger.error("Failed to process review_approved: %s", e, exc_info=True)


# ------------------------------------------------------------------ #
#  CI status polling                                                   #
# ------------------------------------------------------------------ #

def _wait_for_ci(provider: GitProvider, repo_full_name: str, sha: str, timeout: int = 300) -> str:
    """Poll commit status until CI finishes or timeout. Returns final status.

    Returns 'success', 'failure', 'error', 'pending', or 'none'.
    'none' means no CI is configured for this repo.

    Waits 10s before the first check to give CI time to register 'pending'.
    Without this, Raven can merge before CI even starts.
    """
    initial_delay = 10
    interval = 15
    elapsed = 0

    # Give CI time to pick up the job and set status to 'pending'
    time.sleep(initial_delay)
    elapsed += initial_delay

    while elapsed < timeout:
        status = provider.get_commit_status(repo_full_name, sha)
        if status in ("success", "failure", "error", "none"):
            return status
        # Still pending — wait and retry
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


MAX_FILE_LINES = 500  # Skip files larger than this (generated/minified)
MAX_FILES = 10  # Limit total files to avoid token bloat


def _fetch_changed_files(provider: GitProvider, repo_full_name: str, head_sha: str, clean_diff: str) -> dict[str, str]:
    """Fetch full contents of changed files for review context."""
    file_chunks = split_diff_by_file(clean_diff)
    file_contents: dict[str, str] = {}
    for filename, _ in file_chunks[:MAX_FILES]:
        try:
            content = provider.fetch_file(repo_full_name, filename, ref=head_sha)
            if content and content.count("\n") <= MAX_FILE_LINES:
                file_contents[filename] = content
        except Exception as e:
            logger.debug("Could not fetch %s for context: %s", filename, e)
    return file_contents


def _notify_if_needed(repo_full_name: str, pr_number: int, pr_title: str, pr_url: str, review: dict) -> None:
    """Send notification — severity filtering is handled per-channel in notifier."""
    notify(repo_full_name, f"PR #{pr_number}: {pr_title}", review,
           link=pr_url, action="needs_review")


SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def _format_comment(review: dict) -> str:
    severity = review.get("severity", "low")
    summary = review.get("summary", "")
    findings = review.get("findings", [])
    emoji = SEVERITY_EMOJI.get(severity, "🟢")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "🦅 **Raven Review**",
        "",
        f"**{emoji} {severity.upper()}** — {summary}",
    ]

    if findings:
        lines.append("")
        lines.append("**Findings:**")
        for f in findings:
            f_sev = f.get("severity", "low")
            f_emoji = SEVERITY_EMOJI.get(f_sev, "🟢")
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
    lines.append(f"*Reviewed by Raven · {timestamp}*")

    return "\n".join(lines)


def _is_skipped_repo(repo_full_name: str) -> bool:
    skip_list = os.environ.get("SKIP_REPOS", "")
    if not skip_list:
        return False
    skipped = {r.strip() for r in skip_list.split(",") if r.strip()}
    return repo_full_name in skipped


def _is_bot_author(*names: str) -> bool:
    skip_authors_raw = os.environ.get("SKIP_AUTHORS", "")
    default_bots = {"bot", "github-actions", "dependabot", "renovate", "gitea-actions"}
    skipped = default_bots | {a.strip().lower() for a in skip_authors_raw.split(",") if a.strip()}
    for name in names:
        if not name:
            continue
        n = name.lower()
        if n in skipped or n.endswith("[bot]") or "bot" in n.split("-"):
            return True
    return False
