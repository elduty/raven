"""gitea.py — GiteaProvider: Gitea API client implementing GitProvider ABC."""

import base64
import hashlib
import hmac
import logging
import os
from urllib.parse import quote

import requests
from flask import abort

from raven.providers import GitProvider

logger = logging.getLogger(__name__)

RAVEN_LABEL_NAME = os.environ.get("RAVEN_LABEL_NAME", "raven-reviewed")



class GiteaProvider(GitProvider):
    """Gitea API client implementing the GitProvider interface."""

    name = "gitea"

    def __init__(self, base_url=None, token=None, webhook_secret: str = ""):
        self.base_url = (base_url or os.environ["GITEA_URL"]).rstrip("/")
        self.token = token or os.environ["GITEA_TOKEN"]
        if not webhook_secret:
            raise ValueError("webhook_secret is required — empty secret allows forged webhooks")
        self.webhook_secret = webhook_secret
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.token}",
            "Content-Type": "application/json",
        })
        self._username = None
        self._can_dismiss: bool | None = None  # None = unknown, True/False after first attempt

    def get_authenticated_user(self) -> str:
        """Return the login name of the authenticated user (cached)."""
        if self._username is None:
            resp = self.session.get(f"{self.base_url}/api/v1/user", timeout=10)
            resp.raise_for_status()
            login = resp.json().get("login", "")
            if not login:
                raise RuntimeError("Gitea API returned empty login for authenticated user")
            self._username = login
        return self._username

    # ------------------------------------------------------------------ #
    #  Diff fetching                                                       #
    # ------------------------------------------------------------------ #

    def find_open_pr_for_branch(self, repo_full_name: str, branch: str) -> dict | None:
        """Find an open PR whose head branch matches the given branch name."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls"
        max_pages = 20
        for page in range(1, max_pages + 1):
            resp = self.session.get(
                url,
                params={"state": "open", "limit": 50, "page": page},
                timeout=15,
            )
            resp.raise_for_status()
            prs = resp.json()
            if not prs:
                return None
            for pr in prs:
                if pr.get("head", {}).get("ref", "") == branch:
                    return pr
        logger.warning("Reached max pages (%d) searching for PR on branch %s", max_pages, branch)
        return None

    def get_pr_head_sha(self, repo_full_name: str, pr_number: int) -> str:
        """Return the current head SHA of a PR. Raises if missing."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        sha = resp.json().get("head", {}).get("sha", "")
        if not sha:
            raise RuntimeError(f"Gitea returned empty head SHA for PR #{pr_number}")
        return sha

    def get_pr_base_ref(self, repo_full_name: str, pr_number: int) -> str:
        """Return the PR's base branch name. Raises if missing."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        ref = resp.json().get("base", {}).get("ref", "")
        if not ref:
            raise RuntimeError(f"Gitea returned empty base ref for PR #{pr_number}")
        return ref

    def fetch_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Return the raw unified diff for a pull request."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}.diff"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        if not resp.text.strip():
            raise RuntimeError(f"Gitea returned empty diff for PR #{pr_number}")
        return resp.text

    def get_pr_description(self, repo_full_name: str, pr_number: int) -> str:
        """Return the PR body (description) as plain text, or "" on any failure."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("body", "") or ""
        except Exception as e:
            logger.warning("Failed to fetch PR #%d description: %s", pr_number, e)
            return ""

    # ------------------------------------------------------------------ #
    #  Comment posting                                                     #
    # ------------------------------------------------------------------ #

    # ── New methods for the comment-thread-context feature ───────────── #

    def _fetch_review_comments(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return all PR review comments (paginated). Internal helper.

        Both the reviews list and each review's comments list are paginated
        — long-running PRs are where thread context matters most, so we
        iterate all pages (capped at max_pages=10) rather than silently
        dropping older reviews.
        """
        owner, repo = _split_repo(repo_full_name)
        reviews_url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        max_pages = 10
        reviews: list[dict] = []
        for page in range(1, max_pages + 1):
            try:
                resp = self.session.get(reviews_url, params={"limit": 50, "page": page}, timeout=10)
                resp.raise_for_status()
            except requests.HTTPError:
                return reviews
            batch = resp.json()
            if not batch:
                break
            reviews.extend(batch)
            if len(batch) < 50:
                break
        all_comments: list[dict] = []
        for review in reviews:
            if not review.get("comments_count"):
                continue
            rid = review["id"]
            comments_url = (
                f"{self.base_url}/api/v1/repos/{owner}/{repo}"
                f"/pulls/{pr_number}/reviews/{rid}/comments"
            )
            for page in range(1, max_pages + 1):
                try:
                    r = self.session.get(comments_url, params={"limit": 50, "page": page}, timeout=10)
                    r.raise_for_status()
                except requests.HTTPError:
                    break
                batch = r.json()
                if not batch:
                    break
                all_comments.extend(batch)
                if len(batch) < 50:
                    break
        return all_comments

    def get_comment_thread(self, repo_full_name: str, pr_number: int,
                           root_comment_id: int) -> list[dict]:
        """Return the conversation thread containing root_comment_id,
        chronologically ordered.

        Gitea has no in_reply_to_id on review comments. What the UI calls
        a "conversation" is the group of comments sharing the same
        (path, position) across the PR's reviews. So we fetch all review
        comments, find the root, and return every comment with the same
        anchor in created_at order.

        Outdated comments (force-push/rebase moved or deleted the line)
        have position=null. Falls back to original_position to keep the
        anchor stable; if the root has no usable position at all (both
        null), return only the root so we don't over-group every
        outdated comment at the same path into one synthetic thread.
        """
        all_comments = self._fetch_review_comments(repo_full_name, pr_number)
        by_id = {c["id"]: c for c in all_comments}
        root = by_id.get(root_comment_id)
        if not root:
            return []

        def _anchor(c: dict) -> tuple:
            # Prefer current position; fall back to original_position for
            # outdated comments (force-push moved/deleted the line).
            pos = c.get("position") or c.get("original_position")
            return (c.get("path"), pos)

        def _render(c: dict) -> dict:
            pos = c.get("position") or c.get("original_position")
            return {
                "id": c["id"],
                "parent_id": None,  # Gitea has no reply tree
                "user": {"login": (c.get("user") or {}).get("login", "")},
                "body": c.get("body", ""),
                "file_path": c.get("path"),
                "line": pos,
                "resolved": c.get("resolver") is not None,
            }

        root_anchor = _anchor(root)
        if root_anchor[1] is None:
            # No usable position on either field — don't over-group every
            # null-anchored comment on the same file into one synthetic
            # thread. Return just the root.
            return [_render(root)]

        thread_comments = [
            c for c in all_comments if _anchor(c) == root_anchor
        ]
        thread_comments.sort(key=lambda c: c.get("created_at", ""))
        return [_render(c) for c in thread_comments]

    def get_pr_state(self, repo_full_name: str, pr_number: int) -> str:
        """Map Gitea PR state to canonical 'open' / 'merged' / 'closed'."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("merged"):
            return "merged"
        return "open" if data.get("state") == "open" else "closed"

    def get_pr_metadata(self, repo_full_name: str, pr_number: int) -> dict:
        """Return {'title', 'html_url'}. Best-effort; returns {} on
        fetch failure so callers can degrade to defaults."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            logger.warning("Gitea get_pr_metadata for PR #%d failed: %s", pr_number, e)
            return {}
        return {"title": data.get("title", ""), "html_url": data.get("html_url", "")}

    def retract_finding(self, repo_full_name: str, pr_number: int,
                        comment_id: int) -> bool:
        """Resolve a previously-posted inline review comment.

        Uses Gitea's native ``POST /pulls/comments/{id}/resolve`` endpoint
        (added in Gitea 1.24). The comment body is preserved; only the
        resolved state changes. On older Gitea the endpoint returns 404
        and we log + return False; the dev can manually resolve via UI.
        403/404 do not raise.
        """
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/comments/{comment_id}/resolve"
        try:
            resp = self.session.post(url, timeout=10)
            if resp.status_code in (403, 404):
                logger.warning("Gitea retract comment %s failed: HTTP %s",
                               comment_id, resp.status_code)
                return False
            resp.raise_for_status()
            return True
        except requests.HTTPError as e:
            logger.warning("Gitea retract comment %s failed: %s", comment_id, e)
            return False

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return all comments on a PR (paginated)."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/issues/{pr_number}/comments"
        all_comments: list[dict] = []
        max_pages = 10
        for page in range(1, max_pages + 1):
            resp = self.session.get(url, params={"limit": 50, "page": page}, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_comments.extend(batch)
        return all_comments

    def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str,
                        parent_comment_id: int | None = None) -> dict:
        """Post a comment on a PR.

        Gitea issue comments are flat — ``parent_comment_id`` is accepted for
        interface compatibility but ignored.
        """
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/issues/{pr_number}/comments"
        resp = self.session.post(url, json={"body": body}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def react_to_comment(self, repo_full_name: str, pr_number: int,
                         comment_id: int, content: str = "eyes") -> None:
        """Post an emoji reaction on a PR issue comment.

        Only the issue-comments reactions endpoint is used; diff/review
        comments live under a different URL and return 404 here, which is
        swallowed. Never raises — this is fire-and-forget UX.
        """
        owner, repo = _split_repo(repo_full_name)
        url = (
            f"{self.base_url}/api/v1/repos/{owner}/{repo}"
            f"/issues/comments/{comment_id}/reactions"
        )
        try:
            resp = self.session.post(url, json={"content": content}, timeout=5)
            if resp.status_code not in (200, 201, 404):
                logger.debug("Reaction on comment %s returned %s",
                             comment_id, resp.status_code)
        except Exception as e:
            logger.debug("Reaction on comment %s failed: %s", comment_id, e)

    def submit_review(self, repo_full_name: str, pr_number: int, body: str,
                      approve: bool, inline_comments: list[dict] | None = None,
                      commit_id: str = "", comment_only: bool = False) -> dict:
        """Submit a formal PR review with optional inline comments.

        approve: True -> APPROVED, False -> REQUEST_CHANGES.
        inline_comments: normalized [{"file": "...", "line": N, "body": "..."}] dicts.
        commit_id: pin review to a specific commit SHA (prevents stale reviews on force-push).
        comment_only: when True, posts the review with event=COMMENT — no
            APPROVED / REQUEST_CHANGES verdict, but inline comments still
            attach. Used for advisory mode where the verdict is
            non-blocking.

        Returns the Gitea review dict, extended with an ``inline_comments``
        key carrying per-input-entry ``{file, line, comment_id}`` — same
        length as input, with comment_id=None for any entry that was
        filtered (missing/invalid file/line) or whose returned ID we
        couldn't match. Used by the comment-thread-context retraction
        cache cleanup.
        """
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        if comment_only:
            event = "COMMENT"
        else:
            event = "APPROVED" if approve else "REQUEST_CHANGES"
        payload: dict = {"body": body, "event": event}
        if commit_id:
            payload["commit_id"] = commit_id
        normalized = self._normalize_inline_comments(inline_comments) if inline_comments else []
        if normalized:
            payload["comments"] = normalized
        resp = self.session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        # Capture per-inline-comment IDs by following up with a GET on
        # the new review. Match by (path, new_position, body). Body is
        # unique per finding (severity emoji + message). Filtered entries
        # (invalid file/line) get comment_id=None.
        posted_inline: list[dict] = []
        if inline_comments:
            lookup: dict[tuple, int] = {}
            review_id = result.get("id")
            if review_id and normalized:
                comments_url = (
                    f"{self.base_url}/api/v1/repos/{owner}/{repo}"
                    f"/pulls/{pr_number}/reviews/{review_id}/comments"
                )
                # Paginate — Gitea defaults to 50/page on this endpoint
                # too; reviews with >50 inline findings would otherwise
                # lose IDs beyond page 1 and break retraction matching.
                max_pages = 10
                try:
                    for page in range(1, max_pages + 1):
                        cresp = self.session.get(
                            comments_url, params={"limit": 50, "page": page},
                            timeout=10,
                        )
                        cresp.raise_for_status()
                        batch = cresp.json()
                        if not batch:
                            break
                        for rc in batch:
                            key = (
                                rc.get("path"),
                                rc.get("position") or rc.get("new_position"),
                                rc.get("body"),
                            )
                            lookup[key] = rc.get("id")
                        if len(batch) < 50:
                            break
                except requests.HTTPError as e:
                    logger.warning("Could not fetch inline comment IDs for review %s: %s",
                                   review_id, e)
            # Anchor-only fallback: when exact body match misses (e.g.
            # Gitea server-side normalises whitespace), fall back to
            # (path, position) IF exactly one returned comment sits at
            # that anchor. Per-hit logged at DEBUG so a review with N
            # findings doesn't emit N WARNING lines; a single summary
            # WARNING fires below when fallback_count > 0.
            anchor_lookup: dict[tuple, list[int]] = {}
            for (rpath, rpos, _rbody), rid in lookup.items():
                anchor_lookup.setdefault((rpath, rpos), []).append(rid)
            unmatched = 0
            fallback_count = 0
            for c in inline_comments:
                file_path = c.get("file", "")
                line = c.get("line", 0)
                body_text = c.get("body", "")
                cid = None
                if file_path and isinstance(line, int) and line > 0:
                    cid = lookup.get((file_path, line, body_text))
                    if cid is None:
                        ids = anchor_lookup.get((file_path, line), [])
                        if len(ids) == 1:
                            cid = ids[0]
                            fallback_count += 1
                            logger.debug(
                                "Inline comment ID matched by (path, line) fallback for %s:%d",
                                file_path, line,
                            )
                        elif normalized:
                            unmatched += 1
                posted_inline.append({
                    "file": file_path, "line": line, "comment_id": cid,
                })
            if fallback_count:
                logger.warning(
                    "%d/%d inline comments matched by (path, line) fallback on review %s — "
                    "Gitea may have normalised the bodies",
                    fallback_count, len(normalized), review_id,
                )
            if unmatched:
                logger.warning(
                    "Could not match %d/%d inline comments to returned IDs on review %s — "
                    "retraction-by-id will be unavailable for those entries",
                    unmatched, len(normalized), review_id,
                )

        result["inline_comments"] = posted_inline
        return result

    @staticmethod
    def _normalize_inline_comments(comments: list[dict]) -> list[dict]:
        """Convert normalized inline comments to Gitea format with severity emoji."""
        gitea_comments = []
        for c in comments:
            file_path = c.get("file", "")
            line = c.get("line", 0)
            body = c.get("body", "")
            if file_path and isinstance(line, int) and line > 0:
                gitea_comments.append({
                    "path": file_path,
                    "new_position": line,
                    "body": body,
                })
        return gitea_comments

    def dismiss_review(self, repo_full_name: str, pr_number: int, review_id: int) -> None:
        """Dismiss a single PR review. Skips silently if lacking admin permissions."""
        if self._can_dismiss is False:
            return

        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/dismissals"
        resp = self.session.post(url, json={"message": "Superseded by new review"}, timeout=10)

        if resp.status_code in (200, 201):
            self._can_dismiss = True
            logger.info("Dismissed review %d on PR #%d", review_id, pr_number)
        elif resp.status_code == 403:
            self._can_dismiss = False
            logger.info("Review dismissal requires admin — disabling for this session")
        else:
            logger.warning("Failed to dismiss review %d on PR #%d: %s", review_id, pr_number, resp.status_code)

    def dismiss_previous_reviews(self, repo_full_name: str, pr_number: int, bot_user: str,
                                  exclude_id: int | None = None) -> None:
        """Dismiss all previous reviews by bot_user, excluding exclude_id."""
        for r in self.get_pr_reviews(repo_full_name, pr_number):
            rid = r.get("id")
            if rid == exclude_id:
                continue
            if r.get("user", {}).get("login") == bot_user and r.get("state") in ("REQUEST_CHANGES", "APPROVED"):
                self.dismiss_review(repo_full_name, pr_number, rid)

    def get_pr_reviews(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return all reviews on a PR (paginated), normalised.

        Passes through ``id`` (used by ``dismiss_previous_reviews``).
        Normalises review state to common form: "APPROVED", "REQUEST_CHANGES", "COMMENT".
        """
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        all_reviews: list[dict] = []
        max_pages = 10
        for page in range(1, max_pages + 1):
            resp = self.session.get(url, params={"limit": 50, "page": page}, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            for r in batch:
                all_reviews.append({
                    "id": r.get("id"),
                    "user": {"login": (r.get("user") or {}).get("login", "")},
                    "state": r.get("state", ""),
                })
        return all_reviews

    def get_pr_requested_reviewers(self, repo_full_name: str, pr_number: int) -> list[str]:
        """Return login names of users requested to review this PR.

        Reads the ``requested_reviewers`` field from the PR object
        itself. The dedicated ``/pulls/{n}/requested_reviewers``
        endpoint returns 404 on some Gitea versions (self-hosted 1.22
        has been observed), and silently returning ``[]`` there would
        make auto-add and the reviewer-status gate misjudge the PR's
        state. The PR object's field is populated reliably across
        versions.
        """
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
        users = data.get("requested_reviewers") or []
        return [u.get("login", "") for u in users if isinstance(u, dict)]

    def add_self_as_reviewer(self, repo_full_name: str, pr_number: int) -> None:
        """Request the authenticated bot user as a reviewer. Idempotent."""
        owner, repo = _split_repo(repo_full_name)
        username = self.get_authenticated_user()
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/requested_reviewers"
        resp = self.session.post(url, json={"reviewers": [username]}, timeout=10)
        # 201 = created, 200 = already requested (Gitea treats as no-op)
        if resp.status_code not in (200, 201):
            resp.raise_for_status()

    # ------------------------------------------------------------------ #
    #  PR operations                                                       #
    # ------------------------------------------------------------------ #

    def merge_pr(self, repo_full_name: str, pr_number: int, commit_title: str = "",
                 strategy: str = "squash", head_sha: str = "",
                 merge_when_checks_succeed: bool = False) -> bool:
        """Merge a PR. Strategy: 'squash', 'merge', 'rebase', 'fast-forward-only'. Returns True on success."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/pulls/{pr_number}/merge"
        payload: dict = {
            "Do": strategy,
            "merge_title_field": commit_title or f"Merge PR #{pr_number}",
            "delete_branch_after_merge": True,
        }
        if head_sha:
            payload["head_commit_id"] = head_sha
        if merge_when_checks_succeed:
            payload["merge_when_checks_succeed"] = True
        resp = self.session.post(url, json=payload, timeout=15)
        if resp.status_code in (200, 204):
            logger.info("PR #%d merged (%s) in %s", pr_number, strategy, repo_full_name)
            return True
        logger.error("Failed to merge PR #%d: %s %s", pr_number, resp.status_code, resp.text[:200])
        return False

    def add_label_to_pr(self, repo_full_name: str, pr_number: int) -> None:
        """Add the raven-reviewed label to a PR."""
        owner, repo = _split_repo(repo_full_name)
        labels_url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/labels"
        resp = self.session.get(labels_url, timeout=10)
        if resp.status_code != 200:
            return
        label_id = None
        for label in resp.json():
            if label.get("name") == RAVEN_LABEL_NAME:
                label_id = label["id"]
                break
        if not label_id:
            return
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/issues/{pr_number}/labels"
        self.session.post(url, json={"labels": [label_id]}, timeout=10)

    def get_commit_status(self, repo_full_name: str, sha: str) -> str:
        """Return the combined commit status: 'success', 'pending', 'failure', 'error', or 'none'."""
        owner, repo = _split_repo(repo_full_name)
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/commits/{sha}/status"
        resp = self.session.get(url, timeout=10)
        if resp.status_code != 200:
            return "none"
        data = resp.json()
        state = data.get("state", "")
        if not state or not data.get("statuses"):
            return "none"
        return state

    # ------------------------------------------------------------------ #
    #  File fetching                                                       #
    # ------------------------------------------------------------------ #

    def fetch_file(self, repo_full_name: str, path: str, ref: str = "HEAD") -> str:
        """Return decoded file contents, or empty string if not found."""
        owner, repo = _split_repo(repo_full_name)
        # Quote the path so names with '#', '?', ' ', or other URL-sensitive
        # characters don't break routing or get mis-parsed as query strings.
        # safe='/' preserves directory separators.
        encoded_path = quote(path, safe="/")
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/contents/{encoded_path}"
        resp = self.session.get(url, params={"ref": ref}, timeout=15)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        data = resp.json()
        content_b64 = data.get("content", "")
        if content_b64:
            return base64.b64decode(content_b64).decode("utf-8", errors="replace")
        return ""

    def list_directory(self, repo_full_name: str, path: str, ref: str = "HEAD") -> list[str]:
        """List regular files directly under ``path`` at ``ref`` (flat).

        Returns [] for missing directories (404) and any other API error
        so a missing ``.claude/rules/`` degrades gracefully rather than
        blocking the review.
        """
        owner, repo = _split_repo(repo_full_name)
        encoded_path = quote(path, safe="/")
        url = f"{self.base_url}/api/v1/repos/{owner}/{repo}/contents/{encoded_path}"
        try:
            resp = self.session.get(url, params={"ref": ref}, timeout=15)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            entries = resp.json()
        except Exception as e:
            logger.warning("Failed to list %s@%s: %s", path, ref, e)
            return []
        if not isinstance(entries, list):
            # Gitea returns an object (not a list) when ``path`` is a file,
            # not a directory. Treat that as "no directory here".
            return []
        # Gitea returns full repo-rooted paths. Trust the server but add
        # a sanity guard that the returned entry is actually a direct
        # child of ``path`` — rejects surprising API changes or hostile
        # responses whose 'path' escapes the requested directory
        # (e.g. ``..\\..\\etc\\passwd``).
        expected_prefix = path.rstrip("/") + "/"
        result: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            entry_path = entry.get("path")
            if not entry_path or not isinstance(entry_path, str):
                continue
            if not entry_path.startswith(expected_prefix):
                logger.warning("Gitea list_directory returned entry outside %s: %r — skipping", path, entry_path)
                continue
            tail = entry_path[len(expected_prefix):]
            if "/" in tail or tail in ("", ".", ".."):
                # Direct children only; no further path separators allowed.
                continue
            result.append(entry_path)
        return result

    # ------------------------------------------------------------------ #
    #  Webhook handling                                                    #
    # ------------------------------------------------------------------ #

    def validate_signature(self, request) -> None:
        """Validate HMAC-SHA256 signature from X-Gitea-Signature header."""
        secret = self.webhook_secret
        sig_header = request.headers.get("X-Gitea-Signature", "")

        if not sig_header:
            logger.warning("Missing X-Gitea-Signature header")
            abort(403, "Missing signature")

        body = request.get_data()
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig_header):
            logger.warning("Webhook signature mismatch")
            abort(403, "Invalid signature")

    def parse_webhook(self, request) -> tuple[str, dict] | None:
        """Parse Gitea webhook into normalized (event_type, payload) or None if ignored.

        Event types: "push", "pr_opened", "pr_updated", "pr_reopened",
                     "review_requested", "review_approved", "review_rejected",
                     "comment", "diff_comment"
        """
        payload = request.get_json(force=True, silent=True) or {}
        event = request.headers.get("X-Gitea-Event", "")

        if event == "push":
            return self._parse_push(payload)
        elif event == "pull_request_review_request":
            return self._parse_review_request(payload)
        elif event in ("pull_request", "pull_request_sync"):
            return self._parse_pull_request(payload, is_sync=(event == "pull_request_sync"))
        elif event in ("issue_comment", "pull_request_comment"):
            return self._parse_comment(payload, event)
        elif event in ("pull_request_review_approved", "pull_request_review_rejected"):
            return self._parse_review_event(payload, event)
        else:
            return None

    def _parse_push(self, payload: dict) -> tuple[str, dict] | None:
        """Parse push event into normalized payload. Ignores tag pushes."""
        ref = payload.get("ref", "")
        if not ref.startswith("refs/heads/"):
            return None
        branch = ref.removeprefix("refs/heads/")
        default_branch = payload.get("repository", {}).get("default_branch", "main")
        repo = payload.get("repository", {}).get("full_name", "")
        sender = (payload.get("pusher", {}).get("login", "")
                  or payload.get("sender", {}).get("login", ""))

        return ("push", {
            "repo": repo,
            "sender": sender,
            "branch": branch,
            "default_branch": default_branch,
            "pr_number": None,
            "pr_title": None,
            "pr_url": None,
            "head_sha": None,
            "head_ref": None,
            "base_ref": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        })

    def _parse_review_request(self, payload: dict) -> tuple[str, dict] | None:
        """Parse pull_request_review_request event."""
        action = payload.get("action", "")
        if action != "review_requested":
            return None
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "")
        sender = payload.get("sender", {}).get("login", "")
        requested = payload.get("requested_reviewer", {}).get("login", "")

        return ("review_requested", {
            "repo": repo,
            "sender": sender,
            "pr_number": pr.get("number"),
            "pr_title": pr.get("title", ""),
            "pr_url": pr.get("html_url", ""),
            "head_sha": pr.get("head", {}).get("sha", ""),
            "head_ref": pr.get("head", {}).get("ref", ""),
            "base_ref": pr.get("base", {}).get("ref", ""),
            "requested_reviewer": requested,
            "branch": None,
            "default_branch": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        })

    def _parse_pull_request(self, payload: dict, is_sync: bool = False) -> tuple[str, dict] | None:
        """Parse pull_request or pull_request_sync event into normalized payload."""
        action = payload.get("action", "")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "")
        sender = payload.get("sender", {}).get("login", "")

        if is_sync:
            event_type = "pr_updated"
        else:
            action_map = {
                "opened": "pr_opened",
                "synchronize": "pr_updated",
                "reopened": "pr_reopened",
                "review_requested": "review_requested",
            }
            event_type = action_map.get(action)
            if event_type is None:
                return None

        normalized = {
            "repo": repo,
            "sender": sender,
            "pr_number": pr.get("number"),
            "pr_title": pr.get("title", ""),
            "pr_url": pr.get("html_url", ""),
            "head_sha": pr.get("head", {}).get("sha", ""),
            "head_ref": pr.get("head", {}).get("ref", ""),
            "base_ref": pr.get("base", {}).get("ref", ""),
            "branch": None,
            "default_branch": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        }

        if event_type == "review_requested":
            normalized["requested_reviewer"] = payload.get("requested_reviewer", {}).get("login", "")

        return (event_type, normalized)

    def _parse_comment(self, payload: dict, gitea_event: str) -> tuple[str, dict] | None:
        """Parse issue_comment or pull_request_comment event."""
        action = payload.get("action", "")
        if action != "created":
            return None

        # issue_comment requires is_pull
        if gitea_event == "issue_comment" and not payload.get("is_pull"):
            return None

        comment = payload.get("comment", {})
        repo = payload.get("repository", {}).get("full_name", "")
        sender = payload.get("sender", {}).get("login", "")

        # Determine PR number depending on event type
        if gitea_event == "pull_request_comment":
            pr_number = payload.get("pull_request", {}).get("number")
        else:
            pr_number = payload.get("issue", {}).get("number")

        # Determine if this is a diff comment
        file_path = comment.get("path", "") if gitea_event == "pull_request_comment" else None
        line = (comment.get("line") or None) if gitea_event == "pull_request_comment" else None
        is_diff_comment = gitea_event == "pull_request_comment"
        event_type = "diff_comment" if is_diff_comment else "comment"

        return (event_type, {
            "repo": repo,
            "sender": sender,
            "pr_number": pr_number,
            "pr_title": None,
            "pr_url": None,
            "head_sha": None,
            "head_ref": None,
            "base_ref": None,
            "comment_body": comment.get("body", ""),
            "comment_user": comment.get("user", {}).get("login", ""),
            "comment_id": comment.get("id"),
            "file_path": file_path,
            "line": line,
            "branch": None,
            "default_branch": None,
        })

    def _parse_review_event(self, payload: dict, gitea_event: str) -> tuple[str, dict] | None:
        """Parse pull_request_review_approved / pull_request_review_rejected events."""
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "")
        sender = payload.get("sender", {}).get("login", "")

        event_type = "review_approved" if gitea_event == "pull_request_review_approved" else "review_rejected"

        return (event_type, {
            "repo": repo,
            "sender": sender,
            "pr_number": pr.get("number"),
            "pr_title": pr.get("title", ""),
            "pr_url": pr.get("html_url", ""),
            "head_sha": pr.get("head", {}).get("sha", ""),
            "head_ref": pr.get("head", {}).get("ref", ""),
            "base_ref": pr.get("base", {}).get("ref", ""),
            "branch": None,
            "default_branch": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        })


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _split_repo(repo_full_name: str) -> tuple[str, str]:
    """Split 'owner/repo' into ('owner', 'repo')."""
    parts = repo_full_name.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid repo_full_name: {repo_full_name!r}")
    return parts[0], parts[1]
