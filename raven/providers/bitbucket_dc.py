"""bitbucket_dc.py — BitbucketDCProvider: Bitbucket Data Center API client implementing GitProvider ABC."""

import hashlib
import hmac
import logging
import time
from urllib.parse import quote

import requests
from flask import abort

from raven.providers import GitProvider

logger = logging.getLogger(__name__)


class BitbucketDCProvider(GitProvider):
    """Bitbucket Data Center API client implementing the GitProvider interface."""

    name = "bitbucket-dc"
    supports_comment_threads = True

    def __init__(self, base_url: str, token: str, webhook_secret: str, username: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/rest/api/latest"
        self.build_status_url = f"{self.base_url}/rest/build-status/latest"
        self.token = token
        if not webhook_secret:
            raise ValueError("webhook_secret is required — empty secret allows forged webhooks")
        self.webhook_secret = webhook_secret
        self._username = username or None
        self._default_branch_cache: dict[str, tuple[str, float]] = {}  # repo -> (branch, timestamp)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------ #
    #  Identity                                                           #
    # ------------------------------------------------------------------ #

    def get_authenticated_user(self) -> str:
        """Return the username of the authenticated user (cached).

        BB DC tokens don't have a direct 'whoami' endpoint. The username
        must be passed at init time, or we attempt to infer it from the
        /rest/api/latest/application-properties endpoint (which at least
        confirms the token works).
        """
        if self._username is None:
            raise RuntimeError(
                "BitbucketDCProvider requires 'username' at init — "
                "BB DC tokens have no whoami endpoint"
            )
        return self._username

    # ------------------------------------------------------------------ #
    #  PR discovery                                                       #
    # ------------------------------------------------------------------ #

    def find_open_pr_for_branch(self, repo_full_name: str, branch: str) -> dict | None:
        """Find an open PR whose source branch matches the given branch name."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests"
        resp = self.session.get(
            url,
            params={"state": "OPEN", "at": f"refs/heads/{branch}", "direction": "OUTGOING"},
            timeout=15,
        )
        resp.raise_for_status()
        values = resp.json().get("values", [])
        if values:
            pr = values[0]
            # Normalize to match the field names server.py expects
            return {
                "number": pr.get("id"),
                "title": pr.get("title", ""),
                "html_url": (pr.get("links", {}).get("self") or [{}])[0].get("href", ""),
                "head": {
                    "sha": pr.get("fromRef", {}).get("latestCommit", ""),
                    "ref": pr.get("fromRef", {}).get("displayId", ""),
                },
                "base": {
                    "ref": pr.get("toRef", {}).get("displayId", ""),
                },
            }
        return None

    def get_pr_head_sha(self, repo_full_name: str, pr_number: int) -> str:
        """Return the current head SHA of a PR from fromRef.latestCommit."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        sha = resp.json().get("fromRef", {}).get("latestCommit", "")
        if not sha:
            raise RuntimeError(f"Bitbucket DC returned empty head SHA for PR #{pr_number}")
        return sha

    def get_pr_description(self, repo_full_name: str, pr_number: int) -> str:
        """Return the PR description, or "" on any failure."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("description", "") or ""
        except Exception as e:
            logger.warning("Failed to fetch PR #%d description: %s", pr_number, e)
            return ""

    # ------------------------------------------------------------------ #
    #  Diff & file content                                                #
    # ------------------------------------------------------------------ #

    def fetch_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        """Return the raw unified diff for a pull request.

        Some BB Server versions (observed on 9.4.x) return 500 when the
        request carries ``Accept: text/plain``. Let the server pick its
        default (JSON in recent versions) and rely on
        ``_json_diff_to_unified`` to convert. The content-type check keeps
        the plain-text branch available if a server does return text.
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/diff"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            diff_text = self._json_diff_to_unified(resp.json())
        else:
            diff_text = resp.text
        if not diff_text.strip():
            raise RuntimeError(f"Bitbucket DC returned empty diff for PR #{pr_number}")
        return diff_text

    def _json_diff_to_unified(self, data: dict) -> str:
        """Convert BB DC JSON diff response to unified diff format."""
        lines: list[str] = []
        for diff_entry in data.get("diffs", []):
            src = (diff_entry.get("source") or {}).get("toString", "/dev/null")
            dst = (diff_entry.get("destination") or {}).get("toString", "/dev/null")
            src_header = "/dev/null" if src == "/dev/null" else f"a/{src}"
            dst_header = "/dev/null" if dst == "/dev/null" else f"b/{dst}"
            # Use the real file path for diff --git header
            file_path = dst if dst != "/dev/null" else src
            lines.append(f"diff --git a/{file_path} b/{file_path}")
            lines.append(f"--- {src_header}")
            lines.append(f"+++ {dst_header}")
            for hunk in diff_entry.get("hunks", []):
                src_line = hunk.get("sourceLine", 1)
                src_span = hunk.get("sourceSpan", 0)
                dst_line = hunk.get("destinationLine", 1)
                dst_span = hunk.get("destinationSpan", 0)
                lines.append(f"@@ -{src_line},{src_span} +{dst_line},{dst_span} @@")
                for segment in hunk.get("segments", []):
                    seg_type = segment.get("type", "CONTEXT")
                    prefix = " "
                    if seg_type == "ADDED":
                        prefix = "+"
                    elif seg_type == "REMOVED":
                        prefix = "-"
                    for line_obj in segment.get("lines", []):
                        lines.append(f"{prefix}{line_obj.get('line', '')}")
        return "\n".join(lines) + "\n"

    def fetch_file(self, repo_full_name: str, path: str, ref: str = "HEAD") -> str:
        """Return file contents, or empty string if not found.

        BB DC browse endpoint returns JSON with a 'lines' array.
        """
        project, repo = _split_repo(repo_full_name)
        # Quote the path so names with '#', '?', ' ', or other URL-sensitive
        # characters don't break routing or get mis-parsed as query strings.
        # safe='/' preserves directory separators.
        encoded_path = quote(path, safe="/")
        url = f"{self.api_url}/projects/{project}/repos/{repo}/browse/{encoded_path}"
        all_lines: list[str] = []
        start = 0
        max_pages = 20
        for _ in range(max_pages):
            resp = self.session.get(url, params={"at": ref, "start": start, "limit": 1000}, timeout=15)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            data = resp.json()
            for line in data.get("lines", []):
                all_lines.append(line.get("text", ""))
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 1000)
        if not all_lines:
            return ""
        # Trailing newline for parity with Gitea's fetch_file (and common
        # file conventions) so downstream line-count/offset logic doesn't
        # undercount by one on the last line.
        return "\n".join(all_lines) + "\n"

    def list_directory(self, repo_full_name: str, path: str, ref: str = "HEAD") -> list[str]:
        """List regular files directly under ``path`` at ``ref`` (flat).

        BB DC's ``/browse`` endpoint serves both files and directories:
        ``{type: "DIRECTORY", children: {values: [{type, path: {...}}]}}``
        for a directory hit, ``{type: "FILE", lines: [...]}`` for a file
        hit. We page through ``children`` and return full paths for
        entries whose ``type == "FILE"``.

        Missing directories (404) and any other transport error return
        [] so the review flow degrades gracefully rather than blocks.
        """
        project, repo = _split_repo(repo_full_name)
        encoded_path = quote(path, safe="/")
        url = f"{self.api_url}/projects/{project}/repos/{repo}/browse/{encoded_path}"
        files: list[str] = []
        start = 0
        max_pages = 10
        try:
            for _ in range(max_pages):
                resp = self.session.get(url, params={"at": ref, "start": start, "limit": 500}, timeout=15)
                if resp.status_code == 404:
                    return []
                resp.raise_for_status()
                data = resp.json()
                children = data.get("children") or {}
                values = children.get("values") or []
                for entry in values:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("type") != "FILE":
                        continue
                    # BB DC returns the child name in path.toString or
                    # path.components[-1]; prepend the parent path so the
                    # return value is repo-rooted like Gitea's.
                    entry_path = (entry.get("path") or {}).get("toString")
                    if not entry_path:
                        components = (entry.get("path") or {}).get("components") or []
                        entry_path = components[-1] if components else None
                    if not entry_path or not isinstance(entry_path, str):
                        continue
                    # BB DC's contract for a directory listing is that
                    # each child's path component is just the filename,
                    # not a nested/escaping path. Enforce it explicitly
                    # — rejects both hostile responses and surprising
                    # upstream API changes. Matches Gitea's "direct
                    # children only" guard.
                    if "/" in entry_path or "\\" in entry_path or entry_path in (".", ".."):
                        logger.warning(
                            "BB DC list_directory returned suspicious child name: %r — skipping",
                            entry_path,
                        )
                        continue
                    files.append(f"{path.rstrip('/')}/{entry_path}")
                if children.get("isLastPage", True):
                    break
                start = children.get("nextPageStart", start + 500)
        except Exception as e:
            logger.warning("Failed to list %s@%s: %s", path, ref, e)
            return []
        return files

    # ------------------------------------------------------------------ #
    #  Comments                                                           #
    # ------------------------------------------------------------------ #

    def get_comment_thread_authors(self, repo_full_name: str, pr_number: int,
                                   comment_id: int) -> list[str]:
        """Fetch the thread rooted at ``comment_id`` and return unique author
        slugs (root plus any nested ``comments[]`` replies).

        BB DC returns the root's replies inline in a single GET — the
        conceptual thread is flat but the response nests child comments one
        level. Walk recursively just in case a server returns deeper nesting.
        Returns [] on 404 so the caller can treat missing comments as an
        unknown thread.
        """
        project, repo = _split_repo(repo_full_name)
        url = (
            f"{self.api_url}/projects/{project}/repos/{repo}"
            f"/pull-requests/{pr_number}/comments/{comment_id}"
        )
        resp = self.session.get(url, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        seen: list[str] = []

        def _walk(node: dict) -> None:
            slug = (node.get("author") or {}).get("slug", "")
            if slug and slug not in seen:
                seen.append(slug)
            for child in node.get("comments") or []:
                if isinstance(child, dict):
                    _walk(child)

        _walk(resp.json())
        return seen

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return all comments on a PR via the activities endpoint."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/activities"
        all_comments: list[dict] = []
        start = 0
        max_pages = 10
        for _ in range(max_pages):
            resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for activity in data.get("values", []):
                if activity.get("action") == "COMMENTED":
                    comment = activity.get("comment", {})
                    if comment:
                        # Normalize to match contract: {user: {login}, body}
                        all_comments.append({
                            "user": {"login": comment.get("author", {}).get("slug", "")},
                            "body": comment.get("text", ""),
                            "id": comment.get("id"),
                        })
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 50)
        return all_comments

    def post_pr_comment(self, repo_full_name: str, pr_number: int, body: str,
                        parent_comment_id: int | None = None) -> dict:
        """Post a comment on a PR.

        If parent_comment_id is set, the comment is posted as a reply. BB DC
        threads are flat — replying to any comment in a thread lands the new
        comment in the same thread.
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/comments"
        payload: dict = {"text": body}
        if parent_comment_id:
            payload["parent"] = {"id": parent_comment_id}
        resp = self.session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    #  Reviews                                                            #
    # ------------------------------------------------------------------ #

    def submit_review(self, repo_full_name: str, pr_number: int, body: str,
                      approve: bool, inline_comments: list[dict] | None = None,
                      commit_id: str = "") -> dict:
        """Submit a review: post comment + approve or set needs-work.

        approve=True  -> post comment + POST /approve
        approve=False -> post comment + PUT /participants/{user} with NEEDS_WORK
        commit_id is accepted for ABC compatibility but not used by BB DC.
        """
        project, repo = _split_repo(repo_full_name)

        # Post inline comments first
        if inline_comments:
            self._post_inline_comments(project, repo, pr_number, inline_comments)

        # Post the main review comment
        comment_url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/comments"
        resp = self.session.post(comment_url, json={"text": body}, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        # Approve or set needs-work
        if approve:
            approve_url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/approve"
            approve_resp = self.session.post(approve_url, timeout=10)
            if approve_resp.status_code not in (200, 409):
                # 409 = already approved, which is fine
                approve_resp.raise_for_status()
        else:
            username = self.get_authenticated_user()
            participant_url = (
                f"{self.api_url}/projects/{project}/repos/{repo}"
                f"/pull-requests/{pr_number}/participants/{username}"
            )
            needs_work_resp = self.session.put(
                participant_url, json={"status": "NEEDS_WORK"}, timeout=10
            )
            needs_work_resp.raise_for_status()

        return result

    def _post_inline_comments(self, project: str, repo: str, pr_number: int,
                              comments: list[dict]) -> None:
        """Post inline (anchor) comments on a PR diff.

        Anchors use lineType=ADDED and fileType=TO because Raven's findings
        target new/changed code (the + side of the diff). BB DC also supports
        REMOVED/FROM for deleted lines and CONTEXT for unchanged context lines,
        but those don't apply to review findings.
        """
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/comments"
        for c in comments:
            file_path = c.get("file", "")
            line = c.get("line", 0)
            body = c.get("body", "")
            if file_path and isinstance(line, int) and line > 0:
                payload = {
                    "text": body,
                    "anchor": {
                        "line": line,
                        "lineType": "ADDED",
                        "fileType": "TO",
                        "path": file_path,
                    },
                }
                try:
                    resp = self.session.post(url, json=payload, timeout=10)
                    resp.raise_for_status()
                except Exception as e:
                    logger.warning("Failed to post inline comment on %s:%d: %s", file_path, line, e)

    def dismiss_previous_reviews(self, repo_full_name: str, pr_number: int, bot_user: str,
                                  exclude_id: int | None = None) -> None:
        """No-op on BB DC -- participant status is replaced by submit_review, not stacked."""
        logger.debug("dismiss_previous_reviews is a no-op on Bitbucket DC (status already overwritten)")

    def get_pr_reviews(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return participants with their review status, normalized for server.py."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/participants"
        all_participants: list[dict] = []
        start = 0
        max_pages = 10
        for _ in range(max_pages):
            resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            all_participants.extend(data.get("values", []))
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 50)
        normalized: list[dict] = []
        for p in all_participants:
            user = p.get("user", {})
            status = p.get("status", "UNAPPROVED")
            # Map BB DC status to Gitea-compatible state
            if status == "APPROVED":
                state = "APPROVED"
            elif status == "NEEDS_WORK":
                state = "REQUEST_CHANGES"
            else:
                state = "COMMENT"
            normalized.append({
                "user": {"login": user.get("slug", "")},
                "state": state,
            })
        return normalized

    def get_pr_requested_reviewers(self, repo_full_name: str, pr_number: int) -> list[str]:
        """Return usernames of users added as reviewers on this PR."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        resp = self.session.get(url, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        reviewers = resp.json().get("reviewers", [])
        return [r.get("user", {}).get("slug", "") for r in reviewers if isinstance(r, dict)]

    def add_self_as_reviewer(self, repo_full_name: str, pr_number: int) -> None:
        """Add the authenticated bot user as a reviewer via POST /participants. Idempotent.

        BB DC returns 409 Conflict if the user is already a participant — tolerated.
        The bot can't be added as a reviewer on its own PRs; 409 covers that case too.
        """
        project, repo = _split_repo(repo_full_name)
        username = self.get_authenticated_user()
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/participants"
        resp = self.session.post(
            url,
            json={"user": {"name": username}, "role": "REVIEWER"},
            timeout=10,
        )
        # 200/201 = added, 409 = already a participant (idempotent no-op)
        if resp.status_code not in (200, 201, 409):
            resp.raise_for_status()

    # ------------------------------------------------------------------ #
    #  PR operations                                                      #
    # ------------------------------------------------------------------ #

    def merge_pr(self, repo_full_name: str, pr_number: int,
                 commit_title: str = "", strategy: str = "squash",
                 head_sha: str = "", merge_when_checks_succeed: bool = False) -> bool:
        """Merge a PR with deleteSourceBranch. Strategy is controlled by repo settings in BB DC.

        head_sha and merge_when_checks_succeed are accepted for ABC compatibility but not used.
        """
        project, repo = _split_repo(repo_full_name)
        # Need PR version for merge
        pr_url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        pr_resp = self.session.get(pr_url, timeout=10)
        if pr_resp.status_code != 200:
            logger.error("Failed to fetch PR #%d for merge: %s", pr_number, pr_resp.status_code)
            return False
        version = pr_resp.json().get("version", 0)

        merge_url = f"{pr_url}/merge"
        resp = self.session.post(
            merge_url,
            json={"deleteSourceBranch": True, "message": commit_title or f"Merge PR #{pr_number}"},
            params={"version": version},
            timeout=15,
        )
        if resp.status_code in (200, 204):
            logger.info("PR #%d merged in %s", pr_number, repo_full_name)
            return True
        logger.error("Failed to merge PR #%d: %s %s", pr_number, resp.status_code, resp.text[:200])
        return False

    def add_label_to_pr(self, repo_full_name: str, pr_number: int) -> None:
        """No-op — Bitbucket Data Center has no label system."""
        logger.debug("add_label_to_pr is a no-op on Bitbucket Data Center")

    # ------------------------------------------------------------------ #
    #  CI status                                                          #
    # ------------------------------------------------------------------ #

    def get_commit_status(self, repo_full_name: str, sha: str) -> str:
        """Return aggregated commit status: 'success', 'pending', 'failure', or 'none'.

        BB DC statuses: SUCCESSFUL, FAILED, INPROGRESS.
        """
        url = f"{self.build_status_url}/commits/{sha}"
        all_statuses: list[dict] = []
        api_error = False
        start = 0
        max_pages = 5
        for _ in range(max_pages):
            resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
            if resp.status_code != 200:
                logger.warning("Build status API returned %d for %s", resp.status_code, sha[:8])
                api_error = True
                break
            data = resp.json()
            all_statuses.extend(data.get("values", []))
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 50)
        if not all_statuses:
            return "pending" if api_error else "none"
        if api_error:
            # Partial data — treat as pending to avoid merging with incomplete CI info
            return "pending"
        states = [s.get("state", "") for s in all_statuses]
        if any(s in ("FAILED", "CANCELLED") for s in states):
            return "failure"
        if all(s == "SUCCESSFUL" for s in states):
            return "success"
        if any(s in ("INPROGRESS", "UNKNOWN") for s in states):
            return "pending"
        return "none"

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    _NEGATIVE_CACHE_TTL = 300  # retry failed lookups after 5 minutes

    def _get_default_branch(self, project: str, repo: str) -> str:
        """Fetch the default branch for a repo (cached). Failures cached with TTL."""
        key = f"{project}/{repo}"
        cached = self._default_branch_cache.get(key)
        if cached is not None:
            branch, ts = cached
            if branch or (time.time() - ts < self._NEGATIVE_CACHE_TTL):
                return branch
            # Negative cache expired — retry
        url = f"{self.api_url}/projects/{project}/repos/{repo}/branches/default"
        now = time.time()
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                branch = resp.json().get("displayId", "")
                self._default_branch_cache[key] = (branch, now)
                return branch
            logger.debug("Default branch lookup for %s returned %d", key, resp.status_code)
        except Exception as e:
            logger.debug("Default branch lookup for %s failed: %s", key, e)
        self._default_branch_cache[key] = ("", now)
        return ""

    # ------------------------------------------------------------------ #
    #  Webhook handling                                                   #
    # ------------------------------------------------------------------ #

    def validate_signature(self, request) -> None:
        """Validate HMAC-SHA256 signature from X-Hub-Signature header.

        Format: sha256=<hex>. Strip the sha256= prefix before comparing.

        Exception: BB DC's "Test connection" button fires a ``diagnostics:ping``
        event with no signature header even when a secret is configured. Allow
        it through so operators can verify reachability from the BB DC UI; the
        payload contains no event data and is subsequently ignored by
        parse_webhook.
        """
        if request.headers.get("X-Event-Key", "") == "diagnostics:ping":
            return

        sig_header = request.headers.get("X-Hub-Signature", "")
        if not sig_header:
            logger.warning("Missing X-Hub-Signature header")
            abort(403, "Missing signature")

        # Strip sha256= prefix
        if sig_header.startswith("sha256="):
            sig_hex = sig_header[7:]
        else:
            logger.warning("X-Hub-Signature missing sha256= prefix")
            abort(403, "Invalid signature format")

        body = request.get_data()
        expected = hmac.new(
            self.webhook_secret.encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig_hex):
            logger.warning("Webhook signature mismatch")
            abort(403, "Invalid signature")

    def parse_webhook(self, request) -> tuple[str, dict] | None:
        """Parse Bitbucket DC webhook into normalized (event_type, payload) or None.

        Event types: "push", "pr_opened", "pr_updated", "pr_reopened",
                     "review_requested", "comment", "diff_comment"
        """
        payload = request.get_json(force=True, silent=True) or {}
        event = request.headers.get("X-Event-Key", "")

        if event == "repo:refs_changed":
            return self._parse_push(payload)
        elif event == "pr:opened":
            return self._parse_pr_event(payload, "pr_opened")
        elif event == "pr:from_ref_updated":
            return self._parse_pr_event(payload, "pr_updated")
        elif event == "pr:reopened":
            return self._parse_pr_event(payload, "pr_reopened")
        elif event == "pr:comment:added":
            return self._parse_comment(payload)
        elif event == "pr:reviewer:updated":
            return self._parse_pr_event(payload, "review_requested")
        else:
            return None

    def _parse_push(self, payload: dict) -> tuple[str, dict] | None:
        """Parse repo:refs_changed event. Ignores tag pushes."""
        changes = payload.get("changes", [])
        # Only handle branch refs
        branch_change = None
        for change in changes:
            ref = change.get("ref", {})
            if ref.get("type") == "BRANCH":
                branch_change = change
                break
        if branch_change is None:
            return None

        branch = branch_change["ref"].get("displayId", "")
        repo_data = payload.get("repository", {})
        project = repo_data.get("project", {}).get("key", "")
        repo_slug = repo_data.get("slug", "")
        repo_full = f"{project}/{repo_slug}"
        sender = payload.get("actor", {}).get("slug", "")

        # BB DC doesn't include default branch directly in push payload
        default_branch = self._get_default_branch(project, repo_slug)

        return ("push", {
            "repo": repo_full,
            "sender": sender,
            "branch": branch,
            "default_branch": default_branch,
            "pr_number": None,
            "pr_title": None,
            "pr_url": None,
            "head_sha": branch_change.get("toHash", ""),
            "head_ref": None,
            "base_ref": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        })

    def _parse_pr_event(self, payload: dict, event_type: str) -> tuple[str, dict] | None:
        """Parse pr:opened, pr:from_ref_updated, pr:reopened, pr:reviewer:updated events."""
        pr = payload.get("pullRequest", {})
        repo_data = pr.get("toRef", {}).get("repository", {})
        project = repo_data.get("project", {}).get("key", "")
        repo_slug = repo_data.get("slug", "")
        repo_full = f"{project}/{repo_slug}"
        sender = payload.get("actor", {}).get("slug", "")

        from_ref = pr.get("fromRef", {})
        to_ref = pr.get("toRef", {})

        # Build PR URL from links
        pr_links = pr.get("links", {}).get("self", [])
        pr_url = pr_links[0].get("href", "") if pr_links else ""

        normalized = {
            "repo": repo_full,
            "sender": sender,
            "pr_number": pr.get("id"),
            "pr_title": pr.get("title", ""),
            "pr_url": pr_url,
            "head_sha": from_ref.get("latestCommit", ""),
            "head_ref": from_ref.get("displayId", ""),
            "base_ref": to_ref.get("displayId", ""),
            "branch": None,
            "default_branch": None,
            "comment_body": None,
            "comment_user": None,
            "comment_id": None,
            "file_path": None,
            "line": None,
        }

        if event_type == "review_requested":
            added = payload.get("addedReviewers", [])
            if added:
                normalized["requested_reviewer"] = added[0].get("slug", "")

        return (event_type, normalized)

    def _parse_comment(self, payload: dict) -> tuple[str, dict] | None:
        """Parse pr:comment:added event. Check anchor for diff_comment vs comment."""
        pr = payload.get("pullRequest", {})
        comment = payload.get("comment", {})
        repo_data = pr.get("toRef", {}).get("repository", {})
        project = repo_data.get("project", {}).get("key", "")
        repo_slug = repo_data.get("slug", "")
        repo_full = f"{project}/{repo_slug}"
        sender = payload.get("actor", {}).get("slug", "")

        anchor = comment.get("anchor")
        is_diff_comment = anchor is not None
        event_type = "diff_comment" if is_diff_comment else "comment"

        file_path = anchor.get("path", "") if anchor else None
        line = anchor.get("line") if anchor else None

        # Parent comment id — present when this comment is a reply in a thread.
        # BB DC puts it at the payload root; some versions also nest a parent
        # object inside the comment itself. Use ``(x or {})`` rather than a
        # dict.get() default so an explicit ``"parent": null`` doesn't raise.
        parent_comment_id = payload.get("commentParentId") \
            or (comment.get("parent") or {}).get("id")

        return (event_type, {
            "repo": repo_full,
            "sender": sender,
            "pr_number": pr.get("id"),
            "pr_title": None,
            "pr_url": None,
            "head_sha": None,
            "head_ref": None,
            "base_ref": None,
            "comment_body": comment.get("text", ""),
            "comment_user": comment.get("author", {}).get("slug", ""),
            "comment_id": comment.get("id"),
            "parent_comment_id": parent_comment_id,
            "file_path": file_path,
            "line": line,
            "branch": None,
            "default_branch": None,
        })


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _split_repo(repo_full_name: str) -> tuple[str, str]:
    """Split 'project/repo' into ('project', 'repo')."""
    parts = repo_full_name.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid repo_full_name: {repo_full_name!r}")
    return parts[0], parts[1]
