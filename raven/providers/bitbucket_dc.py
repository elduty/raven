"""bitbucket_dc.py — BitbucketDCProvider: Bitbucket Data Center API client implementing GitProvider ABC."""

import hashlib
import hmac
import logging
import time
from urllib.parse import quote

import requests
from flask import abort

from raven.providers import GitProvider, DiffTruncatedError

logger = logging.getLogger(__name__)

# Max pages (50 activities/page) scanned by the three activities-endpoint
# consumers: thread-root discovery, resolved-comment IDs, PR comment context.
# 30 pages = 1500 activities, comfortably above a chatty multi-week PR while
# staying cheap. Loops that exit without ``isLastPage=true`` emit a WARNING
# so an operator can spot pathological PRs that may be dropping state.
_ACTIVITIES_MAX_PAGES = 30


class BitbucketDCProvider(GitProvider):
    """Bitbucket Data Center API client implementing the GitProvider interface."""

    name = "bitbucket-dc"
    supports_comment_threads = True

    def __init__(self, base_url: str, token: str, webhook_secret: str, username: str = ""):
        self.base_url = base_url.rstrip("/")
        # ``/rest/api/latest/`` aliases to ``/rest/api/1.0/`` — the only
        # versioned API surface BB DC publishes. v9.0's "REST v2 migration"
        # was an internal Jersey/Jackson rearchitecture, not a URL bump;
        # ``latest`` is the documented stable entrypoint. We track it
        # deliberately (rather than pinning to ``1.0``) so a future v2 URL
        # would only surface as a deliberate Atlassian-side rollout.
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

    def get_pr_base_ref(self, repo_full_name: str, pr_number: int) -> str:
        """Return the PR's base branch name from ``toRef.displayId``."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        ref = resp.json().get("toRef", {}).get("displayId", "")
        if not ref:
            raise RuntimeError(f"Bitbucket DC returned empty base ref for PR #{pr_number}")
        return ref

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
        ``_json_diff_to_unified`` to convert.

        Note: on modern BB DC the JSON branch is the **primary** code
        path, not a fallback — the session-level Content-Type doesn't
        send an Accept header and BB DC picks JSON by default. The
        text-branch is the fallback (kept for older servers that ignore
        the Content-Type and return plain text).
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/diff"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = resp.json()
            # Fail closed on a truncated diff: BB DC caps diff size and
            # returns only part of the change. Reviewing the partial diff
            # would let Raven APPROVE + auto-merge code the model never saw
            # (audit 2026-06-13 finding #1). Refuse instead — the review
            # flow's error handler posts an actionable comment and blocks
            # the merge; the comment / cached-merge flows abort safely too.
            if self._diff_response_truncated(data):
                raise DiffTruncatedError(
                    f"Bitbucket DC returned a truncated diff for PR #{pr_number}: "
                    f"the change exceeds the server's diff size limit, so part of "
                    f"it is missing from the response. Refusing to review a partial "
                    f"diff (would risk approving/merging unseen code)."
                )
            diff_text = self._json_diff_to_unified(data)
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

    @staticmethod
    def _diff_response_truncated(data: dict) -> bool:
        """True if a BB DC JSON diff response signals truncation at ANY level.

        BB DC caps diff size and flags truncation on the overall response
        AND, independently, on individual file diffs, hunks, segments, and
        lines (Atlassian's streaming-diff model: the top-level flag means
        "at least one hunk was omitted"; the finer flags mark a hunk/segment
        cut mid-content, which the top-level flag does not always reflect).
        Any of them means the model would see incomplete code, so we check
        every level and fail closed. Absent flags read falsy, so a complete
        diff never trips a false positive.
        """
        if not isinstance(data, dict):
            return False
        if data.get("truncated"):
            return True
        for d in data.get("diffs") or []:
            if not isinstance(d, dict):
                continue
            if d.get("truncated"):
                return True
            for h in d.get("hunks") or []:
                if not isinstance(h, dict):
                    continue
                if h.get("truncated"):
                    return True
                for seg in h.get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    if seg.get("truncated"):
                        return True
                    for ln in seg.get("lines") or []:
                        if isinstance(ln, dict) and ln.get("truncated"):
                            return True
        return False

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

    # ── New methods for the comment-thread-context feature ───────────── #

    def _find_thread_root_via_activities(self, repo_full_name: str, pr_number: int,
                                          comment_id: int) -> dict | None:
        """Find a comment's thread root by searching the PR's activities.

        BB DC's ``GET /comments/{id}`` response shape (``ActivityComment``)
        encodes thread relationships only via nested ``comments`` (children)
        arrays — there is NO ``parent`` field. So we can walk DOWN from a
        root, but never UP from a deep reply.

        The activities endpoint
        (``GET /pull-requests/{id}/activities``) exposes the full top-down
        tree: each ``COMMENTED`` activity carries a top-level comment with
        nested children. To find the root of any comment, scan activities
        and recursively descend each top-level comment until we find the
        one containing the target id. Returns that top-level comment dict,
        or ``None`` if not found / on error.
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/activities"
        start = 0
        last_data: dict = {}
        for _ in range(_ACTIVITIES_MAX_PAGES):
            try:
                resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
                resp.raise_for_status()
            except requests.HTTPError as e:
                logger.warning("BB DC activities fetch failed for PR #%d: %s", pr_number, e)
                return None
            data = resp.json()
            last_data = data
            for activity in data.get("values", []):
                if activity.get("action") != "COMMENTED":
                    continue
                top_comment = activity.get("comment")
                if not top_comment:
                    continue
                if self._comment_contains_id(top_comment, comment_id):
                    return top_comment
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 50)
        else:
            if not last_data.get("isLastPage", True):
                logger.warning(
                    "BB DC _find_thread_root for PR #%d: activities scan capped at %d pages; "
                    "thread-root lookup may have missed the seed comment. Bump _ACTIVITIES_MAX_PAGES "
                    "if this PR is legitimately this large.",
                    pr_number, _ACTIVITIES_MAX_PAGES,
                )
        return None

    def _comment_contains_id(self, node: dict, target_id: int) -> bool:
        """Recursive DFS: does this comment tree contain ``target_id``?"""
        if node.get("id") == target_id:
            return True
        for child in node.get("comments") or []:
            if isinstance(child, dict) and self._comment_contains_id(child, target_id):
                return True
        return False

    def get_comment_thread(self, repo_full_name: str, pr_number: int,
                           root_comment_id: int) -> list[dict]:
        """Return the full thread rooted at the CONVERSATION ROOT,
        parent-first DFS.

        The webhook gives us the immediate parent of the trigger comment,
        which on threads deeper than 2 levels is NOT the conversation
        root. BB DC's ``GET /comments/{id}`` response has no ``parent``
        field, so we can't walk UP from a deep node. Instead we search
        the activities endpoint top-down for the tree containing the
        seed id and start emitting from THAT root.
        """
        root_node = self._find_thread_root_via_activities(
            repo_full_name, pr_number, root_comment_id,
        )
        if root_node is None:
            return []

        out: list[dict] = []

        def _walk(node: dict, parent_id: int | None) -> None:
            anchor = node.get("anchor") or {}
            out.append({
                "id": node.get("id"),
                "parent_id": parent_id,
                "user": {"login": (node.get("author") or {}).get("slug", "")},
                "body": node.get("text", ""),
                "file_path": anchor.get("path"),
                "line": anchor.get("line"),
                # Thread resolution (UI "Resolve thread" button) sets
                # ``threadResolved`` per BB DC's RestComment schema. The
                # ``state`` enum (OPEN/PENDING/RESOLVED) is task-specific
                # (severity=BLOCKER) and stays OPEN on regular thread-
                # resolved comments. Treat either as resolved so the AI
                # sees both signals.
                "resolved": (
                    node.get("threadResolved") is True
                    or node.get("state") == "RESOLVED"
                ),
            })
            for child in node.get("comments") or []:
                if isinstance(child, dict):
                    _walk(child, node.get("id"))

        _walk(root_node, None)
        return out

    def get_pr_state(self, repo_full_name: str, pr_number: int) -> str:
        """Map BB DC PR state to canonical 'open' / 'merged' / 'closed'.
        BB DC values: OPEN, MERGED, DECLINED. DECLINED maps to 'closed'."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        state = resp.json().get("state", "OPEN").upper()
        return {"OPEN": "open", "MERGED": "merged", "DECLINED": "closed"}.get(state, "open")

    def get_pr_metadata(self, repo_full_name: str, pr_number: int) -> dict:
        """Return {'title', 'html_url'} for log/notification/merge-commit-title.
        Returns {} on fetch failure; caller falls back to defaults."""
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            logger.warning("BB DC get_pr_metadata for PR #%d failed: %s", pr_number, e)
            return {}
        self_link = ""
        for link in (data.get("links") or {}).get("self", []):
            if link.get("href"):
                self_link = link["href"]
                break
        return {"title": data.get("title", ""), "html_url": self_link}

    def retract_finding(self, repo_full_name: str, pr_number: int,
                        comment_id: int) -> bool:
        """Resolve a comment thread on BB DC.

        BB DC's ``RestComment`` schema has **two distinct writable
        fields** that look similar but aren't:

        * ``state`` (enum ``OPEN`` / ``PENDING`` / ``RESOLVED``) — for
          **tasks** (severity=BLOCKER). Setting it on a regular comment
          or on a comment with a parent returns 400 "Cannot resolve a
          comment with a parent comment."
        * ``threadResolved`` (boolean) — for **thread resolution**.
          This is what the "Resolve thread" UI button maps to.
          ``threadResolvedDate`` and ``threadResolver`` are the readOnly
          counterparts BB DC populates.

        Source: Atlassian's static OpenAPI spec at
        ``dac-static.atlassian.com/server/bitbucket/9.6.swagger.v3.json``.

        We set ``threadResolved: true``. Callers should pass the thread
        root id (``get_comment_thread`` makes the tree available so the
        caller can walk to root in memory — the GET response shape has
        no parent linkage, so the provider can't do it).

        Returns ``False`` on 403/404/409 without raising. On any
        non-2xx, logs the response body (truncated to 500 chars).
        """
        project, repo = _split_repo(repo_full_name)
        base = (
            f"{self.api_url}/projects/{project}/repos/{repo}"
            f"/pull-requests/{pr_number}/comments/{comment_id}"
        )
        try:
            get_resp = self.session.get(base, timeout=10)
            if get_resp.status_code == 404:
                logger.debug("Cannot retract BB DC comment %s — 404", comment_id)
                return False
            get_resp.raise_for_status()
            version = get_resp.json().get("version", 0)
            put_resp = self.session.put(
                base, json={"threadResolved": True, "version": version}, timeout=10,
            )
            if put_resp.status_code in (403, 404, 409):
                logger.warning("BB DC retract comment %s failed: HTTP %s — body: %s",
                               comment_id, put_resp.status_code,
                               (put_resp.text or "")[:500])
                return False
            put_resp.raise_for_status()
            return True
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "") or ""
            logger.warning("BB DC retract comment %s failed: %s — body: %s",
                           comment_id, e, body[:500])
            return False

    def get_resolved_comment_ids(self, repo_full_name: str, pr_number: int) -> set[int]:
        """Return IDs of comments the developer has marked resolved.

        Scans the activities endpoint (the same source ``get_comment_thread``
        uses) and emits IDs of top-level thread roots where either
        ``threadResolved is True`` (the boolean BB DC's "Resolve thread"
        UI button sets per the RestComment schema) or
        ``state == "RESOLVED"`` (legacy resolved-task path for
        severity=BLOCKER comments). Both signals are checked because the
        read-side audit cleanup found that thread-resolved comments leave
        ``state`` at OPEN — relying on ``state`` alone misses every
        UI-resolved thread.

        Raven posts inline review comments at the top level, so checking
        thread roots is sufficient for the carry-forward filter: a Raven
        finding's ``comment_id`` is always the root of its thread. On
        any HTTP error returns ``set()`` — the review proceeds without
        filtering rather than blocking on a transient outage.
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/activities"
        resolved: set[int] = set()
        start = 0
        last_data: dict = {}
        for _ in range(_ACTIVITIES_MAX_PAGES):
            try:
                resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
                resp.raise_for_status()
            except requests.HTTPError as e:
                logger.warning("BB DC get_resolved_comment_ids for PR #%d failed: %s", pr_number, e)
                return set()
            data = resp.json()
            last_data = data
            for activity in data.get("values", []):
                if activity.get("action") != "COMMENTED":
                    continue
                top = activity.get("comment")
                if not top:
                    continue
                if (top.get("threadResolved") is True
                        or top.get("state") == "RESOLVED"):
                    cid = top.get("id")
                    if isinstance(cid, int):
                        resolved.add(cid)
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 50)
        else:
            if not last_data.get("isLastPage", True):
                logger.warning(
                    "BB DC get_resolved_comment_ids for PR #%d: activities scan capped at %d pages; "
                    "some user-resolved findings beyond the cap may carry forward incorrectly.",
                    pr_number, _ACTIVITIES_MAX_PAGES,
                )
        return resolved

    def get_pr_comments(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return all comments on a PR via the activities endpoint.

        Returned in **chronological (oldest-first)** order to match the
        Gitea provider's contract — consumers (``server._process_comment``'s
        ``[-COMMENT_HISTORY:]`` "last N", ``reviewer._build_pr_context_section``'s
        ``reversed(...)`` budget walk) assume oldest-first. The BB DC
        ``/activities`` endpoint pages newest-first, so we collect across
        all pages (newest→oldest) and reverse the assembled list once
        before returning.

        Returns only **top-level** comments (each activity's root comment).
        Replies nested in the ``comments`` array of each activity are not
        extracted — they're already surfaced to the AI via the per-trigger
        ``## Active Thread`` block in the comment-reply prompt, so
        duplicating them here would only bloat the ``## Other PR
        Conversation`` context.
        """
        project, repo = _split_repo(repo_full_name)
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/activities"
        all_comments: list[dict] = []
        start = 0
        last_data: dict = {}
        for _ in range(_ACTIVITIES_MAX_PAGES):
            resp = self.session.get(url, params={"start": start, "limit": 50}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            last_data = data
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
        else:
            if not last_data.get("isLastPage", True):
                logger.warning(
                    "BB DC get_pr_comments for PR #%d: activities scan capped at %d pages; "
                    "older comments beyond the cap are missing from the prompt context.",
                    pr_number, _ACTIVITIES_MAX_PAGES,
                )
        # The activities feed pages newest-first; reverse the fully
        # assembled list once so callers get chronological (oldest-first)
        # order, matching the Gitea provider's contract.
        all_comments.reverse()
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
                      commit_id: str = "", comment_only: bool = False) -> dict:
        """Submit a review: post comment + approve or set needs-work.

        approve=True  -> post comment + POST /approve
        approve=False -> post comment + PUT /participants/{user} with NEEDS_WORK
        commit_id is accepted for ABC compatibility but not used by BB DC.
        comment_only: when True, posts body + inline anchor comments but
            skips the approve / needs-work participant call — review
            remains non-blocking. Used for advisory mode.

        Returns the BB DC main-comment dict, extended with an
        `inline_comments` key carrying per-input-entry `{file, line,
        comment_id}` — same length as input, with comment_id=None for
        any post that failed or was filtered out. Used by the
        comment-thread-context retraction cache cleanup.
        """
        project, repo = _split_repo(repo_full_name)

        # Post the main review comment FIRST. BB DC has no single-POST
        # review API (unlike Gitea), so the main comment and the inline
        # anchors are separate POSTs. If we posted the anchors first and
        # the main-comment POST then raised, the caller treats the whole
        # submit_review as failed and re-posts everything on the next
        # pass — duplicating every (already-posted) inline comment. By
        # posting the main comment first, a main-comment failure leaves
        # zero inline anchors on the PR: no orphans, no dupes (audit #14).
        comment_url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/comments"
        resp = self.session.post(comment_url, json={"text": body}, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        # Then post inline comments; capture per-anchor returned IDs.
        posted_inline = (
            self._post_inline_comments(project, repo, pr_number, inline_comments)
            if inline_comments else []
        )
        result["inline_comments"] = posted_inline

        # Approve / needs-work — skipped in comment_only mode so the
        # review remains advisory.
        if comment_only:
            pass
        elif approve:
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
                              comments: list[dict]) -> list[dict]:
        """Post inline (anchor) comments on a PR diff.

        Anchors use lineType=ADDED and fileType=TO because Raven's findings
        target new/changed code (the + side of the diff). BB DC also supports
        REMOVED/FROM for deleted lines and CONTEXT for unchanged context lines,
        but those don't apply to review findings.

        Returns a list aligned by index with the input ``comments`` list.
        Each entry is ``{file, line, comment_id}``; ``comment_id`` is None
        when the anchor was filtered (missing/invalid file/line) or the
        post failed. Same-length-as-input is the invariant the
        ``_process_pr`` cache-write step relies on to zip submitted
        findings with returned IDs.
        """
        url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}/comments"
        posted: list[dict] = []
        for c in comments:
            file_path = c.get("file", "")
            line = c.get("line", 0)
            body = c.get("body", "")
            entry = {"file": file_path, "line": line, "comment_id": None}
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
                    entry["comment_id"] = resp.json().get("id")
                except Exception as e:
                    logger.warning("Failed to post inline comment on %s:%d: %s", file_path, line, e)
            posted.append(entry)
        return posted

    def dismiss_previous_reviews(self, repo_full_name: str, pr_number: int, bot_user: str,
                                  exclude_id: int | None = None) -> None:
        """No-op on BB DC -- participant status is replaced by submit_review, not stacked."""
        logger.debug("dismiss_previous_reviews is a no-op on Bitbucket DC (status already overwritten)")

    def get_pr_reviews(self, repo_full_name: str, pr_number: int) -> list[dict]:
        """Return participants who have submitted a real review, normalized
        for server.py.

        Bitbucket DC's ``/participants`` endpoint returns *every* user who
        has touched the PR — the author, watchers, anyone who commented or
        reacted — each with a ``status`` field. Most of those have
        ``status=UNAPPROVED`` (the default — they haven't reviewed). On
        Gitea the equivalent ``/reviews`` endpoint only returns people who
        formally submitted an approval or change-request, which is the
        semantic the auto-merge gate is designed around.

        To preserve that semantic across providers, only return
        participants whose status is ``APPROVED`` or ``NEEDS_WORK``. This
        drops the author (who is always a participant), drops watchers,
        and drops random commenters — none of whom should count as
        reviewers gating the auto-merge.
        """
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
            status = p.get("status", "UNAPPROVED")
            if status == "APPROVED":
                state = "APPROVED"
            elif status == "NEEDS_WORK":
                state = "REQUEST_CHANGES"
            else:
                # UNAPPROVED / unknown — author or non-reviewing participant.
                # Skip; they don't represent a formal review submission.
                continue
            user = p.get("user", {})
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

        When head_sha is provided, the merge is pinned to the reviewed head: the PR's
        current fromRef.latestCommit must match head_sha or the merge is refused
        (force-push protection). The `version` param in the merge POST covers the
        remaining GET-to-POST gap — a concurrent push bumps the version and the
        merge fails with 409. merge_when_checks_succeed is accepted for ABC
        compatibility but not used.
        """
        project, repo = _split_repo(repo_full_name)
        # Need PR version for merge
        pr_url = f"{self.api_url}/projects/{project}/repos/{repo}/pull-requests/{pr_number}"
        pr_resp = self.session.get(pr_url, timeout=10)
        if pr_resp.status_code != 200:
            logger.error("Failed to fetch PR #%d for merge: %s", pr_number, pr_resp.status_code)
            return False
        pr_data = pr_resp.json()
        version = pr_data.get("version", 0)

        if head_sha:
            latest_commit = pr_data.get("fromRef", {}).get("latestCommit", "")
            if latest_commit != head_sha:
                logger.warning(
                    "Refusing to merge PR #%d in %s: head moved since review "
                    "(reviewed %s, current %s) — possible force-push",
                    pr_number, repo_full_name, head_sha, latest_commit,
                )
                return False

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
        elif event in ("pr:comment:added", "pr:comment:edited"):
            # Both routed through the same handler. Edits include
            # ``comment.version`` (incremented per edit); the server's
            # dedup key picks that up so each edit gets a distinct slot
            # and a user adding ``@raven`` to an existing comment can
            # still trigger a reply. ``previousComment`` (text before
            # edit) is in the payload for diagnostics but not surfaced.
            return self._parse_comment(payload)
        elif event == "pr:comment:deleted":
            # Surfaced for operator visibility but no state mutation:
            # the cached finding's ``comment_id`` is the Raven-posted
            # inline review comment (which users normally can't delete),
            # so a deletion is almost always a user removing their own
            # reply. Logging is enough; the route returns None and the
            # webhook 200s without dispatching.
            logger.info("BB DC pr:comment:deleted received for PR #%s — no action",
                        (payload.get("pullRequest") or {}).get("id"))
            return None
        elif event == "pr:reviewer:updated":
            return self._parse_pr_event(payload, "review_requested")
        elif event == "pr:reviewer:approved":
            # Mirror of Gitea's pull_request_review_approved. Server's
            # consumer is a no-op (the route only exists so webhook
            # deliveries get a clean "ignored: no action taken" response
            # instead of "unhandled"), but parity matters for operators
            # debugging cross-provider behavior.
            return self._parse_pr_event(payload, "review_approved")
        elif event == "pr:reviewer:changes_requested":
            return self._parse_pr_event(payload, "review_rejected")
        elif event == "pr:reviewer:unapproved":
            # No symmetric Gitea event; no behavioral need. Log + drop.
            logger.debug("BB DC pr:reviewer:unapproved received for PR #%s — no action",
                         (payload.get("pullRequest") or {}).get("id"))
            return None
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
        """Parse pr:comment:added and pr:comment:edited events.

        Both share the same payload shape; the edit case adds a
        ``previousComment`` field (text before edit, not surfaced here)
        and bumps ``comment.version``. The version is propagated as
        ``comment_version`` so the server's dedup includes it — a fresh
        edit (v=2, v=3, …) doesn't collide with the original add (v=1)
        and reprocessing kicks in for things like a user adding
        ``@raven`` to an existing comment.

        Check anchor for diff_comment vs comment.
        """
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
            "comment_version": comment.get("version"),
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
