"""providers — Git platform abstraction layer."""

from abc import ABC, abstractmethod


class GitProvider(ABC):
    """Abstract interface for git platform operations."""

    name: str  # "gitea", "bitbucket-cloud", "bitbucket-dc"
    # True if post_pr_comment with parent_comment_id actually threads the reply.
    # False for flat-comment providers (e.g. Gitea issue comments).
    supports_comment_threads: bool = False

    @abstractmethod
    def get_authenticated_user(self) -> str: ...

    @abstractmethod
    def find_open_pr_for_branch(self, repo: str, branch: str) -> dict | None:
        """Return normalized PR dict or None. Keys: number, title, html_url, head.sha, head.ref, base.ref."""
        ...

    @abstractmethod
    def get_pr_head_sha(self, repo: str, pr_number: int) -> str: ...

    @abstractmethod
    def get_pr_base_ref(self, repo: str, pr_number: int) -> str:
        """Return the PR's base branch name (e.g. ``"main"``).

        Used to fetch base-branch-pinned configuration such as
        ``.claude/rules/`` and prompt overrides. Raises on fetch error or
        missing value — callers should catch and fall back.
        """
        ...

    @abstractmethod
    def fetch_pr_diff(self, repo: str, pr_number: int) -> str: ...

    @abstractmethod
    def fetch_file(self, repo: str, path: str, ref: str = "HEAD") -> str: ...

    def list_directory(self, repo: str, path: str, ref: str = "HEAD") -> list[str]:
        """List file paths directly under ``path`` at ``ref`` (flat, not
        recursive). Used by the review flow to discover repo-supplied
        rule files under ``.claude/rules/``.

        Returned entries are relative to the repo root (e.g.
        ``.claude/rules/security.md``). Only regular files are included;
        subdirectories are skipped since rule loading is flat.

        Default returns ``[]`` — providers that don't implement it
        simply disable the rules feature gracefully. Real implementations
        must be tolerant: any API failure (404, auth, transport) should
        return ``[]`` rather than raise, since a missing directory is
        the common case and must not block the review.
        """
        return []

    @abstractmethod
    def get_pr_comments(self, repo: str, pr_number: int) -> list[dict]:
        """Return comments as [{"user": {"login": str}, "body": str, "id": int}]."""
        ...

    def get_pr_description(self, repo: str, pr_number: int) -> str:
        """Return the PR's top-level description/body as plain text.

        Used to feed author-supplied intent (design notes, "intentionally
        skipping X because Y", referenced tickets) into the review prompt
        so the model has the same context a human reviewer would. Default
        returns an empty string — providers should override. Best-effort:
        any API failure should return ``""`` rather than raise, since
        review must proceed without it.
        """
        return ""

    @abstractmethod
    def post_pr_comment(self, repo: str, pr_number: int, body: str,
                        parent_comment_id: int | None = None) -> dict:
        """Post a comment. If parent_comment_id is set and the provider supports
        threading, the comment is posted as a reply (thread) — providers that
        don't support threading ignore the parent and post a top-level comment.
        """
        ...

    def react_to_comment(self, repo: str, pr_number: int, comment_id: int,
                         content: str = "eyes") -> None:
        """Best-effort emoji reaction on a comment.

        Default is a no-op — platforms without a reactions API keep the base
        class behaviour. Providers that do support reactions (Gitea) override
        this to give immediate acknowledgment that Raven saw a comment before
        the full response is generated. Must not raise on failure — this is
        fire-and-forget UX, not correctness-critical.
        """
        return

    @abstractmethod
    def submit_review(self, repo: str, pr_number: int, body: str,
                      approve: bool, inline_comments: list[dict] | None = None,
                      commit_id: str = "", comment_only: bool = False) -> dict:
        """Submit review. inline_comments: [{"file": str, "line": int, "body": str}].

        ``comment_only=True`` posts the body + inline comments without
        recording a blocking verdict — used for advisory mode. Returns
        dict with ``"id"``.

        Note: the default value here is documentation/intention only;
        Python doesn't propagate ABC defaults to override implementations.
        Call sites pass ``comment_only`` conditionally via dict-spread
        so out-of-tree providers running in non-advisory modes never see
        the new kwarg.
        """
        ...

    @abstractmethod
    def dismiss_previous_reviews(self, repo: str, pr_number: int, bot_user: str,
                                  exclude_id: int | None = None) -> None: ...

    @abstractmethod
    def get_pr_reviews(self, repo: str, pr_number: int) -> list[dict]:
        """Return the list of reviews submitted on the PR.

        Each review dict has the shape::

            {"user": {"login": str}, "state": str}

        ``state`` is one of ``"APPROVED"``, ``"REQUEST_CHANGES"``,
        ``"COMMENTED"``, or provider-specific values. Providers normalise
        their native review shapes into this common form.
        """
        ...

    @abstractmethod
    def get_pr_requested_reviewers(self, repo: str, pr_number: int) -> list[str]: ...

    @abstractmethod
    def add_self_as_reviewer(self, repo: str, pr_number: int) -> None:
        """Add the authenticated bot user as a reviewer on the PR. Idempotent."""
        ...

    @abstractmethod
    def merge_pr(self, repo: str, pr_number: int,
                 commit_title: str = "", strategy: str = "squash",
                 head_sha: str = "", merge_when_checks_succeed: bool = False) -> bool: ...

    @abstractmethod
    def add_label_to_pr(self, repo: str, pr_number: int) -> None: ...

    @abstractmethod
    def get_commit_status(self, repo: str, sha: str) -> str: ...

    @abstractmethod
    def validate_signature(self, request) -> None: ...

    @abstractmethod
    def parse_webhook(self, request) -> tuple[str, dict] | None: ...

    # ── Comment-thread-context feature (concrete defaults so out-of-tree ──
    # ── providers degrade gracefully; in-tree providers override).        ──

    def get_comment_thread(self, repo: str, pr_number: int,
                           root_comment_id: int) -> list[dict]:
        """Return the thread containing root_comment_id, chronologically
        ordered. Provider-specific notion of 'thread':
          - BB DC: the genuine reply-tree rooted at root_comment_id (DFS).
          - Gitea: all review comments sharing the same (path, position).

        Each entry::

            {"id": int, "parent_id": int | None,
             "user": {"login": str}, "body": str,
             "file_path": str | None, "line": int | None,
             "resolved": bool}

        ``parent_id`` is None on Gitea (no reply tree); reflects nesting
        on BB DC. ``resolved`` is True when the provider marks the comment
        resolved.

        Default returns []. Providers without threaded review comments
        inherit this; the thread-context feature degrades to the existing
        flat last-N conversation behaviour.
        """
        return []

    def get_pr_state(self, repo: str, pr_number: int) -> str:
        """Return one of 'open', 'merged', 'closed'. Default returns
        'open' so providers that don't implement this still allow
        comment-driven mutations (the existing safety gates in
        ``_safe_do_merge`` still catch a real state mismatch via the
        head-SHA recheck).
        """
        return "open"

    def retract_finding(self, repo: str, pr_number: int,
                        comment_id: int) -> bool:
        """Retract a Raven-authored inline finding that the conversation
        proved invalid. Returns True on success, False on no-op or
        non-fatal failure. Must not raise on 403 / 404 — log and return
        False.

        Default returns False. Out-of-tree providers without a native
        resolve action see retract calls become silent no-ops; AI-flagged
        retractions log a warning and the comment stays open in the UI.
        Verdict revision and thread context still work normally.
        """
        return False

    def get_pr_metadata(self, repo: str, pr_number: int) -> dict:
        """Return ``{'title': str, 'html_url': str}`` for log /
        notification / merge-commit-title purposes. Default returns
        ``{}`` — caller falls back to defaults. Auto-merge dispatch
        still works; merge-commit titles and notifications are less rich.
        """
        return {}


# Provider registry — populated at app startup
_providers: dict[str, GitProvider] = {}


def register_provider(name: str, provider: GitProvider) -> None:
    _providers[name] = provider


def get_provider(name: str) -> GitProvider | None:
    return _providers.get(name)


def registered_providers() -> dict[str, GitProvider]:
    return dict(_providers)
