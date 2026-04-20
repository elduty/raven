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
    def fetch_pr_diff(self, repo: str, pr_number: int) -> str: ...

    @abstractmethod
    def fetch_file(self, repo: str, path: str, ref: str = "HEAD") -> str: ...

    @abstractmethod
    def get_pr_comments(self, repo: str, pr_number: int) -> list[dict]:
        """Return comments as [{"user": {"login": str}, "body": str, "id": int}]."""
        ...

    @abstractmethod
    def get_comment_thread_authors(self, repo: str, pr_number: int,
                                   comment_id: int) -> list[str]:
        """Return the set of unique author logins/slugs in the comment's
        thread — the comment at ``comment_id`` plus any replies below it.

        Used to detect replies inside a Raven-involved thread so Raven can
        answer without being re-@mentioned, even when Raven wasn't the thread
        root. Providers without threaded comments return an empty list.
        """
        ...

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
                      commit_id: str = "") -> dict:
        """Submit review. inline_comments: [{"file": str, "line": int, "body": str}]. Returns dict with "id"."""
        ...

    @abstractmethod
    def dismiss_previous_reviews(self, repo: str, pr_number: int, bot_user: str,
                                  exclude_id: int | None = None) -> None: ...

    @abstractmethod
    def get_pr_reviews(self, repo: str, pr_number: int) -> list[dict]:
        """Return reviews as [{"user": {"login": str}, "state": "APPROVED"|"REQUEST_CHANGES"|"COMMENT"}]."""
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


# Provider registry — populated at app startup
_providers: dict[str, GitProvider] = {}


def register_provider(name: str, provider: GitProvider) -> None:
    _providers[name] = provider


def get_provider(name: str) -> GitProvider | None:
    return _providers.get(name)


def registered_providers() -> dict[str, GitProvider]:
    return dict(_providers)
