"""providers — Git platform abstraction layer."""

from abc import ABC, abstractmethod


class GitProvider(ABC):
    """Abstract interface for git platform operations."""

    name: str  # "gitea", "bitbucket-cloud", "bitbucket-dc"

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
    def post_pr_comment(self, repo: str, pr_number: int, body: str) -> dict: ...

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
