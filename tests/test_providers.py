"""Tests for the GitProvider ABC concrete defaults (Task 2 of the
comment-thread-context plan).

The four new methods added in this task are NOT @abstractmethod —
they're concrete with safe-degraded defaults so out-of-tree providers
(a public extension contract per CLAUDE.md) keep working. Tests
assert the defaults; the per-provider override tests in
test_gitea.py / test_bitbucket_dc.py prove the impls deviate from
defaults correctly."""

from raven.providers import GitProvider


class _MinimalProvider(GitProvider):
    """Stub implementing only the truly abstract members of GitProvider.
    Used to exercise concrete defaults on the new methods."""
    name = "test"

    def get_authenticated_user(self): return ""
    def find_open_pr_for_branch(self, r, b): return None
    def get_pr_head_sha(self, r, p): return ""
    def get_pr_base_ref(self, r, p): return ""
    def fetch_pr_diff(self, r, p): return ""
    def fetch_file(self, r, p, ref="HEAD"): return ""
    def get_pr_comments(self, r, p): return []
    def post_pr_comment(self, r, p, b, parent_comment_id=None): return {}
    def submit_review(self, r, p, b, a, inline_comments=None, commit_id=""): return {}
    def dismiss_previous_reviews(self, r, p, u, exclude_id=None): return None
    def get_pr_reviews(self, r, p): return []
    def get_pr_requested_reviewers(self, r, p): return []
    def add_self_as_reviewer(self, r, p): return None
    def merge_pr(self, r, p, **kw): return False
    def add_label_to_pr(self, r, p): return None
    def get_commit_status(self, r, sha): return ""
    def validate_signature(self, request): return None
    def parse_webhook(self, request): return None


def test_get_comment_thread_default_returns_empty_list():
    """Out-of-tree provider with no override -> degrades to no thread context."""
    assert _MinimalProvider().get_comment_thread("u/r", 1, 42) == []


def test_get_pr_state_default_returns_open():
    """Default to 'open' so comment-driven mutations aren't blocked by an
    over-conservative default. _safe_do_merge's head-SHA recheck still
    catches real state mismatches."""
    assert _MinimalProvider().get_pr_state("u/r", 1) == "open"


def test_retract_finding_default_returns_false():
    """Out-of-tree provider with no override -> retraction silently no-ops."""
    assert _MinimalProvider().retract_finding("u/r", 1, 42) is False


def test_get_pr_metadata_default_returns_empty_dict():
    """Caller falls back to title='PR #N' and empty URL."""
    assert _MinimalProvider().get_pr_metadata("u/r", 1) == {}
