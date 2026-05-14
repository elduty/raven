"""Tests for providers/gitea.py — GiteaProvider API client."""

import base64
import os
import pytest
import requests
from unittest.mock import MagicMock, patch


from raven.providers.gitea import GiteaProvider, _split_repo

GITEA_BASE = "https://gitea.example.com"


@pytest.fixture()
def client():
    return GiteaProvider(base_url=GITEA_BASE, token="test-token", webhook_secret="testsecret")


def _mock_get(client, status=200, text=None, json_data=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = text or ""
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    return patch.object(client.session, "get", return_value=mock_resp)


def _mock_post(client, status=201, json_data=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    return patch.object(client.session, "post", return_value=mock_resp)


# ------------------------------------------------------------------ #
#  _split_repo                                                        #
# ------------------------------------------------------------------ #

class TestSplitRepo:
    def test_valid(self):
        assert _split_repo("owner/repo") == ("owner", "repo")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _split_repo("no-slash")


# ------------------------------------------------------------------ #
#  Diff fetching                                                      #
# ------------------------------------------------------------------ #

class TestGetPrHeadSha:
    def test_returns_sha(self, client):
        with _mock_get(client, json_data={"head": {"sha": "abc123"}}):
            assert client.get_pr_head_sha("owner/repo", 7) == "abc123"

    def test_url_contains_pr_number(self, client):
        with _mock_get(client, json_data={"head": {"sha": "x"}}) as mock_get:
            client.get_pr_head_sha("owner/repo", 42)
        url = mock_get.call_args[0][0]
        assert "/pulls/42" in url
        assert ".diff" not in url

    def test_missing_sha_raises(self, client):
        with _mock_get(client, json_data={}):
            with pytest.raises(RuntimeError, match="empty head SHA"):
                client.get_pr_head_sha("owner/repo", 7)


class TestGetPrBaseRef:
    def test_returns_base_ref(self, client):
        with _mock_get(client, json_data={"base": {"ref": "main"}}):
            assert client.get_pr_base_ref("owner/repo", 7) == "main"

    def test_url_contains_pr_number(self, client):
        with _mock_get(client, json_data={"base": {"ref": "main"}}) as mock_get:
            client.get_pr_base_ref("owner/repo", 42)
        url = mock_get.call_args[0][0]
        assert "/pulls/42" in url
        assert ".diff" not in url

    def test_missing_base_ref_raises(self, client):
        with _mock_get(client, json_data={}):
            with pytest.raises(RuntimeError, match="empty base ref"):
                client.get_pr_base_ref("owner/repo", 7)


class TestFetchPrDiff:
    def test_success(self, client):
        diff_text = "diff --git a/bar b/bar\n+added line\n"
        with _mock_get(client, text=diff_text):
            result = client.fetch_pr_diff("owner/repo", 7)
        assert "diff --git" in result

    def test_url_uses_dot_diff(self, client):
        with _mock_get(client, text="diff") as mock_get:
            client.fetch_pr_diff("owner/repo", 42)
        url = mock_get.call_args[0][0]
        assert "/pulls/42.diff" in url

    def test_empty_diff_raises(self, client):
        with _mock_get(client, text=""):
            with pytest.raises(RuntimeError, match="empty diff"):
                client.fetch_pr_diff("owner/repo", 7)

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.fetch_pr_diff("owner/repo", 7)


class TestGetPrDescription:
    def test_returns_body(self, client):
        with _mock_get(client, json_data={"body": "Fixes #123\n\nApproach: ..."}):
            assert client.get_pr_description("owner/repo", 7) == "Fixes #123\n\nApproach: ..."

    def test_empty_body_returns_empty_string(self, client):
        with _mock_get(client, json_data={"body": ""}):
            assert client.get_pr_description("owner/repo", 7) == ""

    def test_null_body_returns_empty_string(self, client):
        """Gitea returns `null` (JSON) for PRs with no description, which
        json.loads maps to Python None. The provider must return "" not
        None so the caller can treat the result as a plain string."""
        with _mock_get(client, json_data={"body": None}):
            assert client.get_pr_description("owner/repo", 7) == ""

    def test_http_error_returns_empty_string(self, client):
        """Review must not block on an API hiccup. The reviewer degrades
        gracefully to an empty PR-context section."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_pr_description("owner/repo", 7) == ""


# ------------------------------------------------------------------ #
#  PR comment posting                                                 #
# ------------------------------------------------------------------ #

class TestPostPrComment:
    def test_success(self, client):
        with _mock_post(client, json_data={"id": 42}):
            result = client.post_pr_comment("owner/repo", 3, "🦅 review")
        assert result["id"] == 42

    def test_uses_body_field_not_content(self, client):
        with _mock_post(client) as mock_post:
            client.post_pr_comment("owner/repo", 3, "hello")
        payload = mock_post.call_args[1]["json"]
        assert "body" in payload
        assert payload["body"] == "hello"
        assert "content" not in payload

    def test_url_uses_issues_endpoint(self, client):
        with _mock_post(client) as mock_post:
            client.post_pr_comment("owner/repo", 5, "text")
        url = mock_post.call_args[0][0]
        assert "/issues/5/comments" in url

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("422")
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.post_pr_comment("owner/repo", 1, "body")


class TestReactToComment:
    def test_posts_to_issue_comments_reactions(self, client):
        with _mock_post(client, status=201) as mock_post:
            client.react_to_comment("owner/repo", 3, 42, content="eyes")
        url = mock_post.call_args[0][0]
        assert "/issues/comments/42/reactions" in url
        assert mock_post.call_args[1]["json"] == {"content": "eyes"}

    def test_swallows_404(self, client):
        """Diff/review comments have a different URL — 404 is expected and silent."""
        mock_resp = MagicMock(status_code=404)
        with patch.object(client.session, "post", return_value=mock_resp):
            client.react_to_comment("owner/repo", 3, 42)  # no raise

    def test_never_raises_on_network_error(self, client):
        with patch.object(client.session, "post",
                          side_effect=requests.ConnectionError("down")):
            client.react_to_comment("owner/repo", 3, 42)  # no raise


# ------------------------------------------------------------------ #
#  PR review submission                                               #
# ------------------------------------------------------------------ #

class TestDismissReview:
    def test_success(self, client):
        with _mock_post(client, status=200):
            client.dismiss_review("owner/repo", 3, 42)
        # No exception = success

    def test_url_contains_review_id(self, client):
        with _mock_post(client, status=200) as mock_post:
            client.dismiss_review("owner/repo", 7, 99)
        url = mock_post.call_args[0][0]
        assert "/reviews/99/dismissals" in url

    def test_failure_does_not_raise(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch.object(client.session, "post", return_value=mock_resp):
            client.dismiss_review("owner/repo", 3, 42)  # Should not raise


class TestSubmitReview:
    def test_success(self, client):
        with _mock_post(client, json_data={"id": 1}):
            result = client.submit_review("owner/repo", 3, "review body", approve=True)
        assert result["id"] == 1

    def test_url_contains_reviews(self, client):
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 7, "body", approve=False)
        url = mock_post.call_args[0][0]
        assert "/pulls/7/reviews" in url

    def test_payload_contains_body_and_event(self, client):
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "review text", approve=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["body"] == "review text"
        assert payload["event"] == "APPROVED"

    def test_payload_request_changes_when_not_approved(self, client):
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "review text", approve=False)
        payload = mock_post.call_args[1]["json"]
        assert payload["event"] == "REQUEST_CHANGES"

    def test_payload_includes_commit_id(self, client):
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "body", approve=True, commit_id="sha123")
        payload = mock_post.call_args[1]["json"]
        assert payload["commit_id"] == "sha123"

    def test_payload_omits_commit_id_when_empty(self, client):
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "body", approve=True)
        payload = mock_post.call_args[1]["json"]
        assert "commit_id" not in payload

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("422")
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.submit_review("owner/repo", 1, "body", approve=True)

    def test_comment_only_uses_event_comment(self, client):
        """comment_only=True bypasses the approve flag and sends
        event=COMMENT — the review carries body + inline anchors but
        no APPROVED / REQUEST_CHANGES verdict. Used for advisory mode."""
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "advisory body",
                                  approve=False, commit_id="sha1",
                                  comment_only=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["event"] == "COMMENT"
        # approve flag ignored in comment_only mode
        with _mock_post(client) as mock_post:
            client.submit_review("owner/repo", 3, "advisory body",
                                  approve=True, commit_id="sha1",
                                  comment_only=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["event"] == "COMMENT"


# ------------------------------------------------------------------ #
#  Authenticated user                                                  #
# ------------------------------------------------------------------ #

class TestGetAuthenticatedUser:
    def test_returns_login(self, client):
        with _mock_get(client, json_data={"login": "Raven"}):
            assert client.get_authenticated_user() == "Raven"

    def test_caches_result(self, client):
        with _mock_get(client, json_data={"login": "Raven"}) as mock_get:
            client.get_authenticated_user()
            client.get_authenticated_user()
        mock_get.assert_called_once()


# ------------------------------------------------------------------ #
#  PR reviews                                                          #
# ------------------------------------------------------------------ #

class TestGetPrReviews:
    def test_returns_reviews(self, client):
        reviews = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        page1 = MagicMock(status_code=200, raise_for_status=MagicMock())
        page1.json.return_value = reviews
        page2 = MagicMock(status_code=200, raise_for_status=MagicMock())
        page2.json.return_value = []
        with patch.object(client.session, "get", side_effect=[page1, page2]):
            result = client.get_pr_reviews("owner/repo", 7)
        assert len(result) == 1
        assert result[0]["state"] == "APPROVED"


class TestGetPrRequestedReviewers:
    def test_returns_logins_from_pr_object(self, client):
        """Reads the field from the PR object itself, not the separate
        /requested_reviewers endpoint. That dedicated endpoint returns
        404 on some Gitea versions; the PR object's field is reliable."""
        data = {
            "number": 7,
            "requested_reviewers": [{"login": "alice"}, {"login": "bob"}],
        }
        with _mock_get(client, json_data=data) as mock_get:
            result = client.get_pr_requested_reviewers("owner/repo", 7)
        assert result == ["alice", "bob"]
        url = mock_get.call_args[0][0]
        # Calls the PR object, not the dedicated /requested_reviewers path
        assert url.endswith("/repos/owner/repo/pulls/7")
        assert not url.endswith("/requested_reviewers")

    def test_empty_when_requested_reviewers_null(self, client):
        """Gitea returns requested_reviewers: null when no one is requested."""
        data = {"number": 7, "requested_reviewers": None}
        with _mock_get(client, json_data=data):
            result = client.get_pr_requested_reviewers("owner/repo", 7)
        assert result == []

    def test_empty_when_requested_reviewers_missing(self, client):
        """Defensive: field absent from response entirely."""
        data = {"number": 7}
        with _mock_get(client, json_data=data):
            result = client.get_pr_requested_reviewers("owner/repo", 7)
        assert result == []

    def test_skips_non_dict_entries(self, client):
        """Defensive: if the list contains something other than a user
        dict (unlikely but the old code tolerated it), skip it."""
        data = {
            "number": 7,
            "requested_reviewers": [{"login": "alice"}, None, "not-a-dict"],
        }
        with _mock_get(client, json_data=data):
            result = client.get_pr_requested_reviewers("owner/repo", 7)
        assert result == ["alice"]

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.get_pr_requested_reviewers("owner/repo", 7)


class TestAddSelfAsReviewer:
    def test_posts_authenticated_user(self, client):
        client._username = "Raven"
        with _mock_post(client, status=201) as mock_post:
            client.add_self_as_reviewer("owner/repo", 7)
        url = mock_post.call_args[0][0]
        assert url.endswith("/repos/owner/repo/pulls/7/requested_reviewers")
        assert mock_post.call_args[1]["json"] == {"reviewers": ["Raven"]}

    def test_200_accepted(self, client):
        client._username = "Raven"
        with _mock_post(client, status=200):
            client.add_self_as_reviewer("owner/repo", 7)  # no raise

    def test_http_error_raises(self, client):
        client._username = "Raven"
        mock_resp = MagicMock(status_code=500)
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.add_self_as_reviewer("owner/repo", 7)


# ------------------------------------------------------------------ #
#  PR merge                                                           #
# ------------------------------------------------------------------ #

class TestMergePr:
    def test_merge_success_returns_true(self, client):
        with _mock_post(client, status=204):
            result = client.merge_pr("owner/repo", 7, commit_title="squash title")
        assert result is True

    def test_merge_200_also_succeeds(self, client):
        with _mock_post(client, status=200):
            result = client.merge_pr("owner/repo", 7)
        assert result is True

    def test_merge_failure_returns_false(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 405
        mock_resp.text = "not mergeable"
        with patch.object(client.session, "post", return_value=mock_resp):
            result = client.merge_pr("owner/repo", 7)
        assert result is False

    def test_merge_uses_squash(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 3, commit_title="My PR")
        payload = mock_post.call_args[1]["json"]
        assert payload["Do"] == "squash"
        assert payload["delete_branch_after_merge"] is True

    def test_merge_uses_title_field_not_message(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 3, commit_title="My PR")
        payload = mock_post.call_args[1]["json"]
        assert payload["merge_title_field"] == "My PR"
        assert "merge_message_field" not in payload

    def test_merge_includes_head_commit_id(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 3, head_sha="abc123")
        payload = mock_post.call_args[1]["json"]
        assert payload["head_commit_id"] == "abc123"

    def test_merge_omits_head_commit_id_when_empty(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 3)
        payload = mock_post.call_args[1]["json"]
        assert "head_commit_id" not in payload

    def test_merge_when_checks_succeed(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 3, merge_when_checks_succeed=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["merge_when_checks_succeed"] is True

    def test_merge_url_correct(self, client):
        with _mock_post(client, status=204) as mock_post:
            client.merge_pr("owner/repo", 9)
        url = mock_post.call_args[0][0]
        assert "/pulls/9/merge" in url


# ------------------------------------------------------------------ #
#  Label operations                                                   #
# ------------------------------------------------------------------ #

class TestAddLabelToPr:
    def test_adds_label_when_found(self, client):
        labels_resp = MagicMock()
        labels_resp.status_code = 200
        labels_resp.json.return_value = [{"id": 5, "name": "raven-reviewed"}]

        add_resp = MagicMock()
        add_resp.status_code = 200

        with patch.object(client.session, "get", return_value=labels_resp), \
             patch.object(client.session, "post", return_value=add_resp) as mock_post:
            client.add_label_to_pr("owner/repo", 3)

        payload = mock_post.call_args[1]["json"]
        assert payload["labels"] == [5]

    def test_skips_gracefully_if_label_not_found(self, client):
        labels_resp = MagicMock()
        labels_resp.status_code = 200
        labels_resp.json.return_value = [{"id": 1, "name": "bug"}]

        with patch.object(client.session, "get", return_value=labels_resp), \
             patch.object(client.session, "post") as mock_post:
            client.add_label_to_pr("owner/repo", 3)

        mock_post.assert_not_called()

    def test_skips_gracefully_if_labels_api_fails(self, client):
        labels_resp = MagicMock()
        labels_resp.status_code = 403
        with patch.object(client.session, "get", return_value=labels_resp), \
             patch.object(client.session, "post") as mock_post:
            client.add_label_to_pr("owner/repo", 3)
        mock_post.assert_not_called()


# ------------------------------------------------------------------ #
#  File fetching                                                      #
# ------------------------------------------------------------------ #

class TestFetchFile:
    def test_success(self, client):
        content = base64.b64encode(b"# CLAUDE.md content").decode()
        with _mock_get(client, json_data={"content": content}):
            result = client.fetch_file("owner/repo", "CLAUDE.md", ref="main")
        assert result == "# CLAUDE.md content"

    def test_not_found_returns_empty_string(self, client):
        with _mock_get(client, status=404):
            result = client.fetch_file("owner/repo", "CLAUDE.md")
        assert result == ""

    def test_empty_content_returns_empty_string(self, client):
        with _mock_get(client, json_data={"content": ""}):
            result = client.fetch_file("owner/repo", "CLAUDE.md")
        assert result == ""

    def test_path_with_special_chars_is_url_encoded(self, client):
        """Paths with '#', '?', spaces must be quoted so they don't
        break routing or get mis-parsed as query strings."""
        content = base64.b64encode(b"x").decode()
        with _mock_get(client, json_data={"content": content}) as mock_get:
            client.fetch_file("owner/repo", "docs/issue #42.md", ref="main")
        url = mock_get.call_args[0][0]
        assert "#" not in url  # would be interpreted as fragment otherwise
        assert "docs/issue%20%2342.md" in url

    def test_path_with_subdirs_preserves_slashes(self, client):
        """Directory separators must stay as '/' — only filename parts
        should be percent-encoded."""
        content = base64.b64encode(b"x").decode()
        with _mock_get(client, json_data={"content": content}) as mock_get:
            client.fetch_file("owner/repo", "src/pkg/file.py")
        url = mock_get.call_args[0][0]
        assert "/src/pkg/file.py" in url


class TestListDirectory:
    def test_returns_file_paths(self, client):
        data = [
            {"type": "file", "path": ".claude/rules/security.md"},
            {"type": "file", "path": ".claude/rules/style.md"},
            {"type": "dir",  "path": ".claude/rules/nested"},
        ]
        with _mock_get(client, json_data=data):
            result = client.list_directory("owner/repo", ".claude/rules", ref="abc123")
        # Only regular files, not subdirs
        assert result == [".claude/rules/security.md", ".claude/rules/style.md"]

    def test_missing_directory_returns_empty(self, client):
        """The common case — no .claude/rules/ in the repo. Must not
        raise so the review continues with no rules context."""
        with _mock_get(client, status=404):
            assert client.list_directory("owner/repo", ".claude/rules") == []

    def test_http_error_returns_empty(self, client):
        """Transport error must not block the review either."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.list_directory("owner/repo", ".claude/rules") == []

    def test_path_is_a_file_returns_empty(self, client):
        """Gitea returns an object (not a list) when the path resolves
        to a file rather than a directory. Treat that as "not a directory"."""
        with _mock_get(client, json_data={"type": "file", "path": "README.md"}):
            assert client.list_directory("owner/repo", "README.md") == []

    def test_passes_ref_to_api(self, client):
        with _mock_get(client, json_data=[]) as mock_get:
            client.list_directory("owner/repo", ".claude/rules", ref="deadbeef")
        assert mock_get.call_args[1]["params"]["ref"] == "deadbeef"

    def test_rejects_entries_outside_requested_directory(self):
        """Sanity guard: Gitea's contract says returned entries are
        repo-rooted paths under the requested directory. If a hostile
        response (or future API change) returns paths that escape — e.g.
        ``../../../etc/passwd`` or nested subdirs — drop them rather than
        trust blindly and feed surprising paths into ``fetch_file``."""
        client = GiteaProvider(base_url=GITEA_BASE, token="test-token", webhook_secret="testsecret")
        entries = [
            {"type": "file", "path": ".claude/rules/good.md"},      # legit
            {"type": "file", "path": ".claude/other/evil.md"},      # sibling dir — reject
            {"type": "file", "path": "../../etc/passwd"},           # escape — reject
            {"type": "file", "path": ".claude/rules/nested/x.md"},  # nested — reject (flat only)
        ]
        with _mock_get(client, json_data=entries):
            result = client.list_directory("owner/repo", ".claude/rules")
        assert result == [".claude/rules/good.md"]


# ------------------------------------------------------------------ #
#  Webhook parsing                                                    #
# ------------------------------------------------------------------ #

class TestParseWebhook:
    @pytest.fixture()
    def client(self):
        return GiteaProvider(base_url=GITEA_BASE, token="test-token", webhook_secret="testsecret")

    def _make_request(self, event, payload):
        mock_req = MagicMock()
        mock_req.headers = {"X-Gitea-Event": event}
        mock_req.get_json.return_value = payload
        return mock_req

    def _pr_payload(self, action="opened", sender="alice"):
        return {
            "action": action,
            "repository": {"full_name": "owner/repo"},
            "pull_request": {
                "number": 7,
                "title": "My PR",
                "html_url": "http://x/pulls/7",
                "head": {"ref": "feature", "sha": "abc123"},
                "base": {"ref": "main"},
            },
            "sender": {"login": sender},
        }

    def test_pull_request_sync_parsed_as_pr_updated(self, client):
        result = client.parse_webhook(self._make_request("pull_request_sync", self._pr_payload()))
        assert result is not None
        event_type, data = result
        assert event_type == "pr_updated"
        assert data["pr_number"] == 7
        assert data["head_sha"] == "abc123"

    def test_pull_request_opened(self, client):
        result = client.parse_webhook(self._make_request("pull_request", self._pr_payload("opened")))
        assert result is not None
        assert result[0] == "pr_opened"

    def test_pull_request_synchronize(self, client):
        result = client.parse_webhook(self._make_request("pull_request", self._pr_payload("synchronize")))
        assert result is not None
        assert result[0] == "pr_updated"

    def test_pull_request_closed_ignored(self, client):
        result = client.parse_webhook(self._make_request("pull_request", self._pr_payload("closed")))
        assert result is None

    def test_review_approved_parsed(self, client):
        payload = self._pr_payload("reviewed")
        result = client.parse_webhook(self._make_request("pull_request_review_approved", payload))
        assert result is not None
        event_type, data = result
        assert event_type == "review_approved"
        assert data["pr_number"] == 7
        assert data["sender"] == "alice"

    def test_review_rejected_parsed(self, client):
        payload = self._pr_payload("reviewed", sender="bob")
        result = client.parse_webhook(self._make_request("pull_request_review_rejected", payload))
        assert result is not None
        event_type, data = result
        assert event_type == "review_rejected"
        assert data["sender"] == "bob"

    def test_unknown_event_returns_none(self, client):
        result = client.parse_webhook(self._make_request("release", {}))
        assert result is None

    def test_push_branch(self, client):
        payload = {
            "ref": "refs/heads/feature",
            "repository": {"full_name": "owner/repo", "default_branch": "main"},
            "pusher": {"login": "alice"},
        }
        result = client.parse_webhook(self._make_request("push", payload))
        assert result is not None
        event_type, data = result
        assert event_type == "push"
        assert data["branch"] == "feature"

    def test_push_tag_ignored(self, client):
        payload = {"ref": "refs/tags/v1.0", "repository": {"full_name": "owner/repo"}}
        result = client.parse_webhook(self._make_request("push", payload))
        assert result is None


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_pr_state                       #
# ------------------------------------------------------------------ #

class TestGetPrState:
    def test_open(self, client):
        with _mock_get(client, status=200, json_data={"state": "open", "merged": False}):
            assert client.get_pr_state("u/r", 1) == "open"

    def test_merged(self, client):
        with _mock_get(client, status=200, json_data={"state": "closed", "merged": True}):
            assert client.get_pr_state("u/r", 1) == "merged"

    def test_closed_without_merge(self, client):
        with _mock_get(client, status=200, json_data={"state": "closed", "merged": False}):
            assert client.get_pr_state("u/r", 1) == "closed"


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_pr_metadata                    #
# ------------------------------------------------------------------ #

class TestGetPrMetadata:
    def test_returns_title_and_url(self, client):
        with _mock_get(client, status=200, json_data={
            "title": "Add X",
            "html_url": "https://x/u/r/pulls/1",
        }):
            meta = client.get_pr_metadata("u/r", 1)
        assert meta == {"title": "Add X", "html_url": "https://x/u/r/pulls/1"}

    def test_returns_empty_dict_on_http_error(self, client):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("boom")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_pr_metadata("u/r", 1) == {}


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_comment_thread                 #
# ------------------------------------------------------------------ #

class TestGetCommentThread:
    def test_groups_by_path_and_position(self, client):
        """All review comments at same (path, position) form a thread,
        ordered chronologically. Different positions excluded."""
        reviews = [{"id": 10, "comments_count": 4}]
        review_comments = [
            {"id": 100, "user": {"login": "raven"}, "body": "Finding!",
             "path": "a.py", "position": 5, "resolver": None,
             "created_at": "2026-05-13T10:00:00Z"},
            {"id": 101, "user": {"login": "alice"}, "body": "Why?",
             "path": "a.py", "position": 5, "resolver": None,
             "created_at": "2026-05-13T10:05:00Z"},
            {"id": 102, "user": {"login": "raven"}, "body": "Because X",
             "path": "a.py", "position": 5, "resolver": {"login": "alice"},
             "created_at": "2026-05-13T10:10:00Z"},
            # Different position
            {"id": 200, "user": {"login": "carol"}, "body": "other thread",
             "path": "a.py", "position": 99, "resolver": None,
             "created_at": "2026-05-13T10:00:00Z"},
            # Different file
            {"id": 300, "user": {"login": "dave"}, "body": "unrelated",
             "path": "b.py", "position": 5, "resolver": None,
             "created_at": "2026-05-13T10:00:00Z"},
        ]

        def fake_get(url, **kwargs):
            mr = MagicMock()
            mr.raise_for_status = MagicMock()
            if url.endswith("/reviews"):
                mr.status_code = 200
                mr.json.return_value = reviews
            elif "/reviews/10/comments" in url:
                mr.status_code = 200
                mr.json.return_value = review_comments
            else:
                mr.status_code = 404
            return mr

        with patch.object(client.session, "get", side_effect=fake_get):
            thread = client.get_comment_thread("u/r", 1, 100)
        ids = [c["id"] for c in thread]
        assert ids == [100, 101, 102]
        assert all(c["parent_id"] is None for c in thread)
        assert thread[0]["file_path"] == "a.py"
        assert thread[0]["line"] == 5
        assert thread[2]["resolved"] is True
        assert thread[0]["resolved"] is False

    def test_thread_lookup_works_from_any_member(self, client):
        reviews = [{"id": 10, "comments_count": 2}]
        review_comments = [
            {"id": 100, "user": {"login": "x"}, "body": "a",
             "path": "a.py", "position": 5, "resolver": None,
             "created_at": "2026-05-13T10:00:00Z"},
            {"id": 101, "user": {"login": "y"}, "body": "b",
             "path": "a.py", "position": 5, "resolver": None,
             "created_at": "2026-05-13T10:05:00Z"},
        ]

        def fake_get(url, **kwargs):
            mr = MagicMock()
            mr.raise_for_status = MagicMock()
            if url.endswith("/reviews"):
                mr.status_code = 200
                mr.json.return_value = reviews
            elif "/reviews/10/comments" in url:
                mr.status_code = 200
                mr.json.return_value = review_comments
            else:
                mr.status_code = 404
            return mr

        with patch.object(client.session, "get", side_effect=fake_get):
            t1 = client.get_comment_thread("u/r", 1, 100)
            t2 = client.get_comment_thread("u/r", 1, 101)
        assert [c["id"] for c in t1] == [100, 101]
        assert [c["id"] for c in t2] == [100, 101]

    def test_returns_empty_when_root_not_found(self, client):
        reviews = [{"id": 10, "comments_count": 1}]
        review_comments = [{"id": 100, "user": {"login": "x"}, "body": "t",
                            "path": "a.py", "position": 1, "resolver": None,
                            "created_at": "2026-05-13T10:00:00Z"}]

        def fake_get(url, **kwargs):
            mr = MagicMock()
            mr.raise_for_status = MagicMock()
            if url.endswith("/reviews"):
                mr.status_code = 200
                mr.json.return_value = reviews
            elif "/reviews/10/comments" in url:
                mr.status_code = 200
                mr.json.return_value = review_comments
            else:
                mr.status_code = 404
            return mr

        with patch.object(client.session, "get", side_effect=fake_get):
            thread = client.get_comment_thread("u/r", 1, 999)
        assert thread == []


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: retract_finding                    #
# ------------------------------------------------------------------ #

class TestRetractFinding:
    def test_posts_to_native_resolve_endpoint(self, client):
        post_resp = MagicMock(status_code=204)
        post_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=post_resp) as mock_post:
            assert client.retract_finding("u/r", 1, 42) is True
        url = mock_post.call_args.args[0]
        assert url.endswith("/repos/u/r/pulls/comments/42/resolve")

    def test_404_returns_false(self, client):
        post_resp = MagicMock(status_code=404)
        with patch.object(client.session, "post", return_value=post_resp):
            assert client.retract_finding("u/r", 1, 42) is False

    def test_403_returns_false(self, client):
        post_resp = MagicMock(status_code=403)
        with patch.object(client.session, "post", return_value=post_resp):
            assert client.retract_finding("u/r", 1, 42) is False


# ------------------------------------------------------------------ #
#  Comment-thread-context: submit_review returns per-inline IDs       #
# ------------------------------------------------------------------ #

class TestSubmitReviewInlineIds:
    def test_returns_per_inline_comment_ids(self, client):
        """After POST /reviews, a follow-up GET /reviews/{id}/comments
        returns each inline comment's id. submit_review matches them by
        (path, position, body) and returns inline_comments aligned with
        input."""
        inline = [
            {"file": "a.py", "line": 5, "body": "finding 1"},
            {"file": "b.py", "line": 20, "body": "finding 2"},
        ]
        post_resp = MagicMock(status_code=200)
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"id": 700}  # review id
        get_resp = MagicMock(status_code=200)
        get_resp.raise_for_status = MagicMock()
        get_resp.json.return_value = [
            {"id": 800, "path": "a.py", "position": 5, "body": "finding 1"},
            {"id": 801, "path": "b.py", "position": 20, "body": "finding 2"},
        ]
        with patch.object(client.session, "post", return_value=post_resp), \
             patch.object(client.session, "get", return_value=get_resp):
            result = client.submit_review(
                "u/r", 1, "review body", approve=True,
                inline_comments=inline,
            )
        ic = result["inline_comments"]
        assert len(ic) == 2
        assert ic[0] == {"file": "a.py", "line": 5, "comment_id": 800}
        assert ic[1] == {"file": "b.py", "line": 20, "comment_id": 801}

    def test_filtered_input_gets_none_id(self, client):
        """Inputs missing file/line are filtered before posting. The
        returned list stays aligned with input — filtered entries get
        comment_id=None."""
        inline = [
            {"file": "a.py", "line": 5, "body": "ok"},
            {"file": "", "line": 0, "body": "no file"},
        ]
        post_resp = MagicMock(status_code=200)
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"id": 700}
        get_resp = MagicMock(status_code=200)
        get_resp.raise_for_status = MagicMock()
        get_resp.json.return_value = [
            {"id": 800, "path": "a.py", "position": 5, "body": "ok"},
        ]
        with patch.object(client.session, "post", return_value=post_resp), \
             patch.object(client.session, "get", return_value=get_resp):
            result = client.submit_review(
                "u/r", 1, "body", approve=True, inline_comments=inline,
            )
        ic = result["inline_comments"]
        assert len(ic) == 2
        assert ic[0]["comment_id"] == 800
        assert ic[1]["comment_id"] is None

    def test_follow_up_get_failure_returns_none_ids(self, client):
        """If the GET fails after a successful review post, comment_ids
        fall back to None — the review still landed; we just couldn't
        capture the IDs for retraction matching."""
        import requests
        inline = [{"file": "a.py", "line": 5, "body": "f"}]
        post_resp = MagicMock(status_code=200)
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = {"id": 700}
        get_resp = MagicMock()
        get_resp.raise_for_status.side_effect = requests.HTTPError("boom")
        with patch.object(client.session, "post", return_value=post_resp), \
             patch.object(client.session, "get", return_value=get_resp):
            result = client.submit_review(
                "u/r", 1, "body", approve=True, inline_comments=inline,
            )
        assert result["inline_comments"][0]["comment_id"] is None
