"""Tests for providers/gitea.py — GiteaProvider API client."""

import base64
import os
import pytest
import requests
from unittest.mock import MagicMock, patch

os.environ.setdefault("GITEA_URL", "https://gitea.example.com")
os.environ.setdefault("GITEA_TOKEN", "test-token")

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

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("422")
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.submit_review("owner/repo", 1, "body", approve=True)


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
    def test_returns_logins(self, client):
        data = {"users": [{"login": "alice"}, {"login": "bob"}]}
        with _mock_get(client, json_data=data):
            result = client.get_pr_requested_reviewers("owner/repo", 7)
        assert result == ["alice", "bob"]

    def test_http_error_raises(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                client.get_pr_requested_reviewers("owner/repo", 7)


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
