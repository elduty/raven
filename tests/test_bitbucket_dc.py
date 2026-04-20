"""Tests for providers/bitbucket_dc.py — BitbucketDCProvider API client."""

import hashlib
import hmac as hmac_mod
import json
import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("GITEA_URL", "https://gitea.example.com")
os.environ.setdefault("GITEA_TOKEN", "test-token")

from raven.providers.bitbucket_dc import BitbucketDCProvider, _split_repo

BB_DC_BASE = "https://bitbucket.example.com"


@pytest.fixture()
def client():
    return BitbucketDCProvider(
        base_url=BB_DC_BASE,
        token="test-token",
        webhook_secret="testsecret",
        username="raven-bot",
    )


def _mock_get(client, status=200, text=None, json_data=None, content_type="text/plain"):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = text or ""
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"Content-Type": content_type}
    return patch.object(client.session, "get", return_value=mock_resp)


def _mock_post(client, status=201, json_data=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    return patch.object(client.session, "post", return_value=mock_resp)


def _mock_put(client, status=200, json_data=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = json_data or {}
    mock_resp.raise_for_status = MagicMock()
    return patch.object(client.session, "put", return_value=mock_resp)


def _make_request(payload, event_key, secret="testsecret"):
    """Build a mock Flask request with proper signature."""
    body = json.dumps(payload).encode()
    sig = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()
    req = MagicMock()
    req.headers = {
        "X-Hub-Signature": f"sha256={sig}",
        "X-Event-Key": event_key,
    }
    req.get_data.return_value = body
    req.get_json.return_value = payload
    return req


# ------------------------------------------------------------------ #
#  Initialization                                                     #
# ------------------------------------------------------------------ #

class TestInit:
    def test_creates_provider(self):
        p = BitbucketDCProvider(BB_DC_BASE, "tok", "secret", username="user")
        assert p.name == "bitbucket-dc"
        assert p.base_url == BB_DC_BASE
        assert p.api_url == f"{BB_DC_BASE}/rest/api/latest"
        assert p.build_status_url == f"{BB_DC_BASE}/rest/build-status/latest"

    def test_strips_trailing_slash(self):
        p = BitbucketDCProvider(f"{BB_DC_BASE}/", "tok", "secret", username="u")
        assert p.base_url == BB_DC_BASE

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError, match="webhook_secret is required"):
            BitbucketDCProvider(BB_DC_BASE, "tok", "", username="u")

    def test_auth_header_is_bearer(self):
        p = BitbucketDCProvider(BB_DC_BASE, "mytoken", "secret", username="u")
        assert p.session.headers["Authorization"] == "Bearer mytoken"


# ------------------------------------------------------------------ #
#  _split_repo                                                        #
# ------------------------------------------------------------------ #

class TestSplitRepo:
    def test_valid(self):
        assert _split_repo("PROJECT/repo") == ("PROJECT", "repo")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _split_repo("no-slash")


# ------------------------------------------------------------------ #
#  Identity                                                           #
# ------------------------------------------------------------------ #

class TestGetAuthenticatedUser:
    def test_returns_username(self, client):
        assert client.get_authenticated_user() == "raven-bot"

    def test_raises_without_username(self):
        p = BitbucketDCProvider(BB_DC_BASE, "tok", "secret")
        with pytest.raises(RuntimeError, match="requires 'username'"):
            p.get_authenticated_user()


# ------------------------------------------------------------------ #
#  Webhook signature validation                                       #
# ------------------------------------------------------------------ #

class TestValidateSignature:
    def test_valid_signature(self, client):
        req = _make_request({"test": True}, "pr:opened", secret="testsecret")
        # Should not raise
        client.validate_signature(req)

    def test_invalid_signature_aborts(self, client):
        req = _make_request({"test": True}, "pr:opened", secret="wrong-secret")
        with pytest.raises(Exception):
            client.validate_signature(req)

    def test_missing_header_aborts(self, client):
        req = MagicMock()
        req.headers = {}
        with pytest.raises(Exception):
            client.validate_signature(req)

    def test_missing_prefix_aborts(self, client):
        body = b'{"test": true}'
        sig = hmac_mod.new(b"testsecret", body, hashlib.sha256).hexdigest()
        req = MagicMock()
        req.headers = {"X-Hub-Signature": sig}  # no sha256= prefix
        req.get_data.return_value = body
        with pytest.raises(Exception):
            client.validate_signature(req)

    def test_diagnostics_ping_accepted_without_signature(self, client):
        """BB DC's Test connection button sends diagnostics:ping with no signature.

        Allow it through so operators can verify reachability from the BB DC UI.
        """
        req = MagicMock()
        req.headers = {"X-Event-Key": "diagnostics:ping"}
        client.validate_signature(req)  # no raise


# ------------------------------------------------------------------ #
#  parse_webhook — push                                               #
# ------------------------------------------------------------------ #

class TestParseWebhookPush:
    def test_branch_push(self, client):
        payload = {
            "changes": [{"ref": {"type": "BRANCH", "displayId": "feature-x"}}],
            "repository": {"slug": "myrepo", "project": {"key": "PROJ"}},
            "actor": {"slug": "alice"},
        }
        req = _make_request(payload, "repo:refs_changed")
        with patch.object(client, "_get_default_branch", return_value="main"):
            result = client.parse_webhook(req)
        assert result is not None
        event_type, data = result
        assert event_type == "push"
        assert data["repo"] == "PROJ/myrepo"
        assert data["branch"] == "feature-x"
        assert data["sender"] == "alice"
        assert data["default_branch"] == "main"

    def test_tag_push_ignored(self, client):
        payload = {
            "changes": [{"ref": {"type": "TAG", "displayId": "v1.0"}}],
            "repository": {"slug": "myrepo", "project": {"key": "PROJ"}},
            "actor": {"slug": "alice"},
        }
        req = _make_request(payload, "repo:refs_changed")
        result = client.parse_webhook(req)
        assert result is None


# ------------------------------------------------------------------ #
#  parse_webhook — PR events                                          #
# ------------------------------------------------------------------ #

def _pr_payload():
    return {
        "pullRequest": {
            "id": 42,
            "title": "feat: add widget",
            "fromRef": {
                "displayId": "feature-branch",
                "latestCommit": "abc123",
                "repository": {"slug": "myrepo", "project": {"key": "PROJ"}},
            },
            "toRef": {
                "displayId": "main",
                "repository": {"slug": "myrepo", "project": {"key": "PROJ"}},
            },
            "links": {"self": [{"href": "https://bb.example.com/projects/PROJ/repos/myrepo/pull-requests/42"}]},
        },
        "actor": {"slug": "alice"},
    }


class TestParseWebhookPrOpened:
    def test_pr_opened(self, client):
        req = _make_request(_pr_payload(), "pr:opened")
        event_type, data = client.parse_webhook(req)
        assert event_type == "pr_opened"
        assert data["pr_number"] == 42
        assert data["pr_title"] == "feat: add widget"
        assert data["head_sha"] == "abc123"
        assert data["head_ref"] == "feature-branch"
        assert data["base_ref"] == "main"
        assert data["repo"] == "PROJ/myrepo"
        assert data["sender"] == "alice"


class TestParseWebhookPrUpdated:
    def test_pr_from_ref_updated(self, client):
        req = _make_request(_pr_payload(), "pr:from_ref_updated")
        event_type, data = client.parse_webhook(req)
        assert event_type == "pr_updated"
        assert data["pr_number"] == 42


class TestParseWebhookPrReopened:
    def test_pr_reopened(self, client):
        req = _make_request(_pr_payload(), "pr:reopened")
        event_type, data = client.parse_webhook(req)
        assert event_type == "pr_reopened"


class TestParseWebhookReviewRequested:
    def test_reviewer_updated(self, client):
        req = _make_request(_pr_payload(), "pr:reviewer:updated")
        event_type, data = client.parse_webhook(req)
        assert event_type == "review_requested"


# ------------------------------------------------------------------ #
#  parse_webhook — comments                                           #
# ------------------------------------------------------------------ #

class TestParseWebhookComment:
    def test_general_comment(self, client):
        payload = _pr_payload()
        payload["comment"] = {
            "id": 99,
            "text": "looks good",
            "author": {"slug": "bob"},
        }
        req = _make_request(payload, "pr:comment:added")
        event_type, data = client.parse_webhook(req)
        assert event_type == "comment"
        assert data["comment_body"] == "looks good"
        assert data["comment_user"] == "bob"
        assert data["comment_id"] == 99
        assert data["file_path"] is None
        assert data["line"] is None

    def test_diff_comment_with_anchor(self, client):
        payload = _pr_payload()
        payload["comment"] = {
            "id": 100,
            "text": "nit: rename this",
            "author": {"slug": "bob"},
            "anchor": {"path": "src/main.py", "line": 42},
        }
        req = _make_request(payload, "pr:comment:added")
        event_type, data = client.parse_webhook(req)
        assert event_type == "diff_comment"
        assert data["file_path"] == "src/main.py"
        assert data["line"] == 42
        assert data["comment_body"] == "nit: rename this"

    def test_threaded_reply_extracts_parent_comment_id(self, client):
        """BB DC sends commentParentId at the payload root for thread replies."""
        payload = _pr_payload()
        payload["commentParentId"] = 259147
        payload["comment"] = {
            "id": 259148,
            "text": "follow-up question",
            "author": {"slug": "bob"},
        }
        req = _make_request(payload, "pr:comment:added")
        _, data = client.parse_webhook(req)
        assert data["parent_comment_id"] == 259147

    def test_top_level_comment_has_no_parent(self, client):
        payload = _pr_payload()
        payload["comment"] = {
            "id": 99,
            "text": "top level",
            "author": {"slug": "bob"},
        }
        req = _make_request(payload, "pr:comment:added")
        _, data = client.parse_webhook(req)
        assert data["parent_comment_id"] is None

    def test_explicit_null_parent_does_not_crash(self, client):
        """If BB DC sends "parent": null inside the comment object, parsing
        must not raise AttributeError on (None).get("id")."""
        payload = _pr_payload()
        payload["comment"] = {
            "id": 99,
            "text": "top level",
            "author": {"slug": "bob"},
            "parent": None,
        }
        req = _make_request(payload, "pr:comment:added")
        _, data = client.parse_webhook(req)
        assert data["parent_comment_id"] is None


# ------------------------------------------------------------------ #
#  parse_webhook — unknown event                                      #
# ------------------------------------------------------------------ #

class TestParseWebhookUnknown:
    def test_unknown_event_returns_none(self, client):
        req = _make_request({}, "pr:deleted")
        result = client.parse_webhook(req)
        assert result is None


# ------------------------------------------------------------------ #
#  submit_review                                                      #
# ------------------------------------------------------------------ #

class TestSubmitReview:
    def test_approve_posts_comment_and_approves(self, client):
        comment_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        comment_resp.json.return_value = {"id": 10, "text": "LGTM"}
        approve_resp = MagicMock(status_code=200, raise_for_status=MagicMock())

        with patch.object(client.session, "post", side_effect=[comment_resp, approve_resp]) as mock_post:
            result = client.submit_review("PROJ/repo", 5, "LGTM", approve=True)

        assert result["id"] == 10
        # First call: comment, second call: approve
        calls = mock_post.call_args_list
        assert "/comments" in calls[0][0][0]
        assert calls[0][1]["json"]["text"] == "LGTM"
        assert "/approve" in calls[1][0][0]

    def test_reject_posts_comment_and_sets_needs_work(self, client):
        comment_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        comment_resp.json.return_value = {"id": 11, "text": "Issues found"}
        needs_work_resp = MagicMock(status_code=200, raise_for_status=MagicMock())

        with patch.object(client.session, "post", return_value=comment_resp), \
             patch.object(client.session, "put", return_value=needs_work_resp) as mock_put:
            result = client.submit_review("PROJ/repo", 5, "Issues found", approve=False)

        assert result["id"] == 11
        put_payload = mock_put.call_args[1]["json"]
        assert put_payload["status"] == "NEEDS_WORK"
        assert "/participants/raven-bot" in mock_put.call_args[0][0]

    def test_inline_comments_posted_before_review(self, client):
        inline_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        inline_resp.json.return_value = {}
        comment_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        comment_resp.json.return_value = {"id": 12}
        approve_resp = MagicMock(status_code=200, raise_for_status=MagicMock())

        with patch.object(client.session, "post", side_effect=[inline_resp, comment_resp, approve_resp]) as mock_post:
            client.submit_review(
                "PROJ/repo", 5, "Review body", approve=True,
                inline_comments=[{"file": "main.py", "line": 10, "body": "fix this"}],
            )

        # First call should be the inline comment with anchor
        inline_payload = mock_post.call_args_list[0][1]["json"]
        assert inline_payload["text"] == "fix this"
        assert inline_payload["anchor"]["path"] == "main.py"
        assert inline_payload["anchor"]["line"] == 10
        assert inline_payload["anchor"]["lineType"] == "ADDED"


# ------------------------------------------------------------------ #
#  dismiss_previous_reviews                                           #
# ------------------------------------------------------------------ #

class TestDismissPreviousReviews:
    def test_is_noop(self, client):
        """dismiss_previous_reviews is a no-op on BB DC — status is overwritten by submit_review."""
        with patch.object(client.session, "put") as mock_put, \
             patch.object(client.session, "get") as mock_get:
            client.dismiss_previous_reviews("PROJ/repo", 5, "raven-bot")
        mock_put.assert_not_called()
        mock_get.assert_not_called()


# ------------------------------------------------------------------ #
#  add_self_as_reviewer                                               #
# ------------------------------------------------------------------ #

class TestAddSelfAsReviewer:
    def test_posts_participant_with_reviewer_role(self, client):
        with _mock_post(client, status=201) as mock_post:
            client.add_self_as_reviewer("PROJ/repo", 5)
        url = mock_post.call_args[0][0]
        assert url.endswith("/projects/PROJ/repos/repo/pull-requests/5/participants")
        assert mock_post.call_args[1]["json"] == {
            "user": {"name": "raven-bot"},
            "role": "REVIEWER",
        }

    def test_200_accepted(self, client):
        with _mock_post(client, status=200):
            client.add_self_as_reviewer("PROJ/repo", 5)  # no raise

    def test_409_tolerated_as_idempotent_noop(self, client):
        mock_resp = MagicMock(status_code=409)
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "post", return_value=mock_resp):
            client.add_self_as_reviewer("PROJ/repo", 5)  # no raise

    def test_http_error_raises(self, client):
        mock_resp = MagicMock(status_code=500)
        mock_resp.raise_for_status.side_effect = Exception("500")
        with patch.object(client.session, "post", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                client.add_self_as_reviewer("PROJ/repo", 5)


# ------------------------------------------------------------------ #
#  merge_pr                                                           #
# ------------------------------------------------------------------ #

class TestMergePr:
    def test_merge_success(self, client):
        pr_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        pr_resp.json.return_value = {"version": 3}
        merge_resp = MagicMock(status_code=200)
        merge_resp.json.return_value = {}

        with patch.object(client.session, "get", return_value=pr_resp), \
             patch.object(client.session, "post", return_value=merge_resp) as mock_post:
            result = client.merge_pr("PROJ/repo", 7)

        assert result is True
        post_payload = mock_post.call_args[1]["json"]
        assert post_payload["deleteSourceBranch"] is True
        assert mock_post.call_args[1]["params"]["version"] == 3

    def test_merge_failure(self, client):
        pr_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        pr_resp.json.return_value = {"version": 1}
        merge_resp = MagicMock(status_code=409, text="Conflict")

        with patch.object(client.session, "get", return_value=pr_resp), \
             patch.object(client.session, "post", return_value=merge_resp):
            result = client.merge_pr("PROJ/repo", 7)

        assert result is False

    def test_merge_pr_fetch_fail(self, client):
        pr_resp = MagicMock(status_code=404)
        pr_resp.json.return_value = {}

        with patch.object(client.session, "get", return_value=pr_resp):
            result = client.merge_pr("PROJ/repo", 7)

        assert result is False


# ------------------------------------------------------------------ #
#  get_commit_status aggregation                                      #
# ------------------------------------------------------------------ #

class TestGetCommitStatus:
    def test_all_successful(self, client):
        data = {"values": [{"state": "SUCCESSFUL"}, {"state": "SUCCESSFUL"}]}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "success"

    def test_any_failed(self, client):
        data = {"values": [{"state": "SUCCESSFUL"}, {"state": "FAILED"}]}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "failure"

    def test_inprogress(self, client):
        data = {"values": [{"state": "SUCCESSFUL"}, {"state": "INPROGRESS"}]}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "pending"

    def test_no_statuses(self, client):
        data = {"values": []}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "none"

    def test_api_error_returns_pending(self, client):
        """API errors return 'pending' to avoid merging with incomplete CI info."""
        with _mock_get(client, status=500):
            assert client.get_commit_status("PROJ/repo", "abc123") == "pending"

    def test_cancelled_returns_failure(self, client):
        """CANCELLED builds should block merge, not be treated as absent CI."""
        data = {"values": [{"state": "SUCCESSFUL"}, {"state": "CANCELLED"}]}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "failure"

    def test_unknown_returns_pending(self, client):
        """UNKNOWN state should be treated as pending to avoid premature merge."""
        data = {"values": [{"state": "UNKNOWN"}]}
        with _mock_get(client, json_data=data):
            assert client.get_commit_status("PROJ/repo", "abc123") == "pending"

    def test_uses_build_status_url(self, client):
        data = {"values": [{"state": "SUCCESSFUL"}]}
        with _mock_get(client, json_data=data) as mock_get:
            client.get_commit_status("PROJ/repo", "abc123")
        url = mock_get.call_args[0][0]
        assert "/rest/build-status/latest/commits/abc123" in url


# ------------------------------------------------------------------ #
#  add_label_to_pr — no-op                                           #
# ------------------------------------------------------------------ #

class TestAddLabelToPr:
    def test_is_noop(self, client):
        # Should not make any HTTP calls
        with patch.object(client.session, "get") as mock_get, \
             patch.object(client.session, "post") as mock_post:
            client.add_label_to_pr("PROJ/repo", 5)
        mock_get.assert_not_called()
        mock_post.assert_not_called()


# ------------------------------------------------------------------ #
#  Diff fetching                                                      #
# ------------------------------------------------------------------ #

class TestFetchPrDiff:
    def test_success(self, client):
        diff_text = "diff --git a/bar b/bar\n+added line\n"
        with _mock_get(client, text=diff_text):
            result = client.fetch_pr_diff("PROJ/repo", 7)
        assert "diff --git" in result

    def test_empty_diff_raises(self, client):
        with _mock_get(client, text=""):
            with pytest.raises(RuntimeError, match="empty diff"):
                client.fetch_pr_diff("PROJ/repo", 7)

    def test_url_structure(self, client):
        with _mock_get(client, text="diff") as mock_get:
            client.fetch_pr_diff("PROJ/repo", 42)
        url = mock_get.call_args[0][0]
        assert "/projects/PROJ/repos/repo/pull-requests/42/diff" in url


# ------------------------------------------------------------------ #
#  File fetching                                                      #
# ------------------------------------------------------------------ #

class TestFetchFile:
    def test_success(self, client):
        data = {"lines": [{"text": "line 1"}, {"text": "line 2"}]}
        with _mock_get(client, json_data=data):
            result = client.fetch_file("PROJ/repo", "README.md", ref="main")
        assert result == "line 1\nline 2"

    def test_not_found(self, client):
        with _mock_get(client, status=404):
            result = client.fetch_file("PROJ/repo", "missing.md")
        assert result == ""


# ------------------------------------------------------------------ #
#  Post comment                                                       #
# ------------------------------------------------------------------ #

class TestPostPrComment:
    def test_uses_text_field(self, client):
        with _mock_post(client, json_data={"id": 55}) as mock_post:
            result = client.post_pr_comment("PROJ/repo", 3, "hello")
        payload = mock_post.call_args[1]["json"]
        assert payload["text"] == "hello"
        assert "body" not in payload
        assert "parent" not in payload
        assert result["id"] == 55

    def test_parent_comment_id_creates_thread_reply(self, client):
        with _mock_post(client, json_data={"id": 56}) as mock_post:
            client.post_pr_comment("PROJ/repo", 3, "reply", parent_comment_id=42)
        payload = mock_post.call_args[1]["json"]
        assert payload["text"] == "reply"
        assert payload["parent"] == {"id": 42}


class TestGetCommentThreadAuthors:
    def test_root_only(self, client):
        with _mock_get(client,
                       json_data={"id": 42, "author": {"slug": "alice"}}) as mock_get:
            authors = client.get_comment_thread_authors("PROJ/repo", 3, 42)
        assert authors == ["alice"]
        url = mock_get.call_args[0][0]
        assert url.endswith("/pull-requests/3/comments/42")

    def test_includes_child_replies(self, client):
        """Thread walk must pick up authors nested under comments[]."""
        with _mock_get(client, json_data={
            "id": 42,
            "author": {"slug": "alice"},
            "comments": [
                {"id": 43, "author": {"slug": "raven-bot"}},
                {"id": 44, "author": {"slug": "bob"}},
            ],
        }):
            authors = client.get_comment_thread_authors("PROJ/repo", 3, 42)
        assert set(authors) == {"alice", "raven-bot", "bob"}

    def test_returns_empty_on_404(self, client):
        mock_resp = MagicMock(status_code=404)
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_comment_thread_authors("PROJ/repo", 3, 999) == []

    def test_handles_nested_replies_defensively(self, client):
        """Some BB DC versions may nest comments beyond one level."""
        with _mock_get(client, json_data={
            "id": 1, "author": {"slug": "a"},
            "comments": [
                {"id": 2, "author": {"slug": "b"},
                 "comments": [{"id": 3, "author": {"slug": "c"}}]},
            ],
        }):
            authors = client.get_comment_thread_authors("PROJ/repo", 3, 1)
        assert set(authors) == {"a", "b", "c"}


# ------------------------------------------------------------------ #
#  PR head SHA                                                        #
# ------------------------------------------------------------------ #

class TestGetPrHeadSha:
    def test_returns_sha(self, client):
        data = {"fromRef": {"latestCommit": "deadbeef"}}
        with _mock_get(client, json_data=data):
            assert client.get_pr_head_sha("PROJ/repo", 7) == "deadbeef"

    def test_missing_sha_raises(self, client):
        with _mock_get(client, json_data={}):
            with pytest.raises(RuntimeError, match="empty head SHA"):
                client.get_pr_head_sha("PROJ/repo", 7)


# ------------------------------------------------------------------ #
#  _json_diff_to_unified                                              #
# ------------------------------------------------------------------ #

class TestJsonDiffToUnified:
    def test_modified_file(self, client):
        data = {
            "diffs": [{
                "source": {"toString": "src/app.py"},
                "destination": {"toString": "src/app.py"},
                "hunks": [{
                    "sourceLine": 10,
                    "sourceSpan": 5,
                    "destinationLine": 10,
                    "destinationSpan": 6,
                    "segments": [
                        {"type": "CONTEXT", "lines": [{"line": "import os"}]},
                        {"type": "REMOVED", "lines": [{"line": "old_line"}]},
                        {"type": "ADDED", "lines": [{"line": "new_line"}, {"line": "extra_line"}]},
                        {"type": "CONTEXT", "lines": [{"line": "pass"}]},
                    ],
                }],
            }],
        }
        result = client._json_diff_to_unified(data)
        assert "diff --git a/src/app.py b/src/app.py" in result
        assert "--- a/src/app.py" in result
        assert "+++ b/src/app.py" in result
        assert "@@ -10,5 +10,6 @@" in result
        assert " import os" in result
        assert "-old_line" in result
        assert "+new_line" in result
        assert "+extra_line" in result
        assert " pass" in result

    def test_new_file(self, client):
        data = {
            "diffs": [{
                "source": None,
                "destination": {"toString": "new_file.py"},
                "hunks": [{
                    "sourceLine": 0,
                    "sourceSpan": 0,
                    "destinationLine": 1,
                    "destinationSpan": 2,
                    "segments": [
                        {"type": "ADDED", "lines": [{"line": "hello"}, {"line": "world"}]},
                    ],
                }],
            }],
        }
        result = client._json_diff_to_unified(data)
        assert "--- /dev/null" in result
        assert "+++ b/new_file.py" in result
        assert "+hello" in result
        assert "+world" in result

    def test_deleted_file(self, client):
        data = {
            "diffs": [{
                "source": {"toString": "old_file.py"},
                "destination": None,
                "hunks": [{
                    "sourceLine": 1,
                    "sourceSpan": 2,
                    "destinationLine": 0,
                    "destinationSpan": 0,
                    "segments": [
                        {"type": "REMOVED", "lines": [{"line": "goodbye"}, {"line": "world"}]},
                    ],
                }],
            }],
        }
        result = client._json_diff_to_unified(data)
        assert "--- a/old_file.py" in result
        assert "+++ /dev/null" in result
        assert "-goodbye" in result
        assert "-world" in result

    def test_empty_diffs(self, client):
        result = client._json_diff_to_unified({"diffs": []})
        assert result == "\n"


# ------------------------------------------------------------------ #
#  get_pr_reviews — status normalization                              #
# ------------------------------------------------------------------ #

class TestGetPrReviews:
    def test_approved_maps_to_approved(self, client):
        data = {
            "values": [{"user": {"slug": "alice"}, "status": "APPROVED"}],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 1)
        assert len(reviews) == 1
        assert reviews[0]["user"]["login"] == "alice"
        assert reviews[0]["state"] == "APPROVED"

    def test_needs_work_maps_to_request_changes(self, client):
        data = {
            "values": [{"user": {"slug": "bob"}, "status": "NEEDS_WORK"}],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 1)
        assert reviews[0]["state"] == "REQUEST_CHANGES"

    def test_unapproved_maps_to_comment(self, client):
        data = {
            "values": [{"user": {"slug": "carol"}, "status": "UNAPPROVED"}],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 1)
        assert reviews[0]["state"] == "COMMENT"

    def test_mixed_statuses(self, client):
        data = {
            "values": [
                {"user": {"slug": "alice"}, "status": "APPROVED"},
                {"user": {"slug": "bob"}, "status": "NEEDS_WORK"},
                {"user": {"slug": "carol"}, "status": "UNAPPROVED"},
            ],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 1)
        assert len(reviews) == 3
        states = {r["user"]["login"]: r["state"] for r in reviews}
        assert states["alice"] == "APPROVED"
        assert states["bob"] == "REQUEST_CHANGES"
        assert states["carol"] == "COMMENT"


# ------------------------------------------------------------------ #
#  parse_webhook — review_requested with addedReviewers               #
# ------------------------------------------------------------------ #

class TestParseWebhookReviewRequestedWithReviewers:
    def test_extracts_added_reviewer_slug(self, client):
        payload = _pr_payload()
        payload["addedReviewers"] = [
            {"slug": "dave", "displayName": "Dave"},
            {"slug": "eve", "displayName": "Eve"},
        ]
        req = _make_request(payload, "pr:reviewer:updated")
        event_type, data = client.parse_webhook(req)
        assert event_type == "review_requested"
        assert data["requested_reviewer"] == "dave"
