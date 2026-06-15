"""Tests for providers/bitbucket_dc.py — BitbucketDCProvider API client."""

import hashlib
import hmac as hmac_mod
import json
import os
import pytest
from unittest.mock import MagicMock, patch


from raven.providers import DiffTruncatedError
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

    def test_added_emits_comment_version(self, client):
        """version=1 on a fresh comment; surfaced for the server's
        dedup key so subsequent edits (v=2, …) don't collide."""
        payload = _pr_payload()
        payload["comment"] = {
            "id": 99, "version": 1, "text": "first post",
            "author": {"slug": "bob"},
        }
        req = _make_request(payload, "pr:comment:added")
        _, data = client.parse_webhook(req)
        assert data["comment_version"] == 1

    def test_edited_event_routes_through_same_handler(self, client):
        """pr:comment:edited shares the payload shape with pr:comment:added.
        The version bump lets the server's dedup re-fire on the edit so a
        user adding @raven to an existing comment triggers a reply."""
        payload = _pr_payload()
        payload["comment"] = {
            "id": 99, "version": 2, "text": "second iteration with @raven",
            "author": {"slug": "bob"},
        }
        payload["previousComment"] = "second iteration"
        req = _make_request(payload, "pr:comment:edited")
        event_type, data = client.parse_webhook(req)
        assert event_type == "comment"
        assert data["comment_id"] == 99
        assert data["comment_version"] == 2
        assert data["comment_body"] == "second iteration with @raven"

    def test_edited_diff_comment_preserves_anchor(self, client):
        payload = _pr_payload()
        payload["comment"] = {
            "id": 100, "version": 3, "text": "edited inline note",
            "author": {"slug": "bob"},
            "anchor": {"path": "src/main.py", "line": 42},
        }
        req = _make_request(payload, "pr:comment:edited")
        event_type, data = client.parse_webhook(req)
        assert event_type == "diff_comment"
        assert data["file_path"] == "src/main.py"
        assert data["line"] == 42
        assert data["comment_version"] == 3

    def test_deleted_event_returns_none(self, client):
        """pr:comment:deleted is surfaced (INFO log) but not dispatched.
        The route exists so operators see explicit handling rather than
        an unrecognized-event silent drop."""
        payload = _pr_payload()
        payload["comment"] = {"id": 99, "text": "gone"}
        req = _make_request(payload, "pr:comment:deleted")
        assert client.parse_webhook(req) is None


# ------------------------------------------------------------------ #
#  parse_webhook — review-state events                                #
# ------------------------------------------------------------------ #

class TestParseWebhookReviewState:
    """Parity with Gitea's pull_request_review_approved / _rejected.
    The server's consumer is a no-op (route preserved so webhook
    deliveries get an explicit ignored response), but parity matters
    for operators debugging cross-provider behavior."""

    def test_reviewer_approved_maps_to_review_approved(self, client):
        req = _make_request(_pr_payload(), "pr:reviewer:approved")
        event_type, _ = client.parse_webhook(req)
        assert event_type == "review_approved"

    def test_reviewer_changes_requested_maps_to_review_rejected(self, client):
        req = _make_request(_pr_payload(), "pr:reviewer:changes_requested")
        event_type, _ = client.parse_webhook(req)
        assert event_type == "review_rejected"

    def test_reviewer_unapproved_returns_none(self, client):
        """No symmetric Gitea event + no behavior need → drop silently."""
        req = _make_request(_pr_payload(), "pr:reviewer:unapproved")
        assert client.parse_webhook(req) is None


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

    def test_main_comment_posted_before_inline(self, client):
        """The main review comment is posted FIRST, then inline anchors,
        then approve. Ordering matters: if the main comment POST fails,
        no inline comments will have been posted (no orphans/dupes on the
        re-post that the caller drives after a failed submit_review)."""
        main_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        main_resp.json.return_value = {"id": 12}
        inline_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        inline_resp.json.return_value = {"id": 77}
        approve_resp = MagicMock(status_code=200, raise_for_status=MagicMock())

        with patch.object(client.session, "post", side_effect=[main_resp, inline_resp, approve_resp]) as mock_post:
            client.submit_review(
                "PROJ/repo", 5, "Review body", approve=True,
                inline_comments=[{"file": "main.py", "line": 10, "body": "fix this"}],
            )

        calls = mock_post.call_args_list
        # First call: the main review comment (no anchor).
        first_payload = calls[0][1]["json"]
        assert first_payload["text"] == "Review body"
        assert "anchor" not in first_payload
        # Second call: the inline anchor comment.
        inline_payload = calls[1][1]["json"]
        assert inline_payload["text"] == "fix this"
        assert inline_payload["anchor"]["path"] == "main.py"
        assert inline_payload["anchor"]["line"] == 10
        assert inline_payload["anchor"]["lineType"] == "ADDED"
        # Third call: approve.
        assert "/approve" in calls[2][0][0]

    def test_main_comment_failure_posts_no_inline(self, client):
        """If the MAIN review-comment POST raises, no inline anchors are
        posted — _post_inline_comments is never reached. This guards the
        orphaned/duplicated-comment window (audit #14): the caller treats
        the whole submit_review as failed and re-posts on the next pass,
        so any already-posted inline comments would be duplicated."""
        import requests
        main_resp = MagicMock(status_code=500)
        main_resp.raise_for_status.side_effect = requests.HTTPError("boom")

        with patch.object(client.session, "post", return_value=main_resp), \
             patch.object(client, "_post_inline_comments") as mock_inline:
            with pytest.raises(requests.HTTPError):
                client.submit_review(
                    "PROJ/repo", 5, "Review body", approve=True,
                    inline_comments=[{"file": "main.py", "line": 10, "body": "fix this"}],
                )

        # The main-comment POST failed before any inline anchor was posted.
        mock_inline.assert_not_called()

    def test_empty_body_skips_main_comment(self, client):
        """An empty body (RAVEN_REVIEW_OUTPUT=inline with everything anchored
        inline) must NOT post an empty main comment — BB DC rejects empty
        comment text. Inline anchors and the verdict still post, and the
        return still carries the aligned inline_comments list."""
        inline_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        inline_resp.json.return_value = {"id": 77}
        needs_work_resp = MagicMock(status_code=200, raise_for_status=MagicMock())

        with patch.object(client.session, "post", return_value=inline_resp) as mock_post, \
             patch.object(client.session, "put", return_value=needs_work_resp) as mock_put:
            result = client.submit_review(
                "PROJ/repo", 5, "", approve=False,
                inline_comments=[{"file": "main.py", "line": 10, "body": "fix"}],
            )

        # Every POST that ran is an inline anchor — the empty main comment
        # was skipped (no POST with bare text / no anchor).
        assert mock_post.call_count >= 1
        for call in mock_post.call_args_list:
            assert call[1]["json"].get("anchor") is not None
        # The needs-work verdict is still recorded.
        assert mock_put.call_args[1]["json"]["status"] == "NEEDS_WORK"
        # Return contract preserved: aligned inline_comments list.
        assert len(result["inline_comments"]) == 1
        assert result["inline_comments"][0]["comment_id"] == 77

    def test_happy_path_returns_aligned_inline_comments(self, client):
        """Reorder preserves the return contract: the result is the
        main-comment dict extended with an `inline_comments` list aligned
        to the input (same length, with comment_ids for posted anchors)."""
        inline = [
            {"file": "a.py", "line": 5, "body": "f1"},
            {"file": "b.py", "line": 8, "body": "f2"},
        ]
        posts = []
        def fake_post(url, json=None, timeout=None):
            posts.append((url, json))
            mr = MagicMock(status_code=201, raise_for_status=MagicMock())
            if "anchor" in (json or {}):
                # Anchor posts return ids 200, 201, ... by order seen.
                cid = 200 + sum(1 for p in posts[:-1] if "anchor" in (p[1] or {}))
                mr.json.return_value = {"id": cid}
            elif "/approve" in url:
                mr.json.return_value = {}
            else:
                mr.json.return_value = {"id": 12, "text": "Review body"}
            return mr

        with patch.object(client.session, "post", side_effect=fake_post):
            result = client.submit_review(
                "PROJ/repo", 5, "Review body", approve=True,
                inline_comments=inline,
            )

        # Main-comment dict is the base of the return value.
        assert result["id"] == 12
        ic = result["inline_comments"]
        assert len(ic) == 2  # aligned to input length
        assert ic[0] == {"file": "a.py", "line": 5, "comment_id": 200}
        assert ic[1] == {"file": "b.py", "line": 8, "comment_id": 201}

    def test_comment_only_skips_participant_status(self, client):
        """comment_only=True posts the body comment but skips the
        approve / needs-work participant PUT — keeps the review
        non-blocking. Used for advisory mode."""
        comment_resp = MagicMock(status_code=201, raise_for_status=MagicMock())
        comment_resp.json.return_value = {"id": 99, "text": "advisory"}

        with patch.object(client.session, "post", return_value=comment_resp) as mock_post, \
             patch.object(client.session, "put") as mock_put:
            client.submit_review(
                "PROJ/repo", 7, "advisory body",
                approve=False, comment_only=True,
            )

        # Comment posted; participants endpoint NOT called.
        assert mock_post.called
        mock_put.assert_not_called()
        # And no /approve POST either.
        approve_calls = [c for c in mock_post.call_args_list if "/approve" in c[0][0]]
        assert not approve_calls


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

    def test_merge_refused_on_head_sha_mismatch(self, client):
        """Force-push between review and merge: latestCommit != reviewed head_sha -> no merge POST."""
        pr_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        pr_resp.json.return_value = {
            "version": 3,
            "fromRef": {"latestCommit": "attacker999"},
        }

        with patch.object(client.session, "get", return_value=pr_resp), \
             patch.object(client.session, "post") as mock_post:
            result = client.merge_pr("PROJ/repo", 7, head_sha="reviewed123")

        assert result is False
        mock_post.assert_not_called()

    def test_merge_proceeds_on_head_sha_match(self, client):
        pr_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        pr_resp.json.return_value = {
            "version": 3,
            "fromRef": {"latestCommit": "reviewed123"},
        }
        merge_resp = MagicMock(status_code=200)
        merge_resp.json.return_value = {}

        with patch.object(client.session, "get", return_value=pr_resp), \
             patch.object(client.session, "post", return_value=merge_resp) as mock_post:
            result = client.merge_pr("PROJ/repo", 7, head_sha="reviewed123")

        assert result is True
        mock_post.assert_called_once()
        assert mock_post.call_args[1]["params"]["version"] == 3

    def test_merge_without_head_sha_skips_comparison(self, client):
        """Legacy behavior: empty head_sha merges without pinning, even if latestCommit differs."""
        pr_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
        pr_resp.json.return_value = {
            "version": 5,
            "fromRef": {"latestCommit": "whatever456"},
        }
        merge_resp = MagicMock(status_code=200)
        merge_resp.json.return_value = {}

        with patch.object(client.session, "get", return_value=pr_resp), \
             patch.object(client.session, "post", return_value=merge_resp) as mock_post:
            result = client.merge_pr("PROJ/repo", 7)

        assert result is True
        mock_post.assert_called_once()


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

    # ── Truncated-diff fail-closed (audit 2026-06-13 finding #1) ──────── #
    # BB DC caps diff size and flags truncation on the overall response and
    # on individual file diffs / hunks / segments / lines. A truncated diff
    # means the model would see only part of the PR; reviewing (and possibly
    # APPROVING + auto-merging) that partial diff is the silent-unseen-code
    # hole. fetch_pr_diff must refuse it at every truncation level.

    def test_truncated_top_level_raises(self, client):
        data = {
            "diffs": [{"source": {"toString": "a.py"},
                       "destination": {"toString": "a.py"}, "hunks": []}],
            "truncated": True,
        }
        with _mock_get(client, json_data=data, content_type="application/json"):
            with pytest.raises(DiffTruncatedError, match="truncated"):
                client.fetch_pr_diff("PROJ/repo", 7)

    def test_truncated_per_file_raises(self, client):
        data = {"diffs": [{"source": {"toString": "a.py"},
                           "destination": {"toString": "a.py"},
                           "truncated": True, "hunks": []}]}
        with _mock_get(client, json_data=data, content_type="application/json"):
            with pytest.raises(DiffTruncatedError):
                client.fetch_pr_diff("PROJ/repo", 7)

    def test_truncated_hunk_raises(self, client):
        data = {"diffs": [{"source": {"toString": "a.py"},
                           "destination": {"toString": "a.py"},
                           "hunks": [{"truncated": True, "segments": []}]}]}
        with _mock_get(client, json_data=data, content_type="application/json"):
            with pytest.raises(DiffTruncatedError):
                client.fetch_pr_diff("PROJ/repo", 7)

    def test_truncated_segment_raises(self, client):
        data = {"diffs": [{"source": {"toString": "a.py"},
                           "destination": {"toString": "a.py"},
                           "hunks": [{"segments": [{"truncated": True,
                                                    "lines": []}]}]}]}
        with _mock_get(client, json_data=data, content_type="application/json"):
            with pytest.raises(DiffTruncatedError):
                client.fetch_pr_diff("PROJ/repo", 7)

    def test_non_truncated_json_diff_succeeds(self, client):
        """Guard against false positives: a complete JSON diff (all
        truncated flags false/absent) converts and returns normally."""
        data = {
            "diffs": [{"source": {"toString": "a.py"},
                       "destination": {"toString": "a.py"},
                       "truncated": False,
                       "hunks": [{"sourceLine": 1, "sourceSpan": 1,
                                  "destinationLine": 1, "destinationSpan": 2,
                                  "segments": [{"type": "ADDED",
                                                "lines": [{"line": "x"}]}]}]}],
            "truncated": False,
        }
        with _mock_get(client, json_data=data, content_type="application/json"):
            result = client.fetch_pr_diff("PROJ/repo", 7)
        assert "diff --git a/a.py b/a.py" in result


# ------------------------------------------------------------------ #
#  File fetching                                                      #
# ------------------------------------------------------------------ #

class TestFetchFile:
    def test_success(self, client):
        data = {"lines": [{"text": "line 1"}, {"text": "line 2"}]}
        with _mock_get(client, json_data=data):
            result = client.fetch_file("PROJ/repo", "README.md", ref="main")
        # Trailing newline for parity with Gitea.
        assert result == "line 1\nline 2\n"

    def test_not_found(self, client):
        with _mock_get(client, status=404):
            result = client.fetch_file("PROJ/repo", "missing.md")
        assert result == ""

    def test_empty_file_returns_empty_string_not_newline(self, client):
        """An empty lines array returns '' rather than a bare newline
        so callers can truthiness-check the result."""
        with _mock_get(client, json_data={"lines": []}):
            result = client.fetch_file("PROJ/repo", "empty.txt")
        assert result == ""

    def test_path_with_special_chars_is_url_encoded(self, client):
        """URL-sensitive characters in the path must be percent-encoded
        so they don't break routing."""
        with _mock_get(client, json_data={"lines": [{"text": "x"}]}) as mock_get:
            client.fetch_file("PROJ/repo", "docs/issue #42.md")
        url = mock_get.call_args[0][0]
        assert "#" not in url
        assert "docs/issue%20%2342.md" in url


class TestListDirectory:
    def test_returns_file_paths(self, client):
        data = {
            "type": "DIRECTORY",
            "children": {
                "values": [
                    {"type": "FILE", "path": {"toString": "security.md"}},
                    {"type": "FILE", "path": {"toString": "style.md"}},
                    {"type": "DIRECTORY", "path": {"toString": "nested"}},
                ],
                "isLastPage": True,
            },
        }
        with _mock_get(client, json_data=data):
            result = client.list_directory("PROJ/repo", ".claude/rules", ref="abc")
        # Only FILE entries, with full repo-rooted paths
        assert result == [".claude/rules/security.md", ".claude/rules/style.md"]

    def test_missing_directory_returns_empty(self, client):
        with _mock_get(client, status=404):
            assert client.list_directory("PROJ/repo", ".claude/rules") == []

    def test_http_error_returns_empty(self, client):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.list_directory("PROJ/repo", ".claude/rules") == []

    def test_passes_ref_to_api(self, client):
        with _mock_get(client, json_data={"children": {"values": [], "isLastPage": True}}) as mock_get:
            client.list_directory("PROJ/repo", ".claude/rules", ref="feature-sha")
        assert mock_get.call_args[1]["params"]["at"] == "feature-sha"

    def test_rejects_suspicious_child_names(self, client):
        """BB DC's contract for a directory listing is that each child's
        path component is just the filename. If a response includes a
        ``/``, ``\\``, ``.``, or ``..`` in the child name (hostile or
        upstream bug), drop it rather than stitch it into a weird path
        that flows into fetch_file."""
        data = {
            "type": "DIRECTORY",
            "children": {
                "values": [
                    {"type": "FILE", "path": {"toString": "good.md"}},
                    {"type": "FILE", "path": {"toString": "../../etc/passwd"}},
                    {"type": "FILE", "path": {"toString": "nested/escaped.md"}},
                    {"type": "FILE", "path": {"toString": "."}},
                    {"type": "FILE", "path": {"toString": ".."}},
                ],
                "isLastPage": True,
            },
        }
        with _mock_get(client, json_data=data):
            result = client.list_directory("PROJ/repo", ".claude/rules")
        assert result == [".claude/rules/good.md"]

    def test_pagination(self, client):
        """BB DC paginates via isLastPage / nextPageStart — walk pages
        until isLastPage is True."""
        page1 = {
            "children": {
                "values": [{"type": "FILE", "path": {"toString": "a.md"}}],
                "isLastPage": False,
                "nextPageStart": 1,
            }
        }
        page2 = {
            "children": {
                "values": [{"type": "FILE", "path": {"toString": "b.md"}}],
                "isLastPage": True,
            }
        }
        resp1 = MagicMock(status_code=200, headers={})
        resp1.json.return_value = page1
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock(status_code=200, headers={})
        resp2.json.return_value = page2
        resp2.raise_for_status = MagicMock()
        with patch.object(client.session, "get", side_effect=[resp1, resp2]):
            result = client.list_directory("PROJ/repo", ".claude/rules")
        assert result == [".claude/rules/a.md", ".claude/rules/b.md"]


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
#  PR base ref                                                        #
# ------------------------------------------------------------------ #

class TestGetPrBaseRef:
    def test_returns_base_ref(self, client):
        data = {"toRef": {"displayId": "main"}}
        with _mock_get(client, json_data=data):
            assert client.get_pr_base_ref("PROJ/repo", 7) == "main"

    def test_url_contains_pr_number(self, client):
        data = {"toRef": {"displayId": "main"}}
        with _mock_get(client, json_data=data) as mock_get:
            client.get_pr_base_ref("PROJ/repo", 42)
        url = mock_get.call_args[0][0]
        assert "/pull-requests/42" in url

    def test_missing_base_ref_raises(self, client):
        with _mock_get(client, json_data={}):
            with pytest.raises(RuntimeError, match="empty base ref"):
                client.get_pr_base_ref("PROJ/repo", 7)


class TestGetPrDescription:
    def test_returns_description(self, client):
        with _mock_get(client, json_data={"description": "Fixes DEV-123\n\nApproach: ..."}):
            assert client.get_pr_description("PROJ/repo", 7) == "Fixes DEV-123\n\nApproach: ..."

    def test_missing_description_returns_empty_string(self, client):
        with _mock_get(client, json_data={}):
            assert client.get_pr_description("PROJ/repo", 7) == ""

    def test_null_description_returns_empty_string(self, client):
        with _mock_get(client, json_data={"description": None}):
            assert client.get_pr_description("PROJ/repo", 7) == ""

    def test_http_error_returns_empty_string(self, client):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_pr_description("PROJ/repo", 7) == ""


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

    def test_unapproved_is_dropped(self, client):
        """UNAPPROVED is the default status for the PR author and any
        participant who hasn't formally reviewed. They are NOT reviewers
        in the Gitea sense the auto-merge gate is built around, so drop
        them rather than synthesising a fake COMMENT-state review."""
        data = {
            "values": [{"user": {"slug": "carol"}, "status": "UNAPPROVED"}],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 1)
        assert reviews == []

    def test_mixed_statuses(self, client):
        """APPROVED + NEEDS_WORK come through; UNAPPROVED is filtered."""
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
        assert len(reviews) == 2
        states = {r["user"]["login"]: r["state"] for r in reviews}
        assert states == {"alice": "APPROVED", "bob": "REQUEST_CHANGES"}

    def test_author_excluded_when_unapproved(self, client):
        """Regression guard: BB DC's /participants endpoint always
        includes the PR author with status=UNAPPROVED. Before this filter,
        the auto-merge gate at server.py:781 would see the author in
        get_pr_reviews and bail with 'has other reviewers — leaving open
        for human review' even when Raven was the only formal reviewer."""
        data = {
            "values": [
                {"user": {"slug": "marcin.zasina"}, "status": "UNAPPROVED",
                 "role": "AUTHOR"},
                {"user": {"slug": "raven"}, "status": "APPROVED",
                 "role": "REVIEWER"},
            ],
            "isLastPage": True,
        }
        with _mock_get(client, json_data=data):
            reviews = client.get_pr_reviews("PROJ/repo", 15)
        # Author dropped, Raven kept — the auto-merge gate now sees Raven
        # as the sole reviewer and can proceed.
        assert len(reviews) == 1
        assert reviews[0]["user"]["login"] == "raven"
        assert reviews[0]["state"] == "APPROVED"


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


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_pr_state                       #
# ------------------------------------------------------------------ #

class TestGetPrState:
    def test_open(self, client):
        with _mock_get(client, status=200, json_data={"state": "OPEN"}):
            assert client.get_pr_state("proj/repo", 1) == "open"

    def test_merged(self, client):
        with _mock_get(client, status=200, json_data={"state": "MERGED"}):
            assert client.get_pr_state("proj/repo", 1) == "merged"

    def test_declined_maps_to_closed(self, client):
        with _mock_get(client, status=200, json_data={"state": "DECLINED"}):
            assert client.get_pr_state("proj/repo", 1) == "closed"


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_pr_metadata                    #
# ------------------------------------------------------------------ #

class TestGetPrMetadata:
    def test_returns_title_and_url(self, client):
        with _mock_get(client, status=200, json_data={
            "title": "Add X",
            "links": {"self": [{"href": "https://bb/u/r/pulls/1"}]},
        }):
            meta = client.get_pr_metadata("proj/repo", 1)
        assert meta == {"title": "Add X", "html_url": "https://bb/u/r/pulls/1"}

    def test_returns_empty_dict_on_http_error(self, client):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("boom")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_pr_metadata("proj/repo", 1) == {}


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: get_comment_thread                 #
# ------------------------------------------------------------------ #

class TestGetCommentThread:
    """``get_comment_thread`` finds the true thread root via the
    activities endpoint, since BB DC's ``GET /comments/{id}`` response
    shape has no parent field. Tests mock the activities endpoint."""

    def _activities_response(self, threads, is_last=True, next_start=None):
        """Build a paginated activities response wrapping the given
        top-level threads as ``action=COMMENTED`` activities."""
        return {
            "values": [{"action": "COMMENTED", "comment": t} for t in threads],
            "isLastPage": is_last,
            "nextPageStart": next_start or 0,
        }

    def test_walks_nested_with_resolved_flag(self, client):
        """Seed is the top-level thread root; full tree is emitted.

        Resolved flag must catch both signals BB DC emits: the
        ``threadResolved`` boolean the UI's "Resolve thread" button
        sets, and the legacy ``state == "RESOLVED"`` enum for resolved
        tasks (severity=BLOCKER). Mocks one of each so a future
        regression on either path fails this test."""
        root_thread = {
            "id": 100, "author": {"slug": "raven"}, "text": "root",
            "state": "OPEN", "threadResolved": False,
            "anchor": {"path": "a.py", "line": 5},
            "comments": [
                {"id": 101, "author": {"slug": "alice"}, "text": "reply1",
                 "state": "OPEN", "threadResolved": False,
                 "comments": [
                    # threadResolved=True — what "Resolve thread" sets.
                    {"id": 102, "author": {"slug": "bob"}, "text": "reply2",
                     "state": "OPEN", "threadResolved": True, "comments": []},
                 ]},
                # Legacy: a resolved task (severity=BLOCKER) sets state.
                {"id": 103, "author": {"slug": "carol"}, "text": "reply3",
                 "state": "RESOLVED", "threadResolved": False, "comments": []},
            ],
        }
        with _mock_get(client, status=200,
                       json_data=self._activities_response([root_thread])):
            thread = client.get_comment_thread("proj/repo", 1, 100)
        ids = [c["id"] for c in thread]
        assert ids == [100, 101, 102, 103]
        assert thread[0]["parent_id"] is None
        assert thread[1]["parent_id"] == 100
        assert thread[2]["parent_id"] == 101
        assert thread[0]["file_path"] == "a.py"
        assert thread[0]["line"] == 5
        assert thread[0]["resolved"] is False
        assert thread[1]["resolved"] is False
        assert thread[2]["resolved"] is True   # threadResolved=True
        assert thread[3]["resolved"] is True   # state=RESOLVED (legacy task)

    def test_returns_empty_when_seed_not_in_activities(self, client):
        """Seed id absent from all activities → empty thread."""
        with _mock_get(client, status=200,
                       json_data=self._activities_response([
                           {"id": 1, "author": {"slug": "a"}, "text": "other thread",
                            "state": "OPEN", "comments": []},
                       ])):
            assert client.get_comment_thread("proj/repo", 1, 999) == []

    def test_returns_empty_on_activities_http_error(self, client):
        """Activities endpoint returns 500 → swallow and return []."""
        from requests import HTTPError
        mock_resp = MagicMock(status_code=500)
        mock_resp.raise_for_status.side_effect = HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_comment_thread("proj/repo", 1, 999) == []

    def test_finds_root_from_deep_reply_seed(self, client):
        """The seed (101) is a reply nested under the real root (99).
        Webhook-provided parent_comment_id may be any node in the tree;
        we must still emit the thread starting from the TRUE root."""
        root_thread = {
            "id": 99, "author": {"slug": "raven"}, "text": "the finding",
            "state": "OPEN", "anchor": {"path": "a.py", "line": 5},
            "comments": [
                {"id": 100, "author": {"slug": "alice"}, "text": "mid",
                 "state": "OPEN", "comments": [
                    {"id": 101, "author": {"slug": "bob"}, "text": "leaf",
                     "state": "OPEN", "comments": []},
                 ]},
            ],
        }
        # Seed is 101 (deep reply); other top-level threads in the PR
        # are unrelated but must be scanned past.
        unrelated = {
            "id": 50, "author": {"slug": "x"}, "text": "unrelated",
            "state": "OPEN", "comments": [],
        }
        with _mock_get(client, status=200,
                       json_data=self._activities_response([unrelated, root_thread])):
            thread = client.get_comment_thread("proj/repo", 1, 101)
        ids = [c["id"] for c in thread]
        # Starts at the TRUE root (99), not the webhook seed (101).
        assert ids == [99, 100, 101]
        # parent_id chain links back to root.
        assert thread[0]["parent_id"] is None
        assert thread[1]["parent_id"] == 99
        assert thread[2]["parent_id"] == 100

    def test_paginates_activities_until_found(self, client):
        """Activities is paginated; scan past empty / unrelated pages."""
        page1 = self._activities_response(
            [{"id": 1, "author": {"slug": "a"}, "text": "p1", "comments": []}],
            is_last=False, next_start=50,
        )
        page2 = self._activities_response(
            [{"id": 100, "author": {"slug": "raven"}, "text": "root",
              "state": "OPEN", "anchor": {"path": "a.py", "line": 5},
              "comments": [
                {"id": 101, "author": {"slug": "x"}, "text": "reply",
                 "state": "OPEN", "comments": []},
              ]}],
            is_last=True,
        )
        pages = [page1, page2]
        def fake_get(url, params=None, timeout=None):
            start = (params or {}).get("start", 0)
            mr = MagicMock(status_code=200)
            mr.raise_for_status = MagicMock()
            mr.json.return_value = pages[0] if start == 0 else pages[1]
            return mr
        with patch.object(client.session, "get", side_effect=fake_get):
            thread = client.get_comment_thread("proj/repo", 1, 101)
        ids = [c["id"] for c in thread]
        assert ids == [100, 101]

    def test_pagination_cap_logs_warning_when_more_pages_pending(self, client, caplog):
        """If the activities scan exhausts _ACTIVITIES_MAX_PAGES (30) while
        the server still reports isLastPage=false, log a WARNING so
        operators can spot pathological PRs that may drop state."""
        from raven.providers.bitbucket_dc import _ACTIVITIES_MAX_PAGES
        # Each page is "not last" with one unrelated comment.
        def fake_get(url, params=None, timeout=None):
            mr = MagicMock(status_code=200)
            mr.raise_for_status = MagicMock()
            mr.json.return_value = self._activities_response(
                [{"id": 9999, "author": {"slug": "a"}, "text": "unrelated", "comments": []}],
                is_last=False, next_start=(params or {}).get("start", 0) + 50,
            )
            return mr
        with caplog.at_level("WARNING", logger="raven.providers.bitbucket_dc"), \
             patch.object(client.session, "get", side_effect=fake_get):
            result = client.get_comment_thread("proj/repo", 1, 12345)
        assert result == []  # seed never found
        assert any(
            "activities scan capped" in r.message and str(_ACTIVITIES_MAX_PAGES) in r.message
            for r in caplog.records
        )


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: retract_finding                    #
# ------------------------------------------------------------------ #

class TestRetractFinding:
    def test_resolves_comment(self, client):
        """The PUT body must use threadResolved (writable boolean on
        RestComment per BB DC's OpenAPI spec). The state=RESOLVED field
        is task-specific and returns 400 on regular comments; only
        threadResolved actually marks the thread resolved in the UI."""
        get_resp = MagicMock(status_code=200)
        get_resp.json.return_value = {"id": 42, "version": 3}
        get_resp.raise_for_status = MagicMock()
        put_resp = MagicMock(status_code=200)
        put_resp.raise_for_status = MagicMock()
        with patch.object(client.session, "get", return_value=get_resp), \
             patch.object(client.session, "put", return_value=put_resp) as mock_put:
            assert client.retract_finding("proj/repo", 1, 42) is True
        _, kwargs = mock_put.call_args
        assert kwargs["json"] == {"threadResolved": True, "version": 3}

    def test_404_returns_false(self, client):
        get_resp = MagicMock(status_code=404)
        with patch.object(client.session, "get", return_value=get_resp):
            assert client.retract_finding("proj/repo", 1, 42) is False

    def test_409_version_conflict_returns_false(self, client):
        get_resp = MagicMock(status_code=200)
        get_resp.json.return_value = {"id": 42, "version": 3}
        get_resp.raise_for_status = MagicMock()
        put_resp = MagicMock(status_code=409)
        with patch.object(client.session, "get", return_value=get_resp), \
             patch.object(client.session, "put", return_value=put_resp):
            assert client.retract_finding("proj/repo", 1, 42) is False

# ------------------------------------------------------------------ #
#  Comment-thread-context: submit_review returns per-inline IDs       #
# ------------------------------------------------------------------ #

class TestSubmitReviewInlineIds:
    def test_returns_per_anchor_comment_ids(self, client):
        """Each inline anchor's POST response id is captured and aligned
        with the input. Filtered entries (invalid file/line) get None."""
        # 4 inputs: 2 valid, 1 missing file, 1 invalid line. Valid ones
        # get distinct comment_ids; filtered ones get None.
        inline = [
            {"file": "a.py", "line": 5, "body": "finding 1"},
            {"file": "", "line": 10, "body": "no file"},        # filtered
            {"file": "b.py", "line": 20, "body": "finding 2"},
            {"file": "c.py", "line": 0, "body": "invalid line"},  # filtered
        ]
        posts = []
        def fake_post(url, json=None, timeout=None):
            posts.append((url, json))
            mr = MagicMock(status_code=201)
            mr.raise_for_status = MagicMock()
            if "/approve" in url or "/participants/" in url:
                mr.json.return_value = {}
            elif "anchor" in (json or {}):
                # Each inline post returns a unique id, 100 + post index
                cid = 100 + sum(1 for p in posts[:-1] if "anchor" in (p[1] or {}))
                mr.json.return_value = {"id": cid}
            else:
                mr.json.return_value = {"id": 999}  # main review comment
            return mr
        with patch.object(client.session, "post", side_effect=fake_post):
            result = client.submit_review(
                "proj/repo", 1, "review body", approve=True,
                inline_comments=inline,
            )
        assert "inline_comments" in result
        ic = result["inline_comments"]
        assert len(ic) == 4
        assert ic[0]["comment_id"] == 100
        assert ic[1]["comment_id"] is None  # filtered (no file)
        assert ic[2]["comment_id"] == 101
        assert ic[3]["comment_id"] is None  # filtered (invalid line)

    def test_returns_none_id_when_post_fails(self, client):
        """A 4xx on one anchor doesn't break the others; the failed slot
        gets comment_id=None so retraction-by-id won't target the wrong
        comment via off-by-one alignment."""
        import requests
        inline = [
            {"file": "a.py", "line": 1, "body": "f1"},
            {"file": "a.py", "line": 2, "body": "f2"},
        ]
        anchor_calls = [0]
        def fake_post(url, json=None, timeout=None):
            mr = MagicMock()
            mr.raise_for_status = MagicMock()
            if "anchor" in (json or {}):
                # Key success/failure off the anchor index (not the
                # global call count) so the test is robust to where the
                # main-comment POST lands in the sequence. First anchor
                # succeeds; second fails.
                anchor_calls[0] += 1
                if anchor_calls[0] == 1:
                    mr.status_code = 200
                    mr.json.return_value = {"id": 500}
                else:
                    mr.status_code = 400
                    mr.raise_for_status.side_effect = requests.HTTPError("nope")
            elif "/approve" in url:
                mr.status_code = 200
                mr.json.return_value = {}
            else:
                mr.status_code = 201
                mr.json.return_value = {"id": 999}
            return mr
        with patch.object(client.session, "post", side_effect=fake_post):
            result = client.submit_review(
                "proj/repo", 1, "body", approve=True, inline_comments=inline,
            )
        ic = result["inline_comments"]
        assert ic[0]["comment_id"] == 500
        assert ic[1]["comment_id"] is None


# ------------------------------------------------------------------ #
#  User-resolved-findings filter: get_resolved_comment_ids            #
# ------------------------------------------------------------------ #

class TestGetResolvedCommentIds:
    """``get_resolved_comment_ids`` returns the set of top-level thread
    root IDs the developer has marked resolved on the PR. Both
    ``threadResolved=true`` (UI's "Resolve thread" button per the
    RestComment OpenAPI spec) and the legacy ``state="RESOLVED"`` task
    enum count. Used by ``_process_pr``'s carry-forward filter to drop
    user-resolved findings from the consolidated verdict."""

    def _activities_response(self, threads, is_last=True, next_start=None):
        return {
            "values": [{"action": "COMMENTED", "comment": t} for t in threads],
            "isLastPage": is_last,
            "nextPageStart": next_start or 0,
        }

    def test_collects_threadresolved_and_state_resolved(self, client):
        """Both signals are collected. threadResolved is the modern
        path the UI uses; state=RESOLVED is the legacy task-resolved
        path. A non-resolved thread is NOT in the set."""
        threads = [
            # Thread-resolved via UI (modern path)
            {"id": 100, "threadResolved": True, "state": "OPEN", "comments": []},
            # Resolved task (legacy path)
            {"id": 200, "threadResolved": False, "state": "RESOLVED", "comments": []},
            # Open thread — excluded
            {"id": 300, "threadResolved": False, "state": "OPEN", "comments": []},
        ]
        with _mock_get(client, status=200,
                       json_data=self._activities_response(threads)):
            resolved = client.get_resolved_comment_ids("proj/repo", 1)
        assert resolved == {100, 200}

    def test_returns_empty_when_no_activities(self, client):
        with _mock_get(client, status=200,
                       json_data=self._activities_response([])):
            assert client.get_resolved_comment_ids("proj/repo", 1) == set()

    def test_returns_empty_on_http_error(self, client):
        """Transient API failure → empty set, no raise. The review
        proceeds without filtering rather than blocking."""
        from requests import HTTPError
        mock_resp = MagicMock(status_code=500)
        mock_resp.raise_for_status.side_effect = HTTPError("500")
        with patch.object(client.session, "get", return_value=mock_resp):
            assert client.get_resolved_comment_ids("proj/repo", 1) == set()

    def test_only_collects_top_level_thread_roots(self, client):
        """Raven posts inline review comments as top-level threads, so
        their ``comment_id`` is always the thread root. We only collect
        from top-level — nested replies are ignored even if marked
        resolved on their own (which BB DC doesn't allow anyway)."""
        threads = [
            {"id": 100, "threadResolved": True, "state": "OPEN",
             "comments": [
                # A nested resolved-state comment — must NOT leak into the set
                {"id": 101, "threadResolved": True, "state": "RESOLVED", "comments": []},
             ]},
        ]
        with _mock_get(client, status=200,
                       json_data=self._activities_response(threads)):
            resolved = client.get_resolved_comment_ids("proj/repo", 1)
        assert resolved == {100}

    def test_paginates_activities(self, client):
        page1 = self._activities_response(
            [{"id": 10, "threadResolved": True, "state": "OPEN", "comments": []}],
            is_last=False, next_start=50,
        )
        page2 = self._activities_response(
            [{"id": 20, "threadResolved": True, "state": "OPEN", "comments": []}],
            is_last=True,
        )
        pages = [page1, page2]
        def fake_get(url, params=None, timeout=None):
            start = (params or {}).get("start", 0)
            mr = MagicMock(status_code=200)
            mr.raise_for_status = MagicMock()
            mr.json.return_value = pages[0] if start == 0 else pages[1]
            return mr
        with patch.object(client.session, "get", side_effect=fake_get):
            resolved = client.get_resolved_comment_ids("proj/repo", 1)
        assert resolved == {10, 20}


class TestGetPrComments:
    """``get_pr_comments`` returns top-level PR comments in **chronological
    (oldest-first)** order, matching the Gitea provider's contract.

    The BB DC ``/activities`` endpoint pages newest-first, but every
    consumer (``server._process_comment``'s ``[-COMMENT_HISTORY:]`` "last
    N", ``reviewer._build_pr_context_section``'s ``reversed(...)`` budget
    walk) assumes oldest-first like Gitea. So the provider reverses the
    fully-paginated feed once before returning."""

    def _activities_response(self, comments, is_last=True, next_start=None):
        """Wrap the given comment dicts as ``action=COMMENTED`` activities."""
        return {
            "values": [{"action": "COMMENTED", "comment": c} for c in comments],
            "isLastPage": is_last,
            "nextPageStart": next_start or 0,
        }

    def _comment(self, cid):
        return {"id": cid, "author": {"slug": f"user{cid}"}, "text": f"comment {cid}"}

    def test_returns_chronological_across_pages(self, client):
        """The activities feed is newest-first and paginated: page 1 holds
        the newest comments [3, 2], page 2 the oldest [1]. ``get_pr_comments``
        must reverse the assembled feed so callers see oldest-first → [1, 2, 3]."""
        page1 = self._activities_response(
            [self._comment(3), self._comment(2)], is_last=False, next_start=50,
        )
        page2 = self._activities_response([self._comment(1)], is_last=True)
        pages = [page1, page2]

        def fake_get(url, params=None, timeout=None):
            start = (params or {}).get("start", 0)
            mr = MagicMock(status_code=200)
            mr.raise_for_status = MagicMock()
            mr.json.return_value = pages[0] if start == 0 else pages[1]
            return mr

        with patch.object(client.session, "get", side_effect=fake_get):
            comments = client.get_pr_comments("proj/repo", 1)
        assert [c["id"] for c in comments] == [1, 2, 3]

    def test_single_page_newest_first_reversed(self, client):
        """A single newest-first page [2, 1] is returned oldest-first [1, 2]."""
        with _mock_get(client, status=200,
                       json_data=self._activities_response(
                           [self._comment(2), self._comment(1)])):
            comments = client.get_pr_comments("proj/repo", 1)
        assert [c["id"] for c in comments] == [1, 2]

    def test_normalizes_comment_shape(self, client):
        """Each comment is normalized to the contract shape:
        ``{user: {login}, body, id}`` (author.slug → user.login,
        text → body)."""
        with _mock_get(client, status=200,
                       json_data=self._activities_response([self._comment(7)])):
            comments = client.get_pr_comments("proj/repo", 1)
        assert comments == [{"user": {"login": "user7"}, "body": "comment 7", "id": 7}]

    def test_returns_empty_when_no_comments(self, client):
        with _mock_get(client, status=200,
                       json_data=self._activities_response([])):
            assert client.get_pr_comments("proj/repo", 1) == []
