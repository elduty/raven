"""Tests for notifier.py — channel-based notification dispatch."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from raven.notifier import notify, _load_channels, _format_message


WEBHOOK_CHANNEL = {"type": "webhook", "url": "https://hook.test/hooks", "token": "tok"}


class TestLoadChannels:
    def test_valid_json(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps([WEBHOOK_CHANNEL])}):
            channels = _load_channels()
        assert len(channels) == 1
        assert channels[0]["type"] == "webhook"

    def test_empty_env(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": ""}):
            assert _load_channels() == []

    def test_unset_env(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _load_channels() == []

    def test_invalid_json(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": "not json"}):
            assert _load_channels() == []

    def test_non_array_json(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": '{"type": "webhook"}'}):
            assert _load_channels() == []

    def test_missing_type_skipped(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps([{"url": "https://x"}])}):
            assert _load_channels() == []

    def test_missing_url_skipped(self):
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps([{"type": "webhook"}])}):
            assert _load_channels() == []

    def test_valid_and_invalid_mixed(self):
        channels = [WEBHOOK_CHANNEL, {"type": "webhook"}]
        with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps(channels)}):
            result = _load_channels()
        assert len(result) == 1
        assert result[0]["url"] == WEBHOOK_CHANNEL["url"]


class TestNotify:
    def test_no_channels_returns_false(self):
        with patch("raven.notifier._load_channels", return_value=[]):
            result = notify("owner/repo", "PR #1", {"severity": "high", "summary": "test"})
        assert result is False

    def test_webhook_success(self):
        with patch("raven.notifier._load_channels", return_value=[WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "PR #1: Fix", {"severity": "high", "summary": "SQL injection"}, link="https://git/pr/1", action="needs_review")
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert "SQL injection" in payload["text"]

    def test_bearer_token_sent(self):
        with patch("raven.notifier._load_channels", return_value=[WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    def test_no_token_no_auth_header(self):
        channel = {"type": "webhook", "url": "https://hook.test/hooks"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        headers = mock_post.call_args[1]["headers"]
        assert "Authorization" not in headers

    def test_global_channel_matches_all_repos(self):
        with patch("raven.notifier._load_channels", return_value=[WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("any/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_repos_filter_matches(self):
        channel = {**WEBHOOK_CHANNEL, "repos": ["owner/repo"]}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_repos_filter_skips_non_matching(self):
        channel = {**WEBHOOK_CHANNEL, "repos": ["owner/other"]}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        mock_post.assert_not_called()

    def test_min_severity_filter_matches(self):
        channel = {**WEBHOOK_CHANNEL, "min_severity": "medium"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "high", "summary": "critical bug"})
        assert result is True
        mock_post.assert_called_once()

    def test_min_severity_filter_skips_low(self):
        channel = {**WEBHOOK_CHANNEL, "min_severity": "medium"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        mock_post.assert_not_called()

    def test_no_min_severity_notifies_all(self):
        with patch("raven.notifier._load_channels", return_value=[WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_multiple_channels_one_fails(self):
        import requests as req
        ch1 = {"type": "webhook", "url": "https://fail.test/hooks", "token": "a"}
        ch2 = {"type": "webhook", "url": "https://ok.test/hooks", "token": "b"}
        with patch("raven.notifier._load_channels", return_value=[ch1, ch2]):
            with patch("raven.notifier.requests.post") as mock_post:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = req.HTTPError("500")
                ok_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
                mock_post.side_effect = [fail_resp, ok_resp]
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        assert mock_post.call_count == 2

    def test_http_error_returns_false(self):
        import requests as req
        with patch("raven.notifier._load_channels", return_value=[WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post", side_effect=req.ConnectionError("down")):
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False

    def test_unknown_channel_type_skipped(self):
        channel = {"type": "telegram", "url": "https://t.me/hook"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False

    def test_slack_channel(self):
        channel = {"type": "slack", "url": "https://hooks.slack.com/services/xxx"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        payload = mock_post.call_args[1]["json"]
        assert "text" in payload

    def test_channels_reloaded_each_call(self):
        """Changing NOTIFY_CHANNELS between calls takes effect without restart."""
        first = {"type": "webhook", "url": "https://first.test/hook", "token": "a"}
        second = {"type": "webhook", "url": "https://second.test/hook", "token": "b"}

        with patch("raven.notifier.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())

            with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps([first])}):
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
            assert mock_post.call_args[0][0] == first["url"]

            with patch.dict(os.environ, {"NOTIFY_CHANNELS": json.dumps([second])}):
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
            assert mock_post.call_args[0][0] == second["url"]

            with patch.dict(os.environ, {"NOTIFY_CHANNELS": ""}):
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
            assert result is False
            assert mock_post.call_count == 2


class TestFailureLogRedaction:
    """Webhook URLs are bearer-equivalent secrets (Slack: the path IS the
    credential). requests exceptions embed the full URL, so failure logs must
    never interpolate the exception text."""

    SECRET_URL = "https://hooks.slack.com/services/T00/B00/SECRETtoken"

    def _http_error(self):
        import requests as req
        # Same shape requests builds in raise_for_status():
        return req.HTTPError(f"404 Client Error: Not Found for url: {self.SECRET_URL}")

    def test_secret_url_not_logged_on_failure(self, caplog):
        channel = {"type": "slack", "url": self.SECRET_URL}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = self._http_error()
                mock_post.return_value = fail_resp
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRETtoken" not in all_logs
        assert "/services/" not in all_logs

    def test_failure_still_logged_with_class_and_channel(self, caplog):
        channel = {"type": "slack", "url": self.SECRET_URL}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = self._http_error()
                mock_post.return_value = fail_resp
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert errors, "channel failure must be logged at ERROR"
        assert any("slack" in m for m in errors)
        assert any("HTTPError" in m for m in errors)
        # Host is fine to log — only the path is secret
        assert any("hooks.slack.com" in m for m in errors)

    def test_connection_error_url_not_logged(self, caplog):
        import requests as req
        channel = {"type": "webhook", "url": "https://internal.host/capability/SECRETtoken"}
        exc = req.ConnectionError(
            f"Max retries exceeded with url: /capability/SECRETtoken "
            f"(host='internal.host') for https://internal.host/capability/SECRETtoken"
        )
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post", side_effect=exc):
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRETtoken" not in all_logs
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("ConnectionError" in m for m in errors)

    def test_userinfo_not_logged_on_failure(self, caplog):
        """urlsplit().netloc includes userinfo — only hostname (+port) may be logged."""
        import requests as req
        channel = {"type": "webhook", "url": "https://user:SECRETPASS@internal.host/hook"}
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post", side_effect=req.ConnectionError("down")):
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRETPASS" not in all_logs
        assert "user:" not in all_logs
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("internal.host" in m for m in errors)

    def test_http_status_code_logged_on_failure(self, caplog):
        """Status code distinguishes revoked/mistyped webhook (404) from transient
        5xx and carries no secret material — it must survive the redaction."""
        import requests as req
        channel = {"type": "slack", "url": self.SECRET_URL}
        resp = MagicMock(status_code=404)
        exc = req.HTTPError(f"404 Client Error: Not Found for url: {self.SECRET_URL}", response=resp)
        with patch("raven.notifier._load_channels", return_value=[channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = exc
                mock_post.return_value = fail_resp
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("404" in m for m in errors)
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRETtoken" not in all_logs

    def test_other_channels_still_notified_after_failure(self, caplog):
        import requests as req
        ch1 = {"type": "slack", "url": self.SECRET_URL}
        ch2 = {"type": "webhook", "url": "https://ok.test/hooks", "token": "b"}
        with patch("raven.notifier._load_channels", return_value=[ch1, ch2]):
            with patch("raven.notifier.requests.post") as mock_post:
                fail_resp = MagicMock()
                fail_resp.raise_for_status.side_effect = self._http_error()
                ok_resp = MagicMock(status_code=200, raise_for_status=MagicMock())
                mock_post.side_effect = [fail_resp, ok_resp]
                with caplog.at_level("DEBUG", logger="raven.notifier"):
                    result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        assert mock_post.call_count == 2
        all_logs = "\n".join(r.getMessage() for r in caplog.records)
        assert "SECRETtoken" not in all_logs


class TestFormatMessage:
    def test_needs_review_header(self):
        text = _format_message("owner/repo", "PR #5", {"severity": "medium", "summary": "Missing validation"}, "", "needs_review")
        assert "needs your review" in text
        assert "🟠" in text

    def test_merge_failed_header(self):
        text = _format_message("owner/repo", "PR #3", {"severity": "low", "summary": "Clean"}, "", "merge_failed")
        assert "merge failed" in text

    def test_ci_failed_header(self):
        text = _format_message("owner/repo", "PR #3", {"severity": "low", "summary": "Clean"}, "", "ci_failed")
        assert "CI failed" in text

    def test_review_submit_failed_header(self):
        text = _format_message("owner/repo", "PR #3", {"severity": "low", "summary": "Clean"}, "", "review_submit_failed")
        assert "Failed to submit review" in text

    def test_link_appended(self):
        text = _format_message("owner/repo", "ref", {"severity": "low", "summary": "ok"}, "https://git/pr/1", "")
        assert "https://git/pr/1" in text
