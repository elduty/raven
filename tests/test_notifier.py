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
        with patch("raven.notifier._CHANNELS", []):
            result = notify("owner/repo", "PR #1", {"severity": "high", "summary": "test"})
        assert result is False

    def test_webhook_success(self):
        with patch("raven.notifier._CHANNELS", [WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "PR #1: Fix", {"severity": "high", "summary": "SQL injection"}, link="https://git/pr/1", action="needs_review")
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert "SQL injection" in payload["text"]

    def test_bearer_token_sent(self):
        with patch("raven.notifier._CHANNELS", [WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer tok"

    def test_no_token_no_auth_header(self):
        channel = {"type": "webhook", "url": "https://hook.test/hooks"}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        headers = mock_post.call_args[1]["headers"]
        assert "Authorization" not in headers

    def test_global_channel_matches_all_repos(self):
        with patch("raven.notifier._CHANNELS", [WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("any/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_repos_filter_matches(self):
        channel = {**WEBHOOK_CHANNEL, "repos": ["owner/repo"]}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_repos_filter_skips_non_matching(self):
        channel = {**WEBHOOK_CHANNEL, "repos": ["owner/other"]}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        mock_post.assert_not_called()

    def test_min_severity_filter_matches(self):
        channel = {**WEBHOOK_CHANNEL, "min_severity": "medium"}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "high", "summary": "critical bug"})
        assert result is True
        mock_post.assert_called_once()

    def test_min_severity_filter_skips_low(self):
        channel = {**WEBHOOK_CHANNEL, "min_severity": "medium"}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False
        mock_post.assert_not_called()

    def test_no_min_severity_notifies_all(self):
        with patch("raven.notifier._CHANNELS", [WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        mock_post.assert_called_once()

    def test_multiple_channels_one_fails(self):
        import requests as req
        ch1 = {"type": "webhook", "url": "https://fail.test/hooks", "token": "a"}
        ch2 = {"type": "webhook", "url": "https://ok.test/hooks", "token": "b"}
        with patch("raven.notifier._CHANNELS", [ch1, ch2]):
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
        with patch("raven.notifier._CHANNELS", [WEBHOOK_CHANNEL]):
            with patch("raven.notifier.requests.post", side_effect=req.ConnectionError("down")):
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False

    def test_unknown_channel_type_skipped(self):
        channel = {"type": "telegram", "url": "https://t.me/hook"}
        with patch("raven.notifier._CHANNELS", [channel]):
            result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is False

    def test_slack_channel(self):
        channel = {"type": "slack", "url": "https://hooks.slack.com/services/xxx"}
        with patch("raven.notifier._CHANNELS", [channel]):
            with patch("raven.notifier.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200, raise_for_status=MagicMock())
                result = notify("owner/repo", "ref", {"severity": "low", "summary": "ok"})
        assert result is True
        payload = mock_post.call_args[1]["json"]
        assert "text" in payload


class TestFormatMessage:
    def test_needs_review_header(self):
        text = _format_message("owner/repo", "PR #5", {"severity": "medium", "summary": "Missing validation"}, "", "needs_review")
        assert "needs your review" in text
        assert "🟡" in text

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
