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


class TestFormatMessageUsesRepoScale:
    """The Slack/webhook display path must colour (and default) by the
    reviewed repo's own severity scale, not the built-in low/medium/high
    one. default_scale().emoji() normalizes an unknown name fail-closed to
    the most severe tier, so a custom scale's LEAST severe tier (e.g.
    'nit' in a nit/bug/blocker repo) would render red — the signal
    inverted and meaningless as colour."""

    def test_custom_scale_least_severe_tier_renders_yellow_not_red(self):
        review = {
            "severity": "nit",
            "summary": "s",
            "severity_scale_names": ["blocker", "bug", "nit"],
            "severity_blocks_at": "bug",
        }
        text = _format_message("acme/repo", "ref", review, "", "needs_review")
        assert "🟡" in text
        assert "🔴" not in text

    def test_custom_scale_middle_tier_renders_orange(self):
        review = {
            "severity": "bug",
            "summary": "s",
            "severity_scale_names": ["blocker", "bug", "nit"],
            "severity_blocks_at": "bug",
        }
        text = _format_message("acme/repo", "ref", review, "", "needs_review")
        assert "🟠" in text

    def test_custom_scale_most_severe_tier_still_renders_red(self):
        review = {
            "severity": "blocker",
            "summary": "s",
            "severity_scale_names": ["blocker", "bug", "nit"],
            "severity_blocks_at": "bug",
        }
        text = _format_message("acme/repo", "ref", review, "", "needs_review")
        assert "🔴" in text

    def test_legacy_dict_without_scale_still_defaults_and_renders_todays_colours(self):
        """No severity_scale_names key at all (cached/pre-feature review) —
        must render exactly as before: missing severity defaults to 'low'
        (now: the default scale's least-severe tier, which is the same
        value) and colours by the built-in scale."""
        text = _format_message("acme/repo", "ref", {"summary": "s"}, "", "needs_review")
        assert "🟡" in text
        assert "LOW" in text


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

    def test_clean_review_reads_as_no_issues(self):
        clean = {"severity": "low", "summary": "Clean", "findings": []}
        needs = _format_message("owner/repo", "PR #5", clean, "", "needs_review")
        alert = _format_message("owner/repo", "PR #5", clean, "", "")
        assert "✅ NO ISSUES — needs your review" in needs
        assert "Raven Alert* — ✅ NO ISSUES" in alert
        assert "LOW" not in needs + alert

    def test_blocking_clean_review_keeps_its_tier(self):
        blocked = {"severity": "low", "summary": "Clean", "findings": [], "blocking": True}
        text = _format_message("owner/repo", "PR #5", blocked, "", "")
        assert "🟡 LOW" in text and "NO ISSUES" not in text


class TestEmojiComesFromTheScale:
    def test_no_duplicate_emoji_table(self):
        """The two hardcoded SEVERITY_EMOJI dicts are consolidated into
        SeverityScale.emoji()."""
        import raven.notifier as notifier
        import raven.server as server

        assert not hasattr(notifier, "SEVERITY_EMOJI")
        assert not hasattr(server, "SEVERITY_EMOJI")

    def test_default_scale_emoji_match_todays_colours(self):
        from raven.severity import default_scale

        s = default_scale()
        assert s.emoji("high") == "🔴"
        assert s.emoji("medium") == "🟠"
        assert s.emoji("low") == "🟡"


class TestChannelThresholdAcrossVocabularies:
    def _review(self, severity, names, blocks_at):
        return {"severity": severity, "summary": "s", "findings": [],
                "severity_scale_names": names, "severity_blocks_at": blocks_at}

    def test_known_name_compares_by_rank(self, mocker):
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "medium", "webhook": "u"}])
        notifier.notify("acme/repo", "ref", self._review("high", ["high", "medium", "low"], "medium"), "l", "review")
        assert send.called

    def test_name_absent_from_scale_falls_back_to_blocking(self, mocker):
        """'medium' is meaningless for a nit/bug/blocker repo. Notify when
        the review blocks, rather than comparing across vocabularies."""
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "medium", "webhook": "u"}])
        notifier.notify("acme/repo", "ref", self._review("blocker", ["blocker", "bug", "nit"], "bug"), "l", "review")
        assert send.called

    def test_non_blocking_review_suppressed_under_fallback(self, mocker):
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "medium", "webhook": "u"}])
        notifier.notify("acme/repo", "ref", self._review("nit", ["blocker", "bug", "nit"], "bug"), "l", "review")
        assert not send.called

    def test_legacy_review_without_scale_keeps_todays_filtering(self, mocker):
        """A cached review from before this feature carries no scale keys.
        It must filter exactly as it does today, not degrade into
        always-notify."""
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "high", "webhook": "u"}])
        notifier.notify("acme/repo", "ref",
                        {"severity": "low", "summary": "s", "findings": []},
                        "l", "review")
        assert not send.called

    def test_explicit_blocking_sentinel(self, mocker):
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "blocking", "webhook": "u"}])
        notifier.notify("acme/repo", "ref", self._review("bug", ["blocker", "bug", "nit"], "bug"), "l", "review")
        assert send.called

    def test_no_blocking_tier_fails_toward_notify(self, mocker):
        """blocks_at is None means this repo's scale has no tier that blocks
        a merge (REVIEW_APPROVE_MAX_SEVERITY pinned at the top tier). A
        channel gated on 'blocking' has no rank to compare against, and the
        fail direction here is toward notifying: a missed alert is worse
        than a redundant one, and this path has no merge authority. This is
        the opposite of the merge gate's fail-closed rule — do not make it
        consistent with the gate."""
        import raven.notifier as notifier
        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "blocking", "webhook": "u"}])
        review = self._review("low", ["high", "medium", "low"], None)
        notifier.notify("acme/repo", "ref", review, "l", "review")
        assert send.called


class TestMissingSeverityKeyAgreesAcrossThresholdAndMessage:
    """PR #216 review, Finding 3: a review dict lacking 'severity' must
    resolve to the SAME tier in both _passes_threshold and
    _format_message for the identical dict. They used to disagree:
    _passes_threshold's in-scale branch evaluated scale.rank("") — and an
    unknown name normalizes fail-closed to the scale's MOST severe tier —
    while _format_message already defaulted to scale.least_severe. Impact
    was limited (the reviewer always populates 'severity'), but it's
    exactly the contradictory-defaults defect class this branch exists to
    eliminate."""

    def _review(self, names, blocks_at):
        # Deliberately no 'severity' key.
        return {"summary": "s", "findings": [],
                "severity_scale_names": names, "severity_blocks_at": blocks_at}

    def test_format_message_renders_the_least_severe_tier(self):
        review = self._review(["blocker", "bug", "nit"], "bug")
        # One finding, so the headline shows the tier rather than reading
        # as no issues (an empty list at the least tier does).
        review["findings"] = [{"severity": "nit", "message": "m"}]
        text = _format_message("acme/repo", "ref", review, "", "needs_review")
        assert "NIT" in text
        assert "🟡" in text

    def test_passes_threshold_agrees_with_format_message(self):
        """Same missing-severity dict: a channel gated at the scale's
        LEAST severe tier ('nit') must pass, and one gated at the next
        tier up ('bug') must not — i.e. _passes_threshold reads the
        missing severity as 'nit', exactly what _format_message renders.
        Before the fix, rank("") failed closed to 'blocker' (most severe),
        so BOTH assertions below would flip: 'bug' would pass too."""
        import raven.notifier as notifier

        review = self._review(["blocker", "bug", "nit"], "bug")
        assert notifier._passes_threshold(review, "nit", "acme/repo") is True
        assert notifier._passes_threshold(review, "bug", "acme/repo") is False

    def test_notify_dispatch_agrees_end_to_end(self, mocker):
        """Integration-level pin via the public notify() entry point, not
        just the private helper — a channel filtering at 'bug' (not the
        least-severe tier) must NOT fire for a review with no 'severity'
        key, matching how that same review renders as the LEAST severe
        tier in the message _format_message would have sent."""
        import raven.notifier as notifier

        send = mocker.patch.object(notifier, "_send_slack")
        mocker.patch.object(notifier, "_load_channels", return_value=[
            {"type": "slack", "min_severity": "bug", "webhook": "u"}])
        review = self._review(["blocker", "bug", "nit"], "bug")
        notifier.notify("acme/repo", "ref", review, "l", "review")
        assert not send.called


class TestScaleReconstructionInvariant:
    """A review dict whose severity_blocks_at names a tier absent from
    severity_scale_names must not crash the notifier.

    _scale_from_review builds `ranks` from severity_scale_names but reads
    the blocking tier from a separate field, so SeverityScale.blocks()
    would do ranks[blocks_at_or_above] -> KeyError. All current writers
    emit both fields from one scale object, so this is an unguarded
    invariant rather than a live bug — but the notifier is the last step
    of a completed review, and a KeyError there loses the notification
    for a review that already ran (audit 2026-08-14 LOW).
    """

    def test_scale_from_review_drops_blocking_tier_absent_from_names(self):
        from raven.notifier import _scale_from_review
        scale = _scale_from_review({
            "severity": "bug",
            "severity_scale_names": ["blocker", "bug", "nit"],
            "severity_blocks_at": "critical",   # not a tier in this scale
        })
        assert scale.blocks("bug") is False, (
            "An unresolvable blocking tier must degrade to 'nothing blocks', "
            "not raise"
        )

    def test_passes_threshold_survives_mismatched_blocking_tier(self):
        from raven.notifier import _passes_threshold
        review = {
            "severity": "bug",
            "severity_scale_names": ["blocker", "bug", "nit"],
            "severity_blocks_at": "critical",
        }
        # Must not raise. Gate semantics with an unresolvable blocking
        # tier means "nothing blocks", and notification fails toward
        # notifying, so this returns True.
        assert _passes_threshold(review, "blocking", "u/r") is True
