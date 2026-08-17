"""Tests for server.py — webhook handling, PR flow, signature validation."""

import hashlib
import hmac
import json
import os
import pytest
from unittest.mock import MagicMock, patch


import raven.server as _server_mod
from raven.server import create_app, _is_bot_author, _is_skipped_repo, _format_comment, _fetch_changed_files, _fetch_rules, _findings_by_file, _load_cache, _save_cache, _evict_cache, _process_pr, _process_comment, _wait_for_ci, _should_skip_duplicate, _do_merge, _safe_do_merge, _truncate_diff_for_comment, _extract_code_snippet, _shutdown_executor, _recent_prs, _previous_diffs, _MAX_CACHED_PRS, DEDUP_WINDOW, CacheEntry
from raven.providers import GitProvider, _providers


SECRET = "testsecret"


@pytest.fixture(autouse=True)
def _inline_ci_wait_executor():
    """Make ``ci_wait_executor.submit`` run tasks inline so existing
    tests that assert ``merge_pr.assert_called_once()`` after
    ``_process_pr`` keep working without the background thread race.

    Tests that need to inspect the real executor (dispatch verification,
    shutdown) override this by ``patch("raven.server.ci_wait_executor")``
    — the patch wins over the fixture's module assignment."""
    from concurrent.futures import Future

    class _InlineExecutor:
        def submit(self, fn, *args, **kwargs):
            fut: Future = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                fut.set_exception(exc)
            return fut

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    original = _server_mod.ci_wait_executor
    _server_mod.ci_wait_executor = _InlineExecutor()
    try:
        yield
    finally:
        _server_mod.ci_wait_executor = original


@pytest.fixture()
def app():
    _providers.clear()
    app = create_app()
    app.config["TESTING"] = True
    yield app
    _providers.clear()


@pytest.fixture()
def client(app):
    return app.test_client()


def _sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post(client, payload_dict, event="push", secret=SECRET):
    body = json.dumps(payload_dict).encode()
    sig = _sign(body, secret)
    return client.post(
        "/hook/gitea",
        data=body,
        headers={
            "X-Gitea-Signature": sig,
            "X-Gitea-Event": event,
            "Content-Type": "application/json",
        },
    )


class TestStartupAssertion:
    def setup_method(self):
        _providers.clear()

    def teardown_method(self):
        _providers.clear()

    def test_fails_without_secret(self):
        with patch.dict(os.environ, {"GITEA_WEBHOOK_SECRET": "", "GITEA_URL": "https://x", "GITEA_TOKEN": "t"}):
            with pytest.raises(RuntimeError, match="No git providers configured"):
                create_app()

    def test_fails_without_gitea_url(self):
        with patch.dict(os.environ, {"GITEA_WEBHOOK_SECRET": "s", "GITEA_URL": "", "GITEA_TOKEN": "t"}):
            with pytest.raises(RuntimeError, match="No git providers configured"):
                create_app()

    def test_fails_without_gitea_token(self):
        with patch.dict(os.environ, {"GITEA_WEBHOOK_SECRET": "s", "GITEA_URL": "https://x", "GITEA_TOKEN": ""}):
            with pytest.raises(RuntimeError, match="No git providers configured"):
                create_app()

    def test_reports_all_missing_vars(self):
        with patch.dict(os.environ, {"GITEA_WEBHOOK_SECRET": "", "GITEA_URL": "", "GITEA_TOKEN": ""}):
            with pytest.raises(RuntimeError, match="No git providers configured"):
                create_app()

    # ── BB DC username is required, not optional (audit 2026-07-30) ───── #
    # BB DC tokens expose no whoami endpoint, so get_authenticated_user()
    # raises without it — and it is called ONLY on submit_review's
    # needs-work path. Unset, the failure is asymmetric and confusing:
    # approvals post fine while every review WITH findings raises at submit
    # time and the author sees a generic internal-error comment. The
    # "no providers configured" message already listed it as required.

    def test_bb_dc_fails_fast_without_username(self):
        env = {
            "GITEA_WEBHOOK_SECRET": "", "GITEA_URL": "", "GITEA_TOKEN": "",
            "BITBUCKET_DC_URL": "https://bb.example.com",
            "BITBUCKET_DC_TOKEN": "tok",
            "BITBUCKET_DC_WEBHOOK_SECRET": "sec",
            "BITBUCKET_DC_USERNAME": "",
        }
        with patch.dict(os.environ, env):
            with pytest.raises(RuntimeError, match="BITBUCKET_DC_USERNAME is required"):
                create_app()

    def test_bb_dc_username_whitespace_only_is_rejected(self):
        env = {
            "GITEA_WEBHOOK_SECRET": "", "GITEA_URL": "", "GITEA_TOKEN": "",
            "BITBUCKET_DC_URL": "https://bb.example.com",
            "BITBUCKET_DC_TOKEN": "tok",
            "BITBUCKET_DC_WEBHOOK_SECRET": "sec",
            "BITBUCKET_DC_USERNAME": "   ",
        }
        with patch.dict(os.environ, env):
            with pytest.raises(RuntimeError, match="BITBUCKET_DC_USERNAME is required"):
                create_app()


class TestAIAndWorkerStartupValidation:
    """create_app validates the AI backend + the single-worker invariant at
    boot (audit 07-02 #4 + audit-06-13 #9), so a misconfiguration fails fast
    here instead of surfacing as an opaque per-PR error (or silently racing
    across workers) while /healthz still reads healthy. Provider creds come
    from conftest, so the no-providers gate passes and these later checks run."""

    def setup_method(self):
        from raven.ai import _reset_backend_cache
        _providers.clear()
        _reset_backend_cache()

    def teardown_method(self):
        from raven.ai import _reset_backend_cache
        _providers.clear()
        _reset_backend_cache()

    def test_backend_misconfig_fails_startup(self, monkeypatch):
        # No backend creds at all → get_backend() must raise at startup, not
        # defer an opaque failure to the first review.
        for var in ("RAVEN_AI_BACKEND", "CLAUDE_CODE_OAUTH_TOKEN",
                    "RAVEN_AI_API_BASE", "RAVEN_AI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        with pytest.raises(RuntimeError, match="(?i)AI backend"):
            create_app()

    def test_unknown_effort_logs_warning_at_startup(self, monkeypatch, caplog):
        monkeypatch.setenv("RAVEN_AI_EFFORT", "xhigh")
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.delenv("GUNICORN_CMD_ARGS", raising=False)
        with caplog.at_level("WARNING", logger="raven.server"):
            create_app()
        assert any("xhigh" in r.message and "effort" in r.message.lower()
                   for r in caplog.records)

    def test_multi_worker_web_concurrency_fails_startup(self, monkeypatch):
        monkeypatch.setenv("WEB_CONCURRENCY", "2")
        with pytest.raises(RuntimeError, match="(?i)worker"):
            create_app()

    def test_multi_worker_gunicorn_cmd_args_fails_startup(self, monkeypatch):
        monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
        monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 3 --timeout 300")
        with pytest.raises(RuntimeError, match="(?i)worker"):
            create_app()

    def test_single_worker_web_concurrency_ok(self, monkeypatch):
        # WEB_CONCURRENCY=1 is the required value — must NOT false-positive.
        monkeypatch.setenv("WEB_CONCURRENCY", "1")
        monkeypatch.delenv("GUNICORN_CMD_ARGS", raising=False)
        app = create_app()
        assert app is not None


class TestMetricsAuth:
    def _build_client(self, monkeypatch, token: str | None):
        _providers.clear()
        if token is None:
            monkeypatch.delenv("RAVEN_METRICS_TOKEN", raising=False)
        else:
            monkeypatch.setenv("RAVEN_METRICS_TOKEN", token)
        app = create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def teardown_method(self):
        _providers.clear()

    def test_unset_token_returns_404(self, monkeypatch):
        client = self._build_client(monkeypatch, token=None)
        resp = client.get("/metrics")
        assert resp.status_code == 404

    def test_unset_token_ignores_authorization_header(self, monkeypatch):
        client = self._build_client(monkeypatch, token=None)
        resp = client.get("/metrics", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 404

    def test_missing_header_returns_404(self, monkeypatch):
        client = self._build_client(monkeypatch, token="s3cret")
        resp = client.get("/metrics")
        assert resp.status_code == 404

    def test_wrong_token_returns_404(self, monkeypatch):
        client = self._build_client(monkeypatch, token="s3cret")
        resp = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 404

    def test_malformed_header_returns_404(self, monkeypatch):
        client = self._build_client(monkeypatch, token="s3cret")
        resp = client.get("/metrics", headers={"Authorization": "s3cret"})
        assert resp.status_code == 404

    def test_correct_token_returns_200(self, monkeypatch):
        client = self._build_client(monkeypatch, token="s3cret")
        resp = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
        assert resp.status_code == 200
        assert resp.headers["Content-Type"].startswith("text/plain")


class TestSignatureValidation:
    def test_valid_signature_accepted(self, client):
        payload = {"ref": "refs/heads/feature", "commits": [], "repository": {"full_name": "u/r"}}
        resp = _post(client, payload)
        assert resp.status_code == 200

    def test_invalid_signature_rejected(self, client):
        body = b'{"ref": "refs/heads/main", "commits": [], "repository": {"full_name": "u/r"}}'
        resp = client.post("/hook/gitea", data=body, headers={"X-Gitea-Signature": "badhash", "X-Gitea-Event": "push"})
        assert resp.status_code == 403

    def test_missing_signature_rejected(self, client):
        resp = client.post("/hook/gitea", data=b"{}", headers={"X-Gitea-Event": "push"})
        assert resp.status_code == 403


class TestRequestBodySizeCap:
    """An oversized request body must be rejected with 413 BEFORE the
    HMAC signature check buffers it into memory (audit #12)."""

    def test_max_content_length_configured(self, app):
        # Guard: the hardcoded cap is set so Flask rejects oversized webhook
        # bodies before validate_signature() calls request.get_data().
        assert app.config["MAX_CONTENT_LENGTH"] == 25 * 1024 * 1024

    def test_oversized_body_rejected_with_413(self, client):
        # Functional end-to-end check at the real (hardcoded) cap. Werkzeug
        # rejects on the Content-Length check before reading/buffering the
        # body, so a 25 MB+1 payload is rejected near-instantly (~10ms) — no
        # heavy server-side allocation. This fails (403, not 413) without the
        # MAX_CONTENT_LENGTH cap, so it pins the production behaviour.
        body = b"x" * (25 * 1024 * 1024 + 1)
        resp = client.post(
            "/hook/gitea",
            data=body,
            headers={"X-Gitea-Signature": "irrelevant", "X-Gitea-Event": "push"},
        )
        assert resp.status_code == 413

    def test_normal_size_body_not_rejected_for_size(self, client):
        # A normal webhook payload (well under the cap) still reaches signature
        # validation — rejected for a bad signature (403), not size (413) — so
        # the cap does not interfere with real traffic.
        resp = client.post(
            "/hook/gitea",
            data=b'{"ref": "refs/heads/x", "repository": {"full_name": "u/r"}}',
            headers={"X-Gitea-Signature": "badhash", "X-Gitea-Event": "push"},
        )
        assert resp.status_code == 403


class TestPushEvent:
    def test_push_to_main_skipped(self, client):
        payload = {"ref": "refs/heads/main", "repository": {"full_name": "u/r"}, "pusher": {"login": "human"}}
        resp = _post(client, payload, event="push")
        assert resp.get_json()["status"] == "skipped"

    def test_push_to_feature_branch_no_pr_skipped(self, client):
        payload = {"ref": "refs/heads/feature/foo", "repository": {"full_name": "u/r", "default_branch": "main"}, "pusher": {"login": "human"}}
        provider = _providers["gitea"]
        with patch.object(provider, "find_open_pr_for_branch", return_value=None):
            resp = _post(client, payload, event="push")
        data = resp.get_json()
        assert data["status"] == "skipped"
        assert data["reason"] == "no open PR for branch"

    def test_push_to_feature_branch_with_pr_triggers_review(self, client):
        pr = {"number": 7, "title": "test", "html_url": "http://x", "head": {"ref": "feature/foo", "sha": "abc"}, "base": {"sha": "base"}}
        payload = {"ref": "refs/heads/feature/foo", "repository": {"full_name": "u/r", "default_branch": "main"}, "pusher": {"login": "human"}}
        provider = _providers["gitea"]
        with patch.object(provider, "find_open_pr_for_branch", return_value=pr), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, payload, event="push")
        data = resp.get_json()
        assert data["status"] == "accepted"
        assert mock_executor.submit.called

    def test_push_gitea_error_returns_skipped(self, client):
        payload = {"ref": "refs/heads/feature/foo", "repository": {"full_name": "u/r", "default_branch": "main"}, "pusher": {"login": "human"}}
        provider = _providers["gitea"]
        with patch.object(provider, "find_open_pr_for_branch", side_effect=Exception("connection refused")):
            resp = _post(client, payload, event="push")
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["status"] == "skipped"
        assert data["reason"] == "no open PR for branch"

    def test_push_to_custom_default_branch_skipped(self, client):
        payload = {"ref": "refs/heads/trunk", "repository": {"full_name": "u/r", "default_branch": "trunk"}, "pusher": {"login": "human"}}
        resp = _post(client, payload, event="push")
        assert resp.get_json()["status"] == "skipped"

    def test_unknown_event_ignored(self, client):
        payload = {"repository": {"full_name": "u/r"}}
        resp = _post(client, payload, event="release")
        data = resp.get_json()
        assert data["status"] == "ignored"


class TestPRWebhook:
    def _pr_payload(self, action="opened", repo="owner/repo", pr_number=42, sender="alice"):
        return {
            "action": action,
            "repository": {"full_name": repo},
            "pull_request": {
                "number": pr_number,
                "title": f"PR #{pr_number}",
                "head": {"ref": "feature-branch", "sha": "abc123def"},
                "html_url": f"https://git/pulls/{pr_number}",
            },
            "sender": {"login": sender},
        }

    def test_pr_opened_returns_accepted(self, client):
        with patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._pr_payload("opened"), event="pull_request")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_pr_closed_ignored(self, client):
        resp = _post(client, self._pr_payload("closed"), event="pull_request")
        assert resp.get_json()["status"] == "ignored"

    def test_pr_skipped_repo(self, client):
        with patch.dict(os.environ, {"SKIP_REPOS": "owner/repo"}):
            resp = _post(client, self._pr_payload(), event="pull_request")
        assert resp.get_json()["status"] == "skipped"

    def test_pr_bot_sender_skipped(self, client):
        resp = _post(client, self._pr_payload(sender="dependabot"), event="pull_request")
        assert resp.get_json()["status"] == "skipped"

    def test_review_requested_for_raven_triggers_review(self, client):
        _recent_prs.clear()
        payload = self._pr_payload("review_requested")
        payload["requested_reviewer"] = {"login": "Raven"}
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, payload, event="pull_request")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_review_requested_for_other_user_ignored(self, client):
        payload = self._pr_payload("review_requested")
        payload["requested_reviewer"] = {"login": "alice"}
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"):
            resp = _post(client, payload, event="pull_request")
        assert resp.get_json()["status"] == "ignored"

    def test_review_requested_self_triggered_ignored(self, client):
        """When Raven adds itself as a reviewer, the resulting reviewer-updated
        webhook must not trigger a second review. Otherwise _process_pr runs
        twice for the same PR."""
        _recent_prs.clear()
        payload = self._pr_payload("review_requested", sender="Raven")
        payload["requested_reviewer"] = {"login": "Raven"}
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, payload, event="pull_request")
        assert resp.get_json()["status"] == "ignored"
        assert resp.get_json()["reason"] == "self-triggered"
        mock_executor.submit.assert_not_called()


class TestProcessPr:
    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        return mc

    def _setup_raven_only(self, mc):
        """Configure mocks so Raven is the only reviewer."""
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"

    def test_pr_description_and_comments_passed_to_review_diff(self):
        """Author intent (PR body) and prior-reviewer context (comments)
        reach review_diff so the model can see constraints like
        "intentionally skipping X because Y". Bot login is resolved from
        the provider and forwarded so review_diff can filter the bot's
        own prior comments out of the prompt context."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = "Fixes DEV-123. Intentionally skipping migration — handled in follow-up PR."
        mc.get_pr_comments.return_value = [
            {"user": {"login": "alice"}, "body": "Should this be behind a feature flag?"},
        ]
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mc.get_authenticated_user.return_value = "raven-bot"
            # Gate uses get_authenticated_user() to resolve "raven-bot" and looks
            # in get_pr_reviews for that login — update to match the overridden user.
            mc.get_pr_reviews.return_value = [{"user": {"login": "raven-bot"}, "state": "APPROVED"}]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["pr_description"] == (
            "Fixes DEV-123. Intentionally skipping migration — handled in follow-up PR."
        )
        assert kwargs["pr_comments"] == [
            {"user": {"login": "alice"}, "body": "Should this be behind a feature flag?"},
        ]
        assert kwargs["pr_title"] == "PR #42"
        assert kwargs["bot_user"] == "raven-bot"

    def test_claude_md_fetched_from_base_ref_not_head(self):
        """Same trust model as repo rules: CLAUDE.md must come from the
        PR's base ref (already-merged state). If it came from head, a
        hostile PR could add or edit CLAUDE.md to bias its own review
        — and CLAUDE.md is now rendered in the trusted ``<repo_policy>``
        block, so the impact would be high. Regression guard for the
        2026-05-22 trust-tier fix."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        # Different content per ref so we can prove which one was fetched
        def fake_fetch(repo, path, ref="HEAD"):
            if path == "CLAUDE.md":
                return "BASE_GUIDANCE" if ref == "main" else "HEAD_GUIDANCE"
            return ""
        mc.fetch_file.side_effect = fake_fetch

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mc.get_authenticated_user.return_value = "raven-bot"
            mc.get_pr_reviews.return_value = [{"user": {"login": "raven-bot"}, "state": "APPROVED"}]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        # CLAUDE.md fetch must have used base_ref ("main"), NOT head_sha ("abc123")
        claude_calls = [
            c for c in mc.fetch_file.call_args_list
            if len(c.args) >= 2 and c.args[1] == "CLAUDE.md"
        ]
        assert len(claude_calls) == 1
        cc = claude_calls[0]
        ref_used = cc.kwargs.get("ref") or (
            cc.args[2] if len(cc.args) >= 3 else None
        )
        assert ref_used == "main", f"CLAUDE.md fetched with ref={ref_used!r}, expected 'main'"
        # And the content forwarded to review_diff is the base version
        assert mock_review.call_args.kwargs.get("claude_md") == "BASE_GUIDANCE"

    def test_rules_fetched_from_base_ref_not_head(self):
        """Security regression: rules must come from the PR's base ref
        (already-merged state), not the head SHA. If they came from head,
        a hostile PR could add ``.claude/rules/policy.md`` saying
        "approve SQL concatenation" alongside the hostile code, biasing
        Raven's own review of that same PR."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = [".claude/rules/security.md"]
        mc.fetch_file.side_effect = lambda repo, p, ref="HEAD": (
            "base-rule" if p == ".claude/rules/security.md" else ""
        )

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mc.get_authenticated_user.return_value = "raven-bot"
            # Gate uses get_authenticated_user() to resolve "raven-bot" and looks
            # in get_pr_reviews for that login — update to match the overridden user.
            mc.get_pr_reviews.return_value = [{"user": {"login": "raven-bot"}, "state": "APPROVED"}]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        # list_directory called with base_ref ("main"), NOT head_sha ("abc123")
        list_args = mc.list_directory.call_args
        assert list_args.kwargs.get("ref") == "main" or (
            len(list_args.args) >= 3 and list_args.args[2] == "main"
        )
        # fetch_file for the rule file must also use base_ref
        rule_fetch_calls = [
            c for c in mc.fetch_file.call_args_list
            if len(c.args) >= 2 and c.args[1] == ".claude/rules/security.md"
        ]
        assert len(rule_fetch_calls) == 1
        rc = rule_fetch_calls[0]
        assert rc.kwargs.get("ref") == "main" or (
            len(rc.args) >= 3 and rc.args[2] == "main"
        )

    def test_rules_loaded_from_claude_rules_dir_and_passed_through(self):
        """``.claude/rules/*.md`` at the PR head are fetched and passed
        to review_diff. Non-.md files are ignored."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = [
            ".claude/rules/security.md",
            ".claude/rules/style.md",
            ".claude/rules/NOTES",  # no .md — must be skipped
        ]
        # fetch_file is called for CLAUDE.md + the changed diff files +
        # each rule file. Use side_effect path-aware so each returns the
        # right thing.
        def fake_fetch(repo, path, ref="HEAD"):
            return {
                ".claude/rules/security.md": "Parameterize all SQL.",
                ".claude/rules/style.md": "Use PEP 8.",
            }.get(path, "")
        mc.fetch_file.side_effect = fake_fetch

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mc.get_authenticated_user.return_value = "raven-bot"
            # Gate uses get_authenticated_user() to resolve "raven-bot" and looks
            # in get_pr_reviews for that login — update to match the overridden user.
            mc.get_pr_reviews.return_value = [{"user": {"login": "raven-bot"}, "state": "APPROVED"}]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["rules"] == {
            ".claude/rules/security.md": "Parameterize all SQL.",
            ".claude/rules/style.md": "Use PEP 8.",
        }

    def test_rules_dir_missing_does_not_block_review(self):
        """Common case: the repo has no ``.claude/rules/``. list_directory
        returns []; review must proceed with rules={}."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.fetch_file.return_value = ""

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["rules"] == {}
        mock_review.assert_called_once()

    def test_review_prompt_override_threaded_to_review_diff(self):
        """When .claude/rules/raven/prompts/review.md exists on the base
        branch, its contents are passed to review_diff as prompt_override."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []

        def fetch_file(repo, path, ref="HEAD"):
            if path == ".claude/rules/raven/prompts/review.md":
                assert ref == "main"  # fetched from base branch
                return "REPO-SPECIFIC REVIEW PROMPT"
            return ""

        mc.fetch_file.side_effect = fetch_file
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.submit_review.return_value = {"id": 1}
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        self._setup_raven_only(mc)

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs.get("prompt_override") == "REPO-SPECIFIC REVIEW PROMPT"

    def test_review_prompt_override_none_when_file_missing(self):
        """When the override file doesn't exist, prompt_override is None
        (helper swallows the FileNotFoundError)."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.fetch_file.side_effect = FileNotFoundError()
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.submit_review.return_value = {"id": 1}
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        self._setup_raven_only(mc)

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs.get("prompt_override") is None

    def test_review_proceeds_when_claude_md_missing(self):
        """Regression guard for the explicit 'works without CLAUDE.md'
        requirement. fetch_file returns '' (or raises 404) for CLAUDE.md;
        review still runs and claude_md is empty."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        # fetch_file called for CLAUDE.md + any changed files. Return ""
        # for everything to simulate a repo with neither.
        mc.fetch_file.return_value = ""

        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["claude_md"] == ""
        mock_review.assert_called_once()

    def test_bot_user_resolve_failure_degrades_to_empty_filter(self):
        """If get_authenticated_user blows up, we still want to review —
        just with no bot-comment filter. Review must proceed.

        _process_pr calls get_authenticated_user multiple times (auto-add
        check, reviewer-status gate, PR-context filter, dismiss-previous,
        sole-reviewer check). Real providers cache, but MagicMock doesn't
        — side_effect needs to cover every call. The test's concern is the
        PR-context-filter call: position it third (after auto-add check and
        gate) and assert that bot_user ends up empty."""
        mc = self._make_provider()
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.get_authenticated_user.side_effect = [
            "raven-bot",       # auto-add check
            "raven-bot",       # reviewer-status gate
            Exception("500"),  # PR-context filter resolve — the one we're testing
            "raven-bot",       # dismiss-previous
            "raven-bot",       # sole-reviewer check
        ]
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            mc.get_pr_reviews.return_value = [{"user": {"login": "raven-bot"}, "state": "APPROVED"}]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["bot_user"] == ""
        mock_review.assert_called_once()

    def test_pr_context_fetch_failure_does_not_block_review(self):
        """Description/comment fetch is best-effort — a provider API
        hiccup must not abort the review. Reviewer gets empty context."""
        mc = self._make_provider()
        mc.get_pr_description.side_effect = Exception("500")
        mc.get_pr_comments.side_effect = Exception("500")
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        kwargs = mock_review.call_args.kwargs
        assert kwargs["pr_description"] == ""
        assert kwargs["pr_comments"] == []
        # Review still ran
        mock_review.assert_called_once()

    def test_low_severity_merged_when_ci_passes(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_called_once()
        mock_notify.assert_not_called()

    def test_low_severity_merged_when_no_ci(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "none"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_called_once()

    def test_low_severity_not_merged_when_ci_fails(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "failure"
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()
        mock_notify.assert_called_once()
        assert mock_notify.call_args[1]["action"] == "ci_failed"

    def test_medium_severity_not_merged(self):
        mc = self._make_provider()
        self._setup_raven_only(mc)
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mock_review.return_value = {
                "severity": "medium",
                "summary": "Missing error handling",
                "findings": [{"severity": "medium", "message": "No try/except"}],
            }
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()
        mock_notify.assert_called_once()
        assert mock_notify.call_args[1]["action"] == "needs_review"

    def test_not_merged_when_human_reviewed(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "Raven"}, "state": "APPROVED"},
                {"user": {"login": "alice"}, "state": "COMMENT"},
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()

    def test_not_merged_when_reviewer_requested(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
            mc.get_pr_requested_reviewers.return_value = ["bob"]
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()

    def test_merged_when_only_raven_in_requested_reviewers(self):
        """Regression guard: when Raven auto-adds itself or a human
        re-requests its review, the bot's own login lands in
        requested_reviewers. The auto-merge gate must filter Raven out
        of that list — otherwise every PR Raven self-requests is
        falsely classified as 'has other reviewers' and never merges.

        This bug stayed hidden because the existing reviewer-gate
        tests populated requested_reviewers with non-Raven names only.
        """
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            # Raven approved AND Raven is still in requested_reviewers
            # (mixed case to verify the filter is case-insensitive).
            mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
            mc.get_pr_requested_reviewers.return_value = ["raven"]
            mc.get_pr_head_sha.return_value = "abc123"
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_called_once()

    def test_merge_passes_head_sha_for_atomic_safety(self):
        """head_commit_id is passed to merge_pr so Gitea rejects if SHA changed."""
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_called_once()
        assert mc.merge_pr.call_args.kwargs.get("head_sha") == "abc123"

    def test_empty_diff_after_stripping_posts_comment_no_review(self):
        lockfile_only_diff = (
            "diff --git a/yarn.lock b/yarn.lock\n"
            "index abc..def 100644\n"
            "--- a/yarn.lock\n"
            "+++ b/yarn.lock\n"
            "@@ -1,3 +1,3 @@\n"
            "-old dep\n"
            "+new dep\n"
        )
        mc = self._make_provider()
        self._setup_raven_only(mc)
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = lockfile_only_diff
            mc.fetch_file.return_value = ""
            mc.post_pr_comment.return_value = {"id": 1}
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_not_called()
        mc.merge_pr.assert_not_called()
        mc.post_pr_comment.assert_called_once()
        assert "Empty diff" in mc.post_pr_comment.call_args[0][2]

    def test_review_submit_failure_blocks_merge(self):
        mc = self._make_provider()
        self._setup_raven_only(mc)
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.side_effect = Exception("API error")
            mc.add_label_to_pr.return_value = None
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()
        mock_notify.assert_called_once()
        assert mock_notify.call_args[1]["action"] == "review_submit_failed"

    def test_merge_failure_notifies(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = False
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "Clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mock_notify.assert_called_once()
        assert mock_notify.call_args[1]["action"] == "merge_failed"

    def test_request_changes_not_merged_even_as_sole_reviewer(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mock_review.return_value = {"severity": "medium", "summary": "Issue", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()

    def test_label_applied(self):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "OK", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.add_label_to_pr.assert_called_once_with("owner/repo", 42)

    def test_adds_self_as_reviewer_before_diff_fetch(self):
        """Raven must be added as reviewer before the review work starts,
        so the pending review is visible immediately in the PR list."""
        mc = self._make_provider()
        call_order = []
        mc.add_self_as_reviewer.side_effect = lambda *a, **kw: call_order.append("add_self")
        mc.fetch_pr_diff.side_effect = lambda *a, **kw: (call_order.append("fetch_diff") or "diff --git a/f\n+line\n")
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            # Raven not yet listed — default review-all mode will add it.
            # Gate (second get_pr_reviews call) sees Raven after auto-add.
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                            # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],          # gate check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],          # sole-reviewer check
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mock_review.return_value = {"severity": "low", "summary": "OK", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.add_self_as_reviewer.assert_called_once_with("owner/repo", 42)
        assert call_order.index("add_self") < call_order.index("fetch_diff")

    def test_does_not_auto_add_when_human_already_reviewing(self):
        """In fill-gap mode, if a human has reviewed (or is set to review),
        Raven must not claim the reviewer slot AND must not review — the
        reviewer-status gate blocks the review since Raven is not listed.
        The auto-add step is skipped and the PR is left to its human reviewers."""
        mc = self._make_provider()
        # Human reviewer already posted a review
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "alice"}, "state": "COMMENT",
             "commit_id": "abc123", "stale": False},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server.RAVEN_REVIEW_MODE", "gap"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mc.add_self_as_reviewer.assert_not_called()
        # Reviewer-status gate blocks review — Raven is not listed as a reviewer
        mock_review.assert_not_called()
        mc.submit_review.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_does_not_auto_add_when_human_requested(self):
        """In fill-gap mode, a human pending-reviewer slot keeps Raven out
        of the way. Reviewer-status gate also blocks the review since Raven
        is not listed as a reviewer."""
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = []
        mc.get_pr_requested_reviewers.return_value = ["alice"]
        mc.get_pr_head_sha.return_value = "abc123"
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server.RAVEN_REVIEW_MODE", "gap"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mc.add_self_as_reviewer.assert_not_called()
        # Reviewer-status gate blocks review — Raven is not a listed reviewer
        mock_review.assert_not_called()

    def test_does_not_re_add_when_raven_is_sole_existing_reviewer(self):
        """Re-review case: Raven was added in a previous run. The
        idempotency gate detects Raven is already listed and skips the
        add — both in review-all and fill-gap mode."""
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED",
             "commit_id": "abc123", "stale": False},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.merge_pr.return_value = True
            mc.get_commit_status.return_value = "success"
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mc.add_self_as_reviewer.assert_not_called()

    def test_add_self_as_reviewer_failure_does_not_block_review(self):
        """If add_self_as_reviewer fails, the review should still proceed."""
        mc = self._make_provider()
        mc.add_self_as_reviewer.side_effect = RuntimeError("API down")
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "OK", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.submit_review.assert_called_once()
        mc.merge_pr.assert_called_once()

    def test_concurrent_process_pr_same_pr_skipped(self):
        """If one thread is already reviewing a PR and a second _process_pr
        fires for the same PR, the second exits without fetching a diff or
        calling review_diff — prevents cache races and duplicate reviews."""
        import raven.server as _srv
        _srv._in_progress_prs.add("gitea:owner/repo#42")
        mc = self._make_provider()
        try:
            _process_pr(mc, self._normalized_payload())
        finally:
            _srv._in_progress_prs.discard("gitea:owner/repo#42")
        mc.add_self_as_reviewer.assert_not_called()
        mc.fetch_pr_diff.assert_not_called()
        mc.submit_review.assert_not_called()

    def test_in_progress_guard_cleared_after_normal_flow(self):
        """Normal completion clears the in-progress key so a later push
        to the same PR can trigger a fresh review."""
        import raven.server as _srv
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            self._setup_raven_only(mc)
            mock_review.return_value = {"severity": "low", "summary": "OK", "findings": []}
            _process_pr(mc, self._normalized_payload())
        assert "gitea:owner/repo#42" not in _srv._in_progress_prs

    def test_in_progress_guard_cleared_after_exception(self):
        """Even when processing raises, the in-progress key is released so
        retries aren't blocked forever by a crashed worker."""
        import raven.server as _srv
        mc = self._make_provider()
        mc.add_self_as_reviewer.return_value = None
        mc.fetch_pr_diff.side_effect = RuntimeError("network")
        _process_pr(mc, self._normalized_payload())
        assert "gitea:owner/repo#42" not in _srv._in_progress_prs

    def test_best_effort_failures_increment_error_metric(self):
        """add_self_as_reviewer, dismiss_previous_reviews, and
        add_label_to_pr fail silently at warning level. Each branch now
        also increments raven_errors_total with a distinct type so
        operators can alert on sustained failures (e.g. token scope
        revoked) without scraping logs."""
        from raven.metrics import _counters
        _counters.clear()
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.add_self_as_reviewer.side_effect = RuntimeError("scope lost")
            mc.dismiss_previous_reviews.side_effect = RuntimeError("admin only")
            mc.add_label_to_pr.side_effect = RuntimeError("label missing")
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            # Raven not yet listed — add is attempted (and fails) so metric fires.
            # Gate check (second get_pr_reviews call) returns Raven so that the
            # review still runs and dismiss/label failures can also be exercised.
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                                # auto-add decision
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],              # gate check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],              # sole-reviewer check
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mock_review.return_value = {"severity": "low", "summary": "OK", "findings": []}
            _process_pr(mc, self._normalized_payload())
        keys = list(_counters.keys())
        assert any("self_reviewer_failed" in k for k in keys), keys
        assert any("dismiss_failed" in k for k in keys), keys
        assert any("label_failed" in k for k in keys), keys

    def test_skips_review_when_raven_not_a_reviewer(self):
        """Reviewer-status gate: if Raven isn't listed as a reviewer
        after the auto-add decision, the review doesn't run. Happens
        in fill-gap mode on PRs with existing human reviewers."""
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "alice"}, "state": "COMMENTED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""

        with (
            patch("raven.server.RAVEN_REVIEW_MODE", "gap"),
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mock_review.assert_not_called()
        mc.submit_review.assert_not_called()

    def test_runs_review_when_raven_is_reviewer(self):
        """When Raven is already a listed reviewer, review runs even in
        fill-gap mode with humans present (someone manually added
        Raven)."""
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "alice"}, "state": "COMMENTED"},
            {"user": {"login": "Raven"}, "state": "COMMENTED"},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}

        with (
            patch("raven.server.RAVEN_REVIEW_MODE", "gap"),
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mock_review.assert_called_once()

    def test_runs_review_when_all_prs_mode_auto_adds_raven(self):
        """In RAVEN_REVIEW_MODE="all" mode, Raven auto-adds itself
        even with human reviewers — so the gate passes after auto-add."""
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        # After auto-add, the second get_pr_reviews call (the gate) must
        # show Raven as listed. Simulate that by returning human-only the
        # first time (auto-add decision) and human+Raven the second time
        # (gate check).
        mc.get_pr_reviews.side_effect = [
            [{"user": {"login": "alice"}, "state": "COMMENTED"}],                          # auto-add check
            [{"user": {"login": "alice"}, "state": "COMMENTED"}, {"user": {"login": "Raven"}, "state": "COMMENTED"}],  # gate check
            [{"user": {"login": "alice"}, "state": "COMMENTED"}, {"user": {"login": "Raven"}, "state": "APPROVED"}],   # sole-reviewer merge check (later)
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}

        with (
            patch("raven.server.RAVEN_REVIEW_MODE", "all"),
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())

        mc.add_self_as_reviewer.assert_called_once()
        mock_review.assert_called_once()


class TestClassifiedFailureComment:
    """When a review fails, _process_pr must post a comment that NAMES the
    cause in plain language (timeout / usage cap / rate-limit / auth / …)
    and increment raven_review_failures_total{reason, repo} — instead of
    the old opaque 'Internal error' for every cause. Retry happens inside
    reviewer.py; server.py only classifies what bubbles up. Secrets must
    never reach the comment (no raw str(e)).
    """

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        return mc

    def _run_with_review_error(self, exc):
        from raven.metrics import _counters
        _counters.clear()
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff", side_effect=exc),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._normalized_payload())
        # The failure comment is the only post_pr_comment in this flow.
        assert mc.post_pr_comment.called, "no failure comment posted"
        body = mc.post_pr_comment.call_args[0][2]
        return body, dict(_counters)

    def _failure_metric_reason(self, counters):
        keys = [k for k in counters if k.startswith("raven_review_failures_total")]
        assert len(keys) == 1, f"expected one failure metric, got {keys}"
        key = keys[0]
        assert 'repo="owner/repo"' in key
        import re
        m = re.search(r'reason="([^"]+)"', key)
        return m.group(1)

    def test_timeout_comment_names_cause_and_timeout_value(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("claude CLI timed out after 600s", reason="timeout"))
        assert body.startswith("🦅")
        low = body.lower()
        assert "timed out" in low or "timeout" in low
        # Actionable: the configured RAVEN_AI_TIMEOUT and the knob to raise.
        assert "RAVEN_AI_TIMEOUT" in body
        assert self._failure_metric_reason(counters) == "timeout"

    def test_usage_limit_comment_says_will_retry_on_next_trigger(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("usage limit reached", reason="usage_limit"))
        low = body.lower()
        assert "usage" in low or "limit" in low
        # Tells the operator it recovers on the next push/trigger.
        assert "next" in low
        assert self._failure_metric_reason(counters) == "usage_limit"

    def test_rate_limit_comment_says_retried(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("429 too many requests", reason="rate_limit"))
        low = body.lower()
        assert "rate" in low
        assert "retr" in low  # "retried" / "retry"
        assert self._failure_metric_reason(counters) == "rate_limit"

    def test_backend_5xx_comment_says_retried(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("503 overloaded", reason="backend_5xx"))
        assert "retr" in body.lower()
        assert self._failure_metric_reason(counters) == "backend_5xx"

    def test_auth_comment_classified(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("401 invalid api key", reason="auth"))
        assert "auth" in body.lower() or "credential" in body.lower()
        assert self._failure_metric_reason(counters) == "auth"

    def test_unknown_reason_falls_back_to_generic(self):
        from raven.ai.base import AIError
        body, counters = self._run_with_review_error(
            AIError("weird", reason="unknown"))
        assert body.startswith("🦅")
        assert self._failure_metric_reason(counters) == "unknown"

    def test_non_aierror_exception_classified_as_unknown(self):
        # A plain exception (not from the backend) still gets a metric +
        # the generic comment — reason "unknown".
        body, counters = self._run_with_review_error(RuntimeError("network blew up"))
        assert body.startswith("🦅")
        assert self._failure_metric_reason(counters) == "unknown"

    def test_comment_does_not_leak_exception_detail_with_credentials(self):
        # The message carries a credential-looking URL; it must NOT be
        # interpolated verbatim into the user-facing comment.
        from raven.ai.base import AIError
        secret = "https://user:supersecretpassword@proxy.internal/v1"
        body, _ = self._run_with_review_error(
            AIError(f"AI backend error: connection to {secret} failed", reason="backend_5xx"))
        assert "supersecretpassword" not in body
        assert secret not in body

    def test_truncated_diff_comment_is_actionable_and_blocks_review(self):
        # A provider DiffTruncatedError (diff too large → partial) must fail
        # CLOSED before any review/approve/merge, and the operator comment
        # must name the cause + fix (split PR / raise the limit), NOT the
        # opaque "internal error". Classified metric reason: diff_truncated.
        from raven.metrics import _counters
        from raven.providers import DiffTruncatedError
        _counters.clear()
        mc = self._make_provider()
        mc.fetch_pr_diff.side_effect = DiffTruncatedError(
            "Bitbucket DC returned a truncated diff for PR #42")
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._normalized_payload())
        # Fail closed: never reached the AI review, never approved/merged.
        mock_review.assert_not_called()
        mc.submit_review.assert_not_called()
        mc.merge_pr.assert_not_called()
        # Actionable, classified comment — not the generic internal error.
        assert mc.post_pr_comment.called, "no failure comment posted"
        body = mc.post_pr_comment.call_args[0][2]
        assert body.startswith("🦅")
        low = body.lower()
        assert "truncat" in low or "too large" in low
        assert "internal error" not in low
        assert self._failure_metric_reason(dict(_counters)) == "diff_truncated"

    def test_dedup_cleared_so_retry_can_reattempt(self):
        # Behaviour preserved from the old handler: the dedup entry is
        # cleared on failure so a webhook retry can re-run the review.
        import time
        import raven.server as _srv
        from raven.ai.base import AIError
        mc = self._make_provider()
        key = "gitea:owner/repo#42@abc123"
        with _srv._recent_prs_lock:
            _srv._recent_prs[key] = time.time()
        with (
            patch("raven.server.review_diff", side_effect=AIError("t", reason="timeout")),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._normalized_payload())
        assert key not in _srv._recent_prs


class TestClassifiedFailureLogNoise:
    """A CLASSIFIED failure (diff_truncated, timeout, …) is an expected
    fail-closed condition that already gets a per-reason metric and an
    actionable PR comment — it must not masquerade as a crash. Classified
    reasons log one WARNING without a traceback and do NOT increment
    raven_errors_total (the signal operators alert on for real bugs).
    Only reason="unknown" keeps the ERROR + traceback + raven_errors_total
    behaviour. Applies to both _process_pr and _process_comment.
    """

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _pr_payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _comment_payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "comment_body": "@Raven explain", "comment_user": "alice",
            "comment_id": 999, "file_path": "", "line": 0,
            "_is_mention": True,
        }

    def _pr_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        return mc

    def _run_pr_failure(self, mc, caplog):
        from raven.metrics import _counters
        _counters.clear()
        with (
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            caplog.at_level("WARNING", logger="raven.server"),
        ):
            _process_pr(mc, self._pr_payload())
        return dict(_counters)

    def _run_comment_failure(self, exc, caplog):
        from raven.metrics import _counters
        _counters.clear()
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with (
            patch("raven.server.respond_to_comment", side_effect=exc),
            caplog.at_level("WARNING", logger="raven.server"),
        ):
            _process_comment(mc, self._comment_payload())
        return dict(_counters)

    @staticmethod
    def _errors(caplog):
        return [r for r in caplog.records if r.levelname == "ERROR"]

    @staticmethod
    def _has_counter(counters, name, label):
        return any(k.startswith(name) and label in k for k in counters)

    def test_truncated_diff_logs_warning_without_traceback(self, caplog):
        from raven.providers import DiffTruncatedError
        mc = self._pr_provider()
        mc.fetch_pr_diff.side_effect = DiffTruncatedError("diff too large for PR #42")
        with patch("raven.server.review_diff"):
            counters = self._run_pr_failure(mc, caplog)
        assert self._errors(caplog) == [], (
            "classified failure must not log at ERROR")
        warnings = [r for r in caplog.records
                    if r.levelname == "WARNING" and "diff_truncated" in r.getMessage()]
        assert warnings, "expected a WARNING naming the classified reason"
        assert all(r.exc_info is None for r in warnings), (
            "classified failure must not carry a traceback")
        assert not self._has_counter(counters, "raven_errors_total", 'type="unhandled"')
        # The classified metric and the operator comment are untouched.
        assert self._has_counter(counters, "raven_review_failures_total",
                                 'reason="diff_truncated"')
        assert mc.post_pr_comment.called

    def test_classified_ai_failure_logs_warning_without_traceback(self, caplog):
        from raven.ai.base import AIError
        mc = self._pr_provider()
        with patch("raven.server.review_diff",
                   side_effect=AIError("timed out", reason="timeout")):
            counters = self._run_pr_failure(mc, caplog)
        assert self._errors(caplog) == []
        assert any(r.levelname == "WARNING" and "timeout" in r.getMessage()
                   for r in caplog.records)
        assert not self._has_counter(counters, "raven_errors_total", 'type="unhandled"')

    def test_unknown_pr_failure_keeps_error_log_and_unhandled_metric(self, caplog):
        mc = self._pr_provider()
        with patch("raven.server.review_diff", side_effect=RuntimeError("boom")):
            counters = self._run_pr_failure(mc, caplog)
        errors = self._errors(caplog)
        assert len(errors) == 1
        assert errors[0].exc_info is not None, (
            "unknown failure must keep the full traceback")
        assert self._has_counter(counters, "raven_errors_total", 'type="unhandled"')
        assert self._has_counter(counters, "raven_review_failures_total",
                                 'reason="unknown"')

    def test_classified_comment_failure_logs_warning_without_traceback(self, caplog):
        from raven.ai.base import AIError
        counters = self._run_comment_failure(
            AIError("usage cap hit", reason="usage_limit"), caplog)
        assert self._errors(caplog) == []
        warnings = [r for r in caplog.records
                    if r.levelname == "WARNING" and "usage_limit" in r.getMessage()]
        assert warnings, "expected a WARNING naming the classified reason"
        assert all(r.exc_info is None for r in warnings)
        assert not self._has_counter(counters, "raven_errors_total",
                                     'type="comment_response_failed"')
        assert self._has_counter(counters, "raven_review_failures_total",
                                 'reason="usage_limit"')

    def test_unknown_comment_failure_keeps_error_log_and_metric(self, caplog):
        counters = self._run_comment_failure(RuntimeError("boom"), caplog)
        errors = self._errors(caplog)
        assert len(errors) == 1
        assert errors[0].exc_info is not None
        assert self._has_counter(counters, "raven_errors_total",
                                 'type="comment_response_failed"')
        assert self._has_counter(counters, "raven_review_failures_total",
                                 'reason="unknown"')


class TestWaitForCi:
    def test_initial_delay_skipped_on_terminal_fast_path(self):
        """Fast path: if the first probe already returns a terminal
        state, don't sleep at all. Saves 10s on no-CI repos and re-
        reviews where CI has already finished before Raven's review."""
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "success"
        with patch("raven.server.time.sleep") as mock_sleep:
            _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        mock_sleep.assert_not_called()

    def test_initial_delay_applied_when_pending(self):
        """Slow path: if the first probe returns pending, keep the
        original 10s settle delay before polling again."""
        gitea = MagicMock()
        # First probe pending → enter delay+poll. Second probe success.
        gitea.get_commit_status.side_effect = ["pending", "success"]
        with patch("raven.server.time.sleep") as mock_sleep:
            _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        # Exactly one 10s sleep (the initial delay). Second probe was
        # success so no per-iteration sleep.
        assert mock_sleep.call_args_list[0][0][0] == 10

    def test_returns_on_success(self):
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "success"
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "success"

    def test_returns_on_failure(self):
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "failure"
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "failure"

    def test_returns_when_no_ci(self):
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "none"
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "none"

    def test_no_ci_takes_fast_path(self):
        """Regression guard: a repo with no CI must not force a 10s wait
        before falling through to merge."""
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "none"
        with patch("raven.server.time.sleep") as mock_sleep:
            _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        mock_sleep.assert_not_called()
        # Only one probe — no reason to poll when there's no CI
        gitea.get_commit_status.assert_called_once()

    def test_polls_until_success(self):
        gitea = MagicMock()
        gitea.get_commit_status.side_effect = ["pending", "pending", "success"]
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "success"
        assert gitea.get_commit_status.call_count == 3

    def test_returns_pending_on_timeout(self):
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "pending"
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=20)
        assert result == "pending"

    def test_require_ci_treats_initial_none_as_pending(self, monkeypatch):
        """audit #6: with RAVEN_REQUIRE_CI on, a `none` status on the
        fast-path initial probe must NOT short-circuit to `none` — it is
        treated as pending so we wait for a CI system to register. With a
        small timeout + patched sleep it falls through to the `pending`
        return rather than merging immediately."""
        monkeypatch.setenv("RAVEN_REQUIRE_CI", "1")
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "none"
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=20)
        assert result != "none"
        assert result == "pending"

    def test_require_ci_treats_polled_none_as_pending(self, monkeypatch):
        """audit #6: the `none`→pending coercion must also apply on the
        slow-path poll, not only the initial probe. A `none` arriving
        mid-poll keeps waiting instead of merging; a later real terminal
        state (success) is still honoured."""
        monkeypatch.setenv("RAVEN_REQUIRE_CI", "1")
        gitea = MagicMock()
        # pending → enter poll loop; none mid-poll must not terminate;
        # finally a real success terminates.
        gitea.get_commit_status.side_effect = ["pending", "none", "success"]
        with patch("raven.server.time.sleep"):
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "success"
        assert gitea.get_commit_status.call_count == 3

    def test_require_ci_unset_keeps_none_fast_path(self, monkeypatch):
        """Guard: with RAVEN_REQUIRE_CI unset, an initial `none` still
        returns `none` immediately (existing no-CI-repo behavior)."""
        monkeypatch.delenv("RAVEN_REQUIRE_CI", raising=False)
        gitea = MagicMock()
        gitea.get_commit_status.return_value = "none"
        with patch("raven.server.time.sleep") as mock_sleep:
            result = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=60)
        assert result == "none"
        mock_sleep.assert_not_called()
        gitea.get_commit_status.assert_called_once()

    def test_warning_status_never_merges(self):
        """Pin the audit's `warning` sub-claim as already fail-safe:
        `warning` is not a terminal state in _wait_for_ci, so it polls to
        timeout and returns `pending`; _do_merge then refuses to merge on
        `pending`. (The audit claimed `warning` falls through to merge —
        it does not.)"""
        gitea = MagicMock()
        gitea.name = "gitea"
        gitea.get_commit_status.return_value = "warning"
        with patch("raven.server.time.sleep"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAVEN_GITEA_AUTO_MERGE", None)
            status = _wait_for_ci(gitea, "owner/repo", "abc123", timeout=20)
            assert status == "pending"
            _do_merge(gitea, "owner/repo", 7, "title", "url", {"verdict": "approve"},
                      "abc123", "squash")
        gitea.merge_pr.assert_not_called()


class TestShutdownExecutor:
    def test_shutdown_registered_via_atexit(self):
        """Regression guard that actually catches deletion of the
        ``_register_atexit(_shutdown_executor)`` line.

        We can't inspect CPython's atexit registry (``_exithandlers``
        doesn't exist in Python 3; ``unregister`` returns None and
        doesn't decrement ``_ncallbacks`` — slots are nulled, not
        removed). Instead server.py records its registrations in
        ``_ATEXIT_HOOKS`` so tests can assert on exactly what was
        handed to atexit.
        """
        import raven.server as _srv
        assert _shutdown_executor in _srv._ATEXIT_HOOKS, (
            "_shutdown_executor missing from _ATEXIT_HOOKS — the "
            "_register_atexit(_shutdown_executor) line was removed"
        )

    def test_register_atexit_helper_calls_atexit_register(self):
        """Membership in _ATEXIT_HOOKS is only useful if _register_atexit
        actually hands the function to atexit as well. Asserts the
        helper's contract directly so a refactor that drops the
        atexit.register call (keeping only the list append) fails here."""
        import atexit as _atexit
        import raven.server as _srv
        sentinel = lambda: None  # noqa: E731
        with patch.object(_atexit, "register") as mock_register:
            _srv._register_atexit(sentinel)
        mock_register.assert_called_once_with(sentinel)
        assert sentinel in _srv._ATEXIT_HOOKS
        # Undo the append so we don't pollute state for later tests.
        _srv._ATEXIT_HOOKS.remove(sentinel)

    def test_shutdown_cancels_queued_futures(self):
        """The actual behaviour we care about: ``cancel_futures=True``
        drops work that hasn't started so it never runs during
        interpreter shutdown. Exercises the code path by blocking the
        single worker and queueing more tasks behind it, then asserting
        the queued ones are marked cancelled."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import raven.server as _srv
        real = _srv.executor
        try:
            _srv.executor = ThreadPoolExecutor(max_workers=1)
            gate = threading.Event()
            # Fill the single worker with a task blocked on the gate.
            blocking = _srv.executor.submit(gate.wait, timeout=5)
            # Queue several tasks behind it that should never start.
            queued = [
                _srv.executor.submit(lambda: "should not run")
                for _ in range(3)
            ]
            _shutdown_executor()
            # Let the blocking task finish so threads can join cleanly.
            gate.set()
            blocking.result(timeout=5)
            # The queued tasks must all be cancelled.
            for fut in queued:
                assert fut.cancelled(), (
                    f"expected queued future to be cancelled, got {fut!r}"
                )
            # And the pool must refuse new submissions.
            import pytest
            with pytest.raises(RuntimeError):
                _srv.executor.submit(lambda: None)
        finally:
            _srv.executor = real

    def test_shutdown_terminates_claude_subprocesses(self):
        """The executor shutdown alone doesn't unblock workers that are
        mid-Claude-call — those are stuck in proc.communicate(). The
        shutdown hook must also terminate tracked Claude subprocesses so
        gunicorn's graceful timeout isn't spent waiting for inference
        whose result is discarded on exit."""
        import raven.server as _srv
        from concurrent.futures import ThreadPoolExecutor

        real = _srv.executor
        try:
            _srv.executor = ThreadPoolExecutor(max_workers=1)
            with patch("raven.server.terminate_active_processes") as mock_term:
                _shutdown_executor()
            mock_term.assert_called_once()
        finally:
            _srv.executor = real

    def test_shutdown_drains_ci_wait_executor(self):
        """CI-wait tasks (blocking on time.sleep between polls) must be
        dropped on shutdown so gunicorn's graceful timeout isn't
        consumed by polling work whose merge decision is no longer
        relevant. Queued tasks are cancelled via cancel_futures=True."""
        import raven.server as _srv
        from concurrent.futures import ThreadPoolExecutor

        real_main = _srv.executor
        real_ci = _srv.ci_wait_executor
        try:
            _srv.executor = ThreadPoolExecutor(max_workers=1)
            # Replace ci_wait_executor with a real pool that we can
            # observe. Block its one worker so the queued tasks stay
            # queued and we can verify they get cancelled.
            _srv.ci_wait_executor = ThreadPoolExecutor(max_workers=1)
            import threading as _th
            gate = _th.Event()
            blocking = _srv.ci_wait_executor.submit(gate.wait)
            queued = [
                _srv.ci_wait_executor.submit(lambda: "should not run")
                for _ in range(3)
            ]
            with patch("raven.server.terminate_active_processes"):
                _shutdown_executor()
            gate.set()
            blocking.result(timeout=5)
            for fut in queued:
                assert fut.cancelled(), f"expected cancelled future, got {fut!r}"
            import pytest as _pt
            with _pt.raises(RuntimeError):
                _srv.ci_wait_executor.submit(lambda: None)
        finally:
            _srv.executor = real_main
            _srv.ci_wait_executor = real_ci


class TestCiWaitDispatch:
    """The merge phase dispatches through ``ci_wait_executor`` so review
    workers aren't pinned in time.sleep for the full CI wait. Verifies
    the dispatch happens (rather than calling _do_merge inline) and
    that unhandled exceptions in the wait pool are logged rather than
    silently swallowed by the Future."""

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/x.py b/x.py\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 7}
        mc.add_label_to_pr.return_value = None
        mc.get_authenticated_user.return_value = "Raven"
        # Raven not yet listed → auto-add fires. Gate check (2nd call) sees Raven.
        # Sole-reviewer merge check (3rd call) also sees Raven APPROVED.
        mc.get_pr_reviews.side_effect = [
            [],                                                              # auto-add check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # gate check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # sole-reviewer check
        ]
        mc.get_pr_requested_reviewers.return_value = []
        return mc

    def _payload(self):
        return {
            "repo": "owner/repo", "pr_number": 42, "pr_title": "x",
            "pr_url": "http://x", "head_sha": "abc123",
        }

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def test_process_pr_submits_merge_to_ci_wait_executor(self):
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.review_diff", return_value={
                "severity": "low", "summary": "ok", "findings": []}),
            patch("raven.server.notify"),
        ):
            mock_exec.submit.return_value = MagicMock()
            _process_pr(mc, self._payload())

        # The merge must be submitted to the CI-wait pool, not called
        # inline. First positional arg is the callable (_do_merge).
        mock_exec.submit.assert_called_once()
        call_args = mock_exec.submit.call_args[0]
        assert call_args[0] is _safe_do_merge
        assert call_args[2] == "owner/repo"
        assert call_args[3] == 42
        # Merge is NOT called on the provider here — it'll happen
        # inside the wait-pool task when the pool actually runs it.
        mc.merge_pr.assert_not_called()

    def test_log_future_exception_surfaces_error_with_repo_label(self):
        """An exception raised inside a ci_wait_executor task would
        otherwise sit in the Future forever, since the caller never
        reads the result. ``_log_future_exception`` is attached as a
        done-callback to convert that into a log line + metric tagged
        with the originating repo (passed via functools.partial at the
        submit site) so operators can tell which repo's merges are
        failing from ``raven_errors_total``."""
        from concurrent.futures import Future
        import raven.server as _srv

        boom = RuntimeError("boom")
        fut: Future = Future()
        fut.set_exception(boom)

        with patch("raven.server.logger") as mock_log, \
             patch("raven.server.inc") as mock_inc:
            _srv._log_future_exception(fut, repo="owner/repo")

        mock_log.error.assert_called_once()
        msg = mock_log.error.call_args[0][0]
        assert "Unhandled exception" in msg
        mock_inc.assert_called_once()
        name, labels = mock_inc.call_args[0]
        assert name == "raven_errors_total"
        assert labels["repo"] == "owner/repo"

    def test_log_future_exception_no_op_on_success(self):
        from concurrent.futures import Future
        import raven.server as _srv

        fut: Future = Future()
        fut.set_result("all good")

        with patch("raven.server.logger") as mock_log, \
             patch("raven.server.inc") as mock_inc:
            _srv._log_future_exception(fut, repo="owner/repo")

        mock_log.error.assert_not_called()
        mock_inc.assert_not_called()

    def test_log_future_exception_silent_on_cancelled_future(self):
        """``fut.exception()`` on a cancelled future raises
        ``CancelledError``, which is a ``BaseException`` subclass since
        Python 3.8 and would escape ``except Exception``. Cancellation
        fires whenever ``_shutdown_executor`` drains queued tasks via
        ``cancel_futures=True`` — that's expected, not an error. The
        ``fut.cancelled()`` guard avoids surfacing a scary traceback
        every time the service shuts down cleanly."""
        from concurrent.futures import Future
        import raven.server as _srv

        fut: Future = Future()
        fut.cancel()
        # Force the future to the CANCELLED state (not CANCELLED_AND_NOTIFIED).
        # Either state returns True from fut.cancelled(); the guard handles both.

        with patch("raven.server.logger") as mock_log, \
             patch("raven.server.inc") as mock_inc:
            _srv._log_future_exception(fut, repo="owner/repo")

        mock_log.error.assert_not_called()
        mock_inc.assert_not_called()


class TestProcessPrAdvisoryMode:
    """Advisory mode reshapes _process_pr's post-submit flow:
      - Reviewer-listed gate bypassed (Raven engages on every webhook).
      - submit_review called with comment_only=True.
      - Body uses the 'advisory' header.
      - Auto-merge dispatch + reviewer-state checks skipped after submit.

    Uses monkeypatch.setattr on the module-level RAVEN_REVIEW_MODE
    constant rather than reload(). Reload would create a fresh
    _recent_prs / _previous_diffs dict, decoupling from the references
    imported at module top of this test file — and pollute other test
    classes' fixtures that rely on those references.
    """

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/x.py b/x.py\n+x = 1\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 42, "inline_comments": []}
        mc.add_label_to_pr.return_value = None
        # Empty reviewer lists — in all/gap mode this trips the gate
        # and returns early. Advisory mode must bypass.
        mc.get_pr_reviews.return_value = []
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_authenticated_user.return_value = "raven-bot"
        return mc

    def _payload(self):
        return {
            "repo": "owner/repo", "pr_number": 7, "pr_title": "x",
            "pr_url": "http://x", "head_sha": "abc123",
        }

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def test_advisory_mode_proceeds_past_gate_and_uses_comment_only(self, monkeypatch):
        """Gate bypass + comment_only kwarg + advisory body header +
        no auto-merge dispatch — all in one end-to-end run."""
        monkeypatch.setattr("raven.server.RAVEN_REVIEW_MODE", "advisory")
        mc = self._make_provider()
        with patch("raven.server.review_diff", return_value={
                "severity": "low", "summary": "ok", "findings": []}), \
             patch("raven.server.notify"), \
             patch("raven.server.ci_wait_executor") as mock_exec:
            _process_pr(mc, self._payload())

        # Gate bypass: submit_review reached even though reviewer lists are empty.
        mc.submit_review.assert_called_once()
        call = mc.submit_review.call_args
        assert call.kwargs.get("comment_only") is True
        # Body uses the advisory header.
        body_arg = call.kwargs.get("body") or call.args[2]
        assert "Raven Recommendation" in body_arg
        # Advisory mode never reaches the merge dispatch.
        mock_exec.submit.assert_not_called()

    def test_advisory_mode_bypasses_gate_when_raven_not_listed(self, monkeypatch):
        """Specifically isolate the gate-bypass: confirm advisory mode
        does NOT short-circuit with 'not_reviewer' even though
        get_pr_reviews returns no raven entry."""
        monkeypatch.setattr("raven.server.RAVEN_REVIEW_MODE", "advisory")
        mc = self._make_provider()
        with patch("raven.server.review_diff", return_value={
                "severity": "low", "summary": "ok", "findings": []}), \
             patch("raven.server.notify"), \
             patch("raven.server.inc") as mock_inc, \
             patch("raven.server.ci_wait_executor"):
            _process_pr(mc, self._payload())

        # raven_reviews_skipped_total NOT incremented for not_reviewer in advisory mode.
        skipped_calls = [
            c for c in mock_inc.call_args_list
            if c.args and c.args[0] == "raven_reviews_skipped_total"
            and c.args[1].get("reason") == "not_reviewer"
        ]
        assert not skipped_calls


class TestSafeDoMerge:
    """``_safe_do_merge`` restores the user-visible error path that used
    to live in ``_process_pr``'s outer try/except when the merge was
    synchronous. Without it, dispatching ``_do_merge`` to
    ``ci_wait_executor`` made unexpected merge-phase failures silent
    from the user's perspective (review posted, but no indication that
    the merge never happened)."""

    def setup_method(self):
        _recent_prs.clear()

    def test_wraps_do_merge_on_success(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        review = {"severity": "low", "summary": "ok", "findings": []}

        with patch("raven.server._do_merge") as mock_merge, \
             patch("raven.server.inc") as mock_inc:
            _safe_do_merge(mc, "owner/repo", 42, "t", "u", review, "abc", "squash")

        mock_merge.assert_called_once()
        mock_inc.assert_not_called()
        mc.post_pr_comment.assert_not_called()

    def test_unexpected_exception_posts_user_comment(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        review = {"severity": "low", "summary": "ok", "findings": []}

        with patch("raven.server._do_merge", side_effect=RuntimeError("boom")), \
             patch("raven.server.inc") as mock_inc, \
             patch("raven.server.logger") as mock_log:
            _safe_do_merge(mc, "owner/repo", 42, "t", "u", review, "abc", "squash")

        # User-visible comment so reviewers see something went wrong
        mc.post_pr_comment.assert_called_once()
        comment_body = mc.post_pr_comment.call_args[0][2]
        assert "Internal error during merge phase" in comment_body
        # Metric tagged with the real repo, not "unknown"
        name, labels = mock_inc.call_args[0]
        assert name == "raven_errors_total"
        assert labels["type"] == "merge_unhandled"
        assert labels["repo"] == "owner/repo"
        # Logged with exc_info so operators get a traceback
        mock_log.error.assert_called_once()
        assert mock_log.error.call_args[1].get("exc_info") is True

    def test_unexpected_exception_clears_dedup_for_retry(self):
        """Dedup entry must be cleared so a webhook retry can re-attempt
        the review + merge. Matches the old _process_pr outer handler
        behaviour."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        review = {"severity": "low", "summary": "ok", "findings": []}
        # Simulate a pre-existing dedup entry for this PR (SHA-aware key)
        _recent_prs["gitea:owner/repo#42@abc"] = 1.0

        with patch("raven.server._do_merge", side_effect=RuntimeError("boom")):
            _safe_do_merge(mc, "owner/repo", 42, "t", "u", review, "abc", "squash")

        assert "gitea:owner/repo#42@abc" not in _recent_prs

    def test_post_comment_failure_does_not_mask_original_error(self):
        """If the fallback ``post_pr_comment`` itself fails (e.g. API
        outage), the safety wrapper must still return cleanly — the log
        line and metric are already emitted."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.post_pr_comment.side_effect = Exception("API down")
        review = {"severity": "low", "summary": "ok", "findings": []}

        with patch("raven.server._do_merge", side_effect=RuntimeError("boom")), \
             patch("raven.server.inc"):
            # Must not raise
            _safe_do_merge(mc, "owner/repo", 42, "t", "u", review, "abc", "squash")


class TestFetchRules:
    def _provider(self, **kwargs):
        mc = MagicMock(spec=GitProvider)
        for k, v in kwargs.items():
            getattr(mc, k).return_value = v
        return mc

    def test_empty_when_rules_dir_missing(self):
        mc = self._provider(list_directory=[])
        assert _fetch_rules(mc, "owner/repo", "abc123") == {}
        mc.fetch_file.assert_not_called()

    def test_filters_non_markdown(self):
        mc = self._provider(list_directory=[
            ".claude/rules/a.md",
            ".claude/rules/b.md",
            ".claude/rules/NOTES.txt",
            ".claude/rules/image.png",
        ])
        mc.fetch_file.side_effect = lambda repo, p, ref="HEAD": f"content-of-{p}"
        rules = _fetch_rules(mc, "owner/repo", "abc123")
        assert set(rules.keys()) == {".claude/rules/a.md", ".claude/rules/b.md"}

    def test_sorted_output_for_deterministic_prompts(self):
        """Deterministic order helps prompt-cache hits and test reproducibility."""
        mc = self._provider(list_directory=[
            ".claude/rules/z.md",
            ".claude/rules/a.md",
            ".claude/rules/m.md",
        ])
        mc.fetch_file.side_effect = lambda repo, p, ref="HEAD": "x"
        rules = _fetch_rules(mc, "owner/repo", "abc123")
        assert list(rules.keys()) == [
            ".claude/rules/a.md",
            ".claude/rules/m.md",
            ".claude/rules/z.md",
        ]

    def test_list_directory_error_degrades_to_empty(self):
        """Transport error on directory listing must not block review."""
        mc = MagicMock(spec=GitProvider)
        mc.list_directory.side_effect = Exception("500")
        assert _fetch_rules(mc, "owner/repo", "abc123") == {}
        mc.fetch_file.assert_not_called()

    def test_individual_file_fetch_failure_is_partial(self):
        """One failing file doesn't break the others — we get a partial map."""
        mc = MagicMock(spec=GitProvider)
        mc.list_directory.return_value = [
            ".claude/rules/a.md",
            ".claude/rules/b.md",
        ]
        def fetch(repo, p, ref="HEAD"):
            if p.endswith("a.md"):
                raise Exception("transient")
            return "b content"
        mc.fetch_file.side_effect = fetch
        rules = _fetch_rules(mc, "owner/repo", "abc123")
        assert rules == {".claude/rules/b.md": "b content"}

    def test_empty_rules_dir_env_disables_feature(self):
        import raven.server as _srv
        original = _srv.RULES_DIR
        _srv.RULES_DIR = ""
        try:
            mc = MagicMock(spec=GitProvider)
            assert _fetch_rules(mc, "owner/repo", "abc123") == {}
            # Must not even attempt to list when feature is disabled
            mc.list_directory.assert_not_called()
        finally:
            _srv.RULES_DIR = original


class TestHelpers:
    def test_bot_author_detected(self):
        assert _is_bot_author("dependabot") is True
        assert _is_bot_author("github-actions[bot]") is True
        assert _is_bot_author("renovate") is True
        assert _is_bot_author("Alice") is False
        assert _is_bot_author("alice-helper") is False

    def test_bot_endswith_bot_no_longer_matches(self):
        assert _is_bot_author("jacobot") is False

    def test_bot_affix_matches(self):
        """Suffix ``-bot`` and prefix ``bot-`` identify bot-named accounts
        without matching internal segments or standalone 'bot' word chars."""
        assert _is_bot_author("alice-bot") is True
        assert _is_bot_author("bot-worker") is True
        assert _is_bot_author("bot") is True

    def test_bot_affix_does_not_match_internal_segments(self):
        """Previous heuristic used 'bot' in n.split('-') which flagged
        real-human names whose middle segment happened to equal 'bot'.
        Tighter affix check must NOT match these."""
        assert _is_bot_author("alice-bot-fan") is False
        assert _is_bot_author("user-bot-admin") is False
        # Names that merely contain the letters 'bot' anywhere in a segment
        # (but neither prefix nor suffix) must pass through.
        assert _is_bot_author("rob-bot-the-human") is False

    def test_skipped_repo(self):
        with patch.dict(os.environ, {"SKIP_REPOS": "owner/private, owner/legacy"}):
            assert _is_skipped_repo("owner/private") is True
            assert _is_skipped_repo("owner/legacy") is True
            assert _is_skipped_repo("owner/active") is False

    def test_format_comment_high_severity(self):
        review = {
            "severity": "high",
            "summary": "SQL injection risk",
            "findings": [
                {"severity": "high", "message": "Unescaped input in query"},
                {"severity": "low", "message": "Minor style issue"},
            ],
        }
        comment = _format_comment(review)
        assert "🦅 **Raven Review**" in comment
        assert "🔴" in comment
        assert "HIGH" in comment
        assert "SQL injection risk" in comment
        assert "Unescaped input" in comment
        assert "Reviewed by Raven" in comment
        # Footer mentions the model used so operators / readers can see
        # which backend produced the verdict without digging into config.
        from raven.reviewer import RAVEN_AI_MODEL
        assert RAVEN_AI_MODEL in comment

    def test_severity_emoji_scheme_is_red_orange_yellow_no_green(self, monkeypatch):
        # Severity colors: high=red, medium=orange, low=yellow — explicitly NO
        # green anywhere (per request). Guards against regressing to the old
        # green-for-low scheme.
        from raven.severity import default_scale
        scale = default_scale()
        assert {n: scale.emoji(n) for n in ("high", "medium", "low")} == {
            "high": "🔴", "medium": "🟠", "low": "🟡"}
        assert "🟢" not in {scale.emoji(n) for n in scale.ordered()}
        # Position-based, not gate-based: the three colours must not vary
        # with REVIEW_APPROVE_MAX_SEVERITY (Task 1 regression this test
        # could not previously catch — see Task 4 correction).
        for threshold in ("low", "medium", "high"):
            monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", threshold)
            assert {n: default_scale().emoji(n) for n in ("high", "medium", "low")} == {
                "high": "🔴", "medium": "🟠", "low": "🟡"}
        # A low finding renders yellow (not green) in the rendered body.
        body = _format_comment({"severity": "low", "summary": "x",
                                "findings": [{"severity": "low", "message": "m"}]})
        assert "🟢" not in body
        assert "🟡" in body

    def test_format_comment_advisory_mode_swaps_header(self):
        review = {"severity": "medium", "summary": "minor issue", "findings": []}
        body = _format_comment(review, mode="advisory")
        assert "🦅 **Raven Recommendation**" in body
        assert "Advisory only" in body
        assert "**Raven Review**" not in body

    def test_format_comment_advisory_update_mode_header(self):
        review = {"severity": "low", "summary": "looks fine now", "findings": []}
        body = _format_comment(review, mode="advisory_update")
        assert "🦅 **Raven Updated Recommendation**" in body
        assert "Advisory only" in body

    def test_format_comment_default_mode_keeps_review_header(self):
        """Default mode='review' keeps the existing header so the
        non-advisory render path is unchanged."""
        review = {"severity": "low", "summary": "ok", "findings": []}
        body = _format_comment(review)
        assert "🦅 **Raven Review**" in body
        assert "Recommendation" not in body
        assert "Advisory only" not in body

    def test_format_comment_chunked_shows_file_count(self):
        review = {"severity": "low", "summary": "Looks clean", "findings": [], "chunked": True, "chunks_reviewed": 5}
        comment = _format_comment(review)
        assert "5 files" in comment

    def test_format_comment_empty_findings(self):
        review = {"severity": "low", "summary": "No issues", "findings": []}
        comment = _format_comment(review)
        assert "Findings" not in comment
        assert "🦅 **Raven Review**" in comment

    def test_fetch_changed_files(self):
        diff = "diff --git a/server.py b/server.py\n+line\ndiff --git a/utils.py b/utils.py\n+line\n"
        gitea = MagicMock()
        gitea.fetch_file.side_effect = ["def main(): pass\n", "def helper(): pass\n"]
        contents, omitted = _fetch_changed_files(gitea, "owner/repo", "abc123", diff)
        assert "server.py" in contents
        assert "utils.py" in contents
        assert omitted == []
        assert gitea.fetch_file.call_count == 2

    def test_fetch_changed_files_skips_large_files(self):
        diff = "diff --git a/big.py b/big.py\n+line\n"
        gitea = MagicMock()
        gitea.fetch_file.return_value = "x\n" * 600  # Over MAX_FILE_LINES
        contents, omitted = _fetch_changed_files(gitea, "owner/repo", "abc123", diff)
        assert contents == {}
        # The omission is disclosed, with the filename and line count
        assert len(omitted) == 1
        assert "big.py" in omitted[0]
        assert "600 lines" in omitted[0]

    def test_fetch_changed_files_skips_on_error(self):
        diff = "diff --git a/gone.py b/gone.py\n+line\n"
        gitea = MagicMock()
        gitea.fetch_file.side_effect = Exception("404")
        contents, omitted = _fetch_changed_files(gitea, "owner/repo", "abc123", diff)
        assert contents == {}
        # Fetch failures are not cap omissions — not disclosed as such
        assert omitted == []

    def test_fetch_changed_files_reports_files_beyond_file_cap(self, monkeypatch):
        import raven.server as srv
        monkeypatch.setattr(srv, "MAX_FILES", 1)
        diff = "diff --git a/a.py b/a.py\n+line\ndiff --git a/b.py b/b.py\n+line\n"
        gitea = MagicMock()
        gitea.fetch_file.return_value = "ok = 1\n"
        contents, omitted = _fetch_changed_files(gitea, "owner/repo", "abc123", diff)
        assert list(contents) == ["a.py"]
        # Files beyond the cap are never fetched, but ARE disclosed
        assert gitea.fetch_file.call_count == 1
        assert len(omitted) == 1
        assert "b.py" in omitted[0]
        assert "file cap" in omitted[0]

    def test_fetch_changed_files_line_cap_configurable(self, monkeypatch):
        """Raising MAX_FILE_LINES admits files the default cap rejects."""
        import raven.server as srv
        monkeypatch.setattr(srv, "MAX_FILE_LINES", 1000)
        diff = "diff --git a/big.py b/big.py\n+line\n"
        gitea = MagicMock()
        gitea.fetch_file.return_value = "x\n" * 600
        contents, omitted = _fetch_changed_files(gitea, "owner/repo", "abc123", diff)
        assert "big.py" in contents
        assert omitted == []

    def test_file_context_caps_env_overrides(self, monkeypatch):
        """RAVEN_MAX_FILE_LINES / RAVEN_MAX_FILES override the defaults.
        Tests call the resolver directly (same pattern as
        _resolve_review_mode) — the module-level constants are bound at
        import time."""
        from raven.server import _resolve_file_context_caps
        monkeypatch.setenv("RAVEN_MAX_FILE_LINES", "5000")
        monkeypatch.setenv("RAVEN_MAX_FILES", "25")
        assert _resolve_file_context_caps() == (5000, 25)

    def test_file_context_caps_defaults(self, monkeypatch):
        from raven.server import _resolve_file_context_caps
        monkeypatch.delenv("RAVEN_MAX_FILE_LINES", raising=False)
        monkeypatch.delenv("RAVEN_MAX_FILES", raising=False)
        assert _resolve_file_context_caps() == (500, 10)

    def test_max_severity_from_findings_matches_pre_scale_behavior(self):
        """Unrecognised or missing severities must rank LOWEST, not
        highest — this is NOT scale.normalize()/scale.rank() territory
        (those fail CLOSED for model-emitted severities). This helper
        reproduces the pre-scale
        ``SEVERITY_ORDER.get(f.get("severity", "low"), 0)`` behaviour
        exactly, same reasoning as reviewer._cap_findings and the
        carried-candidates cap in _process_pr."""
        from raven.server import _max_severity_from_findings

        # Missing key, empty string, and None all default to "low" on
        # main — none of them may resolve to "high" here.
        assert _max_severity_from_findings([{"message": "no severity key"}]) == "low"
        assert _max_severity_from_findings([{"severity": ""}]) == "low"
        assert _max_severity_from_findings([{"severity": None}]) == "low"
        # An unrecognised (but non-empty) value also ranks lowest.
        assert _max_severity_from_findings([{"severity": "critical"}]) == "low"
        # Known values, and empty list, are unchanged.
        assert _max_severity_from_findings([{"severity": "low"}]) == "low"
        assert _max_severity_from_findings([]) == "low"
        # Whitespace/case variants of a known name still resolve.
        assert _max_severity_from_findings([{"severity": "  HIGH  "}]) == "high"
        # A recognised finding still wins the max over an unrecognised one.
        assert _max_severity_from_findings(
            [{"severity": "critical"}, {"severity": "medium"}]) == "medium"


class TestIncrementalReview:
    """Verify that re-reviews only process changed files."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        return mc

    def test_second_review_with_no_changes_skips(self):
        diff = "diff --git a/f.py b/f.py\n+line\n"
        # Seed the cache with the hash of the same diff chunk
        import hashlib, time as _time
        chunk_hash = hashlib.sha256("diff --git a/f.py b/f.py\n+line\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"f.py": chunk_hash}, findings={"f.py": []})
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = diff
            mc.fetch_file.return_value = ""
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_not_called()

    def test_second_review_with_changes_reviews_only_changed(self):
        # First review cached a different diff for f.py
        import hashlib, time as _time
        old_hash = hashlib.sha256("diff --git a/f.py b/f.py\n+old\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"f.py": old_hash}, findings={"f.py": []})
        new_diff = "diff --git a/f.py b/f.py\n+new\ndiff --git a/g.py b/g.py\n+added\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_called_once()
        # The diff passed to review_diff should only contain the changed files
        reviewed_diff = mock_review.call_args[0][0]
        assert "f.py" in reviewed_diff
        assert "g.py" in reviewed_diff  # new file, also changed

    def test_first_review_uses_full_diff(self):
        diff = "diff --git a/f.py b/f.py\n+line\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_called_once()

    def test_incremental_declares_scope_to_reviewer(self):
        """The incremental branch reviews a changed-files-only delta, so
        it must tell review_diff it's a delta (is_incremental=True) and
        name the unchanged files — otherwise the prompt presents the
        partial diff as the whole PR and the model infers PR-wide
        absence from files it was never shown."""
        import hashlib, time as _time
        old_hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash_a, "b.py": hash_b}, findings={"a.py": [], "b.py": []})
        # a.py changed, b.py unchanged
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        kwargs = mock_review.call_args.kwargs
        assert kwargs["is_incremental"] is True
        assert kwargs["unchanged_files"] == ["b.py"]

    def test_full_review_does_not_declare_incremental_scope(self):
        """First (full) review → no delta framing."""
        diff = "diff --git a/f.py b/f.py\n+line\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        kwargs = mock_review.call_args.kwargs
        assert kwargs.get("is_incremental", False) is False
        assert not kwargs.get("unchanged_files")

    def test_incremental_carries_forward_findings(self):
        """Carried findings from unchanged files appear in the submitted review."""
        import hashlib, time as _time
        old_hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        old_hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        carried_finding = {"severity": "high", "file": "b.py", "line": 10, "message": "bug in b"}
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash_a, "b.py": old_hash_b}, findings={"a.py": [], "b.py": [carried_finding]})
        # Only a.py changed, b.py is unchanged
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            mock_review.return_value = {"severity": "low", "summary": "a looks ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        # The submitted review body should include the carried finding
        submitted_body = mc.submit_review.call_args[0][2]
        assert "bug in b" in submitted_body
        # Verdict should be REQUEST_CHANGES because carried finding is high
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_carried_finding_survives_grounding_filter(self, monkeypatch):
        """Regression for the evidence-grounding filter: carried findings
        are merged in server.py from the CACHE, not from review_diff's
        output, so the grounding filter (which lives inside review_diff
        and drops fresh findings naming an unseen file) must never touch
        them. Here the incremental delta contains ONLY a.py, so the
        review's provided-set is {a.py}; the carried finding names the
        unchanged b.py — not in the delta. If the filter wrongly reached
        carried findings, b.py would look 'ungrounded' and be dropped.

        Uses the REAL review_diff (backend mocked) so the actual filter
        runs, not a hand-mock that bypasses it."""
        from raven.ai.base import CompletionResult
        import hashlib, time as _time

        old_hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        carried = {"severity": "high", "file": "b.py", "line": 10, "message": "carried bug in b"}
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={"a.py": old_hash_a, "b.py": hash_b},
            findings={"a.py": [], "b.py": [carried]},
        )
        # a.py changed, b.py unchanged → incremental delta is a.py only.
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"

        fake_backend = MagicMock()
        fake_backend.name = "claude_cli"
        # Clean fresh review, no dropped_carried → keep everything carried.
        fake_backend.complete.return_value = CompletionResult(
            text='{"severity": "low", "summary": "a ok", "findings": []}',
            input_tokens=0, output_tokens=0, cost_usd=None,
        )
        monkeypatch.setattr("raven.ai._cached_backend", fake_backend)

        mc = self._make_provider()
        mc.fetch_pr_diff.return_value = new_diff
        mc.fetch_file.return_value = ""
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.get_resolved_comment_ids.return_value = set()
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.side_effect = [
            [],                                                        # auto-add check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
        ]
        with patch("raven.server.notify"):
            _process_pr(mc, self._normalized_payload())

        submitted_body = mc.submit_review.call_args[0][2]
        # The carried b.py finding survived the real grounding filter.
        assert "carried bug in b" in submitted_body
        # And it still pins the verdict (high) → not approved.
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_incremental_verdict_max_across_all(self):
        """Verdict is max severity across new + carried findings."""
        import hashlib, time as _time
        hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": hash_a, "b.py": hash_b}, findings={"a.py": [], "b.py": [{"severity": "medium", "file": "b.py", "message": "issue"}]})
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            # New review is low, but carried is medium
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_incremental_clears_findings_for_changed_file(self):
        """When a file is re-reviewed, its old findings are replaced."""
        import hashlib, time as _time
        old_hash = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash}, findings={"a.py": [{"severity": "high", "file": "a.py", "message": "old bug"}]})
        new_diff = "diff --git a/a.py b/a.py\n+fixed\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                        # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
            ]
            # Re-review finds nothing
            mock_review.return_value = {"severity": "low", "summary": "clean", "findings": []}
            _process_pr(mc, self._normalized_payload())
        # Old "old bug" finding should NOT appear
        submitted_body = mc.submit_review.call_args[0][2]
        assert "old bug" not in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is True

    def test_incremental_dismisses_old_reviews(self):
        """Old Raven reviews are dismissed even on incremental reviews."""
        import hashlib, time as _time
        old_hash = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash}, findings={"a.py": []})
        new_diff = "diff --git a/a.py b/a.py\n+new\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            # Low severity dispatches the auto-merge gates inline (the autouse
            # ci_wait_executor fixture is synchronous); without this patch the
            # MagicMock get_commit_status never returns a terminal state and
            # _wait_for_ci really sleeps out its full 300s timeout.
            patch("raven.server._wait_for_ci", return_value="success"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = [
                {"id": 10, "user": {"login": "Raven"}, "state": "REQUEST_CHANGES"},
            ]
            mc.dismiss_previous_reviews.return_value = None
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.dismiss_previous_reviews.assert_called_once_with("owner/repo", 42, "Raven", exclude_id=1)

    def test_incremental_auto_merge_when_clean(self):
        """Auto-merge proceeds on incremental review when all findings resolved."""
        import hashlib, time as _time
        old_hash = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash}, findings={"a.py": []})
        new_diff = "diff --git a/a.py b/a.py\n+new\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server._wait_for_ci", return_value="success"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                              # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # gate check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # sole-reviewer check
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mc.merge_pr.return_value = True
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_called_once()

    def test_incremental_no_merge_when_carried_high(self):
        """Auto-merge blocked when carried findings keep severity above threshold."""
        import hashlib, time as _time
        old_hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        old_hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(timestamp=_time.time(), hashes={"a.py": old_hash_a, "b.py": old_hash_b}, findings={"a.py": [], "b.py": [{"severity": "high", "file": "b.py", "message": "critical"}]})
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = []
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()

    # NOTE: a "2-tuple legacy format" test previously lived here. It was
    # testing in-memory 2-tuple insertion as a stand-in for legacy on-disk
    # entries. With CacheEntry, in-memory 2-tuples can't exist; the legacy
    # 3-tuple path is exercised in TestCachePersistence via the actual
    # JSON load route — see test_load_legacy_3tuple_entries_yields_none_verdict.

    def test_user_resolved_findings_dropped_from_carry_forward(self):
        """When the developer marks an inline finding resolved via the
        platform UI (Gitea /resolve, BB DC "Resolve thread"), the next
        incremental review must drop it from carry-forward — otherwise
        the consolidated verdict re-litigates a dismissed complaint.
        Cache mutation propagates so the resolved finding is also
        removed from the cache write later in _process_pr."""
        import hashlib, time as _time
        hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        # Two findings on unchanged b.py: one user-resolved (comment_id=42),
        # one still open (comment_id=43). After filtering, only 43 carries.
        cached_findings = {
            "a.py": [],
            "b.py": [
                {"severity": "high", "file": "b.py", "line": 10,
                 "message": "developer dismissed this", "comment_id": 42},
                {"severity": "medium", "file": "b.py", "line": 20,
                 "message": "still valid", "comment_id": 43},
            ],
        }
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={"a.py": hash_a, "b.py": hash_b},
            findings=cached_findings,
        )
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        # Developer resolved comment 42 in the UI; provider reports it.
        mc.get_resolved_comment_ids.return_value = {42}
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        # The submitted review body must NOT include the dismissed
        # finding's message, but MUST include the still-valid one.
        submitted_body = mc.submit_review.call_args[0][2]
        assert "developer dismissed this" not in submitted_body
        assert "still valid" in submitted_body
        # The cache must be filtered too — the dismissed entry is gone
        # so the next push doesn't re-check it.
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        kept_ids = {f.get("comment_id") for f in cached_after}
        assert 42 not in kept_ids
        assert 43 in kept_ids

    def test_get_resolved_comment_ids_failure_proceeds_without_filter(self):
        """Provider API failure must not block the review — fall back
        to current behavior (no filtering) so the user can still get
        an incremental review even if the resolved-state lookup is
        temporarily unavailable."""
        import hashlib, time as _time
        hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={"a.py": hash_a, "b.py": hash_b},
            findings={"a.py": [], "b.py": [
                {"severity": "high", "file": "b.py", "line": 10,
                 "message": "carry me", "comment_id": 42},
            ]},
        )
        new_diff = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"
        mc = self._make_provider()
        mc.get_resolved_comment_ids.side_effect = RuntimeError("transient outage")
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = new_diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],
            ]
            mock_review.return_value = {"severity": "low", "summary": "ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        # Finding still carries — no filter applied.
        submitted_body = mc.submit_review.call_args[0][2]
        assert "carry me" in submitted_body


class TestRebaseTolerance:
    """A rebase rewrites a file's diff chunk — ``index`` blob SHAs, ``@@``
    line numbers, surrounding context — without touching the PR's own
    edits. Re-reviewing on that basis regenerates the file's findings,
    and a regenerated finding has no ``comment_id``, which is what the
    user-resolved filter matches on: every resolution on that file is
    lost. So content-equal files are carried, not re-reviewed — and
    because they are carried rather than regenerated, their line numbers
    have to be moved onto the code's new position by hand."""

    # b.py: one hunk at new-side line 7, the PR's edit on line 10.
    B_BEFORE = (
        "diff --git a/b.py b/b.py\n"
        "index e88160e..b394268 100644\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -7,7 +7,7 @@ def g():\n"
        " ctx7\n ctx8\n ctx9\n"
        "-other10\n"
        "+other10_edited_by_pr\n"
        " ctx11\n"
    )
    # Same edit after the base branch grew b.py by 40 lines above it.
    B_AFTER = (B_BEFORE
               .replace("index e88160e..b394268", "index ed765b6..7a4440c")
               .replace("@@ -7,7 +7,7 @@", "@@ -47,7 +47,7 @@"))
    # The SAME edit relocated by the author to a different part of the
    # file: identical +/- lines (so identical content hash) and an
    # identical hunk shape, but landing in different surrounding code.
    # Indistinguishable from B_AFTER on positions alone — only the
    # context tells them apart.
    B_MOVED = (
        "diff --git a/b.py b/b.py\n"
        "index e88160e..cccccc1 100644\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -47,7 +47,7 @@ def somewhere_else():\n"
        " zzz47\n zzz48\n zzz49\n"
        "-other10\n"
        "+other10_edited_by_pr\n"
        " zzz51\n"
    )
    A_OLD = "diff --git a/a.py b/a.py\n@@ -1,1 +1,1 @@\n+old\n"
    A_NEW = "diff --git a/a.py b/a.py\n@@ -1,1 +1,1 @@\n+new\n"

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_file.return_value = ""
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.get_resolved_comment_ids.return_value = set()
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.side_effect = [
            [],                                                   # auto-add check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],  # gate check
        ]
        return mc

    def _seed(self, chunks: dict, findings: dict, verdict=None):
        """Cache a prior review of ``chunks`` (filename -> chunk)."""
        import hashlib, time as _time
        from raven.reviewer import diff_hash, hunk_positions, hunk_context_digests
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={f: hashlib.sha256(c.encode()).hexdigest()
                    for f, c in chunks.items()},
            content_hashes={f: diff_hash(c) for f, c in chunks.items()},
            hunks={f: hunk_positions(c) for f, c in chunks.items()},
            # Populated exactly as a real post-submit write does — a seed
            # missing it would silently exercise the legacy-entry degrade
            # path instead of current behaviour. That path has its own
            # test (test_legacy_entry_without_context_still_remaps).
            hunk_context={f: hunk_context_digests(c) for f, c in chunks.items()},
            findings=findings,
            verdict=verdict,
        )
        return _previous_diffs["gitea:owner/repo#42"]

    def _finding(self, line=10, **kw):
        f = {"severity": "high", "file": "b.py", "line": line,
             "message": "bug on the PR's edited line", "comment_id": 999}
        f.update(kw)
        return f

    def _run(self, diff, review=None):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            mc.fetch_pr_diff.return_value = diff
            mock_review.return_value = review or {
                "severity": "low", "summary": "a ok", "findings": []}
            _process_pr(mc, self._normalized_payload())
        return mc, mock_review

    def test_shifted_file_is_not_re_reviewed(self):
        """The whole point: b.py's own edits are byte-identical, so it
        stays out of the delta and keeps its comment_id-bearing findings."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding()]})
        _, mock_review = self._run(self.A_NEW + self.B_AFTER)
        reviewed = mock_review.call_args.args[0]
        assert "a.py" in reviewed
        assert "b.py" not in reviewed
        assert mock_review.call_args.kwargs["unchanged_files"] == ["b.py"]

    def test_carried_finding_follows_the_shift(self):
        """Carried findings re-post from the cache, so an unshifted line
        would anchor the inline comment to whatever the rebase slid into
        that position. The hunk moved 7 -> 47, so line 10 -> 50."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        mc, _ = self._run(self.A_NEW + self.B_AFTER)
        inline = mc.submit_review.call_args.kwargs["inline_comments"]
        assert [c["line"] for c in inline if c["file"] == "b.py"] == [50]

    def test_shift_is_recorded_in_the_cache(self):
        """The next pass has to diff against the shifted state, not the
        pre-rebase one, or the remap would be applied twice."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        self._run(self.A_NEW + self.B_AFTER)
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert entry.findings["b.py"][0]["line"] == 50
        assert entry.hunks["b.py"] == [(47, 7)]

    def test_comment_id_survives_the_shift(self):
        """The remap must not cost the finding its comment_id — that is
        the whole identity the user-resolved filter and retraction use."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        self._run(self.A_NEW + self.B_AFTER)
        assert _previous_diffs["gitea:owner/repo#42"].findings["b.py"][0]["comment_id"] == 999

    def test_unshifted_rebase_carries_the_finding_untouched(self):
        """Base edits *below* the PR's hunk rewrite the blob SHAs only.
        Nothing to remap — and nothing to re-review either."""
        b_reindexed = self.B_BEFORE.replace("index e88160e..b394268",
                                            "index 9999999..8888888")
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        mc, mock_review = self._run(self.A_NEW + b_reindexed)
        assert "b.py" not in mock_review.call_args.args[0]
        inline = mc.submit_review.call_args.kwargs["inline_comments"]
        assert [c["line"] for c in inline if c["file"] == "b.py"] == [10]

    def test_real_edit_to_a_shifted_file_still_re_reviews(self):
        """Rebase tolerance must not swallow an actual change: a file
        that both moved AND was edited belongs in the delta."""
        b_edited = self.B_AFTER.replace("+other10_edited_by_pr",
                                        "+other10_edited_again")
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding()]})
        _, mock_review = self._run(self.A_NEW + b_edited)
        assert "b.py" in mock_review.call_args.args[0]

    def test_relocated_edit_is_re_reviewed_not_remapped(self):
        """Audit 2026-08-17 MED. The content hash keeps only the +/- line
        bodies, so an author relocating a byte-identical edit elsewhere in
        the same file produces an identical hash — and identical hunk
        geometry, so the remap "succeeds" and simply shifts the finding.
        The relocated code is then never reviewed in its new position.

        That matters because position is what makes a line dangerous: the
        same statement is inert in a dead branch and live on a hot path.
        A first pass could approve it where it was harmless, and a later
        push consolidate the carried approve into a merge.

        Context is the only signal that separates this from the rebase
        case it is deliberately tolerant of, so a hunk whose surrounding
        code changed must fall back to a real re-review."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        _, mock_review = self._run(self.A_NEW + self.B_MOVED)
        assert "b.py" in mock_review.call_args.args[0], (
            "an edit relocated into different surrounding code must be "
            "re-reviewed, not silently carried onto its new line"
        )

    def test_legacy_entry_without_context_still_remaps(self):
        """Entries written before hunk_context existed have nothing to
        compare, so they degrade to the previous behaviour (remap on
        geometry alone) rather than re-reviewing every carried file.

        Deliberately NOT closed by bumping DIFF_HASH_SCHEME: that wipes
        the whole findings cache, and a full re-review regenerates
        findings without their comment_ids — losing exactly the developer
        resolutions this feature exists to preserve. One unprotected push
        per PR is the cheaper trade; the next review writes the field."""
        entry = self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                           {"a.py": [], "b.py": [self._finding(line=10)]})
        entry.hunk_context = {}          # as loaded from an older cache file
        _, mock_review = self._run(self.A_NEW + self.B_AFTER)
        assert "b.py" not in mock_review.call_args.args[0]

    def test_genuine_rebase_still_carries(self):
        """Guard against over-tightening: the rebase case keeps its
        context byte-identical and must still avoid a re-review."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        _, mock_review = self._run(self.A_NEW + self.B_AFTER)
        assert "b.py" not in mock_review.call_args.args[0]

    def test_unmappable_finding_forces_a_re_review(self):
        """Fail-safe: when the hunks can't be matched one-to-one (a base
        edit landing in context range merges two hunks into one), we
        can't prove where the finding's code went — fall back to the
        pre-rebase-tolerance behaviour and re-review the file."""
        b_two_hunks = (
            "diff --git a/b.py b/b.py\n"
            "@@ -7,7 +7,7 @@ def g():\n"
            "-other10\n"
            "+other10_edited_by_pr\n"
            "@@ -60,3 +60,3 @@ def h():\n"
            "-tail\n"
            "+tail_edited\n"
        )
        b_one_hunk = (
            "diff --git a/b.py b/b.py\n"
            "@@ -7,60 +7,60 @@ def g():\n"
            "-other10\n"
            "+other10_edited_by_pr\n"
            "-tail\n"
            "+tail_edited\n"
        )
        self._seed({"a.py": self.A_OLD, "b.py": b_two_hunks},
                   {"a.py": [], "b.py": [self._finding(line=8)]})
        _, mock_review = self._run(self.A_NEW + b_one_hunk)
        assert "b.py" in mock_review.call_args.args[0]

    def test_finding_outside_every_hunk_forces_a_re_review(self):
        """A line that matches no recorded hunk has no delta to shift by,
        so it is not silently left at a position we can't vouch for."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=900)]})
        _, mock_review = self._run(self.A_NEW + self.B_AFTER)
        assert "b.py" in mock_review.call_args.args[0]

    def test_file_less_findings_do_not_block_the_remap(self):
        """PR-wide findings post no inline comment, so they have nothing
        to anchor and must not drag the file into a re-review."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)],
                    "": [{"severity": "low", "message": "PR-wide note"}]})
        _, mock_review = self._run(self.A_NEW + self.B_AFTER)
        assert "b.py" not in mock_review.call_args.args[0]

    def test_rebase_only_push_reviews_nothing(self):
        """Nothing the PR authored changed, so there is no new code to
        review — regenerating the standing findings would only strand
        the developer's resolutions."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": [self._finding(line=10)]})
        mc, mock_review = self._run(self.A_OLD + self.B_AFTER)
        mock_review.assert_not_called()
        mc.submit_review.assert_not_called()
        # …but the shift is still recorded, so the findings stay anchored.
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert entry.findings["b.py"][0]["line"] == 50
        assert entry.hashes["b.py"] != entry.hashes["a.py"]

    def test_rebase_only_push_does_not_dispatch_a_cached_merge(self):
        """The no-changes skip can send a cached approve straight to a
        merge with no fresh review. A rebased head is not the head that
        approval was computed on, so it must not reach that path — the
        raw-chunk hash, not the content hash, is what gates it."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": []}, verdict="approve")
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server._maybe_dispatch_cached_merge") as mock_dispatch,
        ):
            mc.fetch_pr_diff.return_value = self.A_OLD + self.B_AFTER
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_not_called()
        mock_dispatch.assert_not_called()

    def test_untouched_head_still_reaches_the_cached_merge(self):
        """The converse: a byte-identical re-trigger is still the
        no-changes skip, so the standing-approval recovery path is
        unaffected by any of this."""
        self._seed({"a.py": self.A_OLD, "b.py": self.B_BEFORE},
                   {"a.py": [], "b.py": []}, verdict="approve")
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server._maybe_dispatch_cached_merge") as mock_dispatch,
        ):
            mc.fetch_pr_diff.return_value = self.A_OLD + self.B_BEFORE
            _process_pr(mc, self._normalized_payload())
        mock_review.assert_not_called()
        mock_dispatch.assert_called_once()

    def test_legacy_cache_entry_behaves_as_before(self):
        """An entry written before content_hashes existed has nothing to
        compare against — it must fall back to the raw delta, not to
        'everything changed'."""
        import hashlib, time as _time
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={"a.py": hashlib.sha256(self.A_OLD.encode()).hexdigest(),
                    "b.py": hashlib.sha256(self.B_BEFORE.encode()).hexdigest()},
            findings={"a.py": [], "b.py": [self._finding()]},
        )
        _, mock_review = self._run(self.A_NEW + self.B_BEFORE)
        reviewed = mock_review.call_args.args[0]
        assert "a.py" in reviewed and "b.py" not in reviewed


class TestCarriedFindingsRevalidation:
    """Carried findings are re-validated by the incremental review call
    (drop-or-keep via `dropped_carried`) instead of being merged
    verbatim. Failure modes this guards: (a) a push that satisfies a
    finding in a DIFFERENT file used to re-post the stale demand and
    feed its severity into the verdict; (b) file-less ('' bucket)
    findings were carried unconditionally on every pass — immortal.
    Drop is the EXPLICIT action: a missing key, an empty array, or any
    malformed answer keeps everything, so a schema-echoing model can
    never silently erase carried findings."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_resolved_comment_ids.return_value = set()
        mc.retract_finding.return_value = True
        return mc

    # Two-file diff: a.py changed since the cached hashes, b.py stable.
    _NEW_DIFF = "diff --git a/a.py b/a.py\n+new\ndiff --git a/b.py b/b.py\n+stable\n"

    def _seed_cache(self, findings, coverage_gap_files=None):
        import hashlib, time as _time
        hash_a = hashlib.sha256("diff --git a/a.py b/a.py\n+old\n".encode()).hexdigest()
        hash_b = hashlib.sha256("diff --git a/b.py b/b.py\n+stable\n".encode()).hexdigest()
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=_time.time(),
            hashes={"a.py": hash_a, "b.py": hash_b},
            findings=findings,
            coverage_gap_files=coverage_gap_files or [],
        )

    def _run(self, mc, review_result=None, review_side_effect=None):
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server._wait_for_ci", return_value="success"),
        ):
            mc.fetch_pr_diff.return_value = self._NEW_DIFF
            mc.fetch_file.return_value = ""
            if not isinstance(mc.submit_review.side_effect, Exception):
                mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "Raven"}, "state": "APPROVED"},
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            if review_side_effect is not None:
                mock_review.side_effect = review_side_effect
            else:
                mock_review.return_value = review_result
            _process_pr(mc, self._normalized_payload())
        return mock_review

    def test_carried_candidates_passed_to_review_diff(self):
        """Carried findings from unchanged files — including the '' bucket
        — are handed to review_diff for re-validation."""
        b_finding = {"severity": "high", "file": "b.py", "line": 10, "message": "needs a test"}
        fileless = {"severity": "medium", "message": "file-less observation"}
        self._seed_cache({"a.py": [], "b.py": [b_finding], "": [fileless]})
        mc = self._make_provider()
        mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        carried_kwarg = mock_review.call_args.kwargs["carried_findings"]
        assert carried_kwarg == [b_finding, fileless]

    def test_no_carried_candidates_passes_none(self):
        self._seed_cache({"a.py": [], "b.py": []})
        mc = self._make_provider()
        mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        assert mock_review.call_args.kwargs["carried_findings"] is None

    def test_dropped_carried_removed_from_review_and_cache(self):
        """Findings the model explicitly drops disappear from the verdict
        and (after the submit succeeds) from the cache; the rest carry."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        valid = {"severity": "medium", "file": "b.py", "line": 20, "message": "still valid"}
        self._seed_cache({"a.py": [], "b.py": [stale, valid]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" not in submitted_body
        assert "still valid" in submitted_body
        # Severity recompute uses only the kept finding (medium) — not
        # the dropped high — so the verdict is needs_work, not pinned
        # at high.
        assert mc.submit_review.call_args.kwargs["approve"] is False
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        assert [f["message"] for f in cached_after] == ["still valid"]

    def test_dropping_pinning_finding_unblocks_approve(self):
        """The PR #157 failure: a stale carried high pinned the verdict.
        Once the model drops it, the verdict can approve again."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" not in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is True

    def test_fileless_findings_are_not_immortal(self):
        """'' bucket findings go through the same drop-or-keep: when
        dropped they leave the review body AND the cached '' bucket,
        so they can no longer pin the verdict forever."""
        fileless = {"severity": "high", "message": "immortal observation"}
        self._seed_cache({"a.py": [], "b.py": [], "": [fileless]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "immortal observation" not in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is True
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("", []) == []

    def test_missing_dropped_key_keeps_all_carried(self):
        """Fail-safe: a review result without `dropped_carried` (model
        ignored the block, call degraded, chunked path) keeps every
        carried finding — the pre-re-validation behavior."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        fileless = {"severity": "medium", "message": "file-less observation"}
        self._seed_cache({"a.py": [], "b.py": [stale], "": [fileless]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" in submitted_body
        assert "file-less observation" in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is False
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("b.py") == [stale]
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("") == [fileless]

    def test_empty_dropped_carried_keeps_all(self):
        """`dropped_carried: []` — the schema-echo answer a weak model
        produces — must keep everything. Under the old confirm-or-drop
        contract an echoed empty array was a silent drop-ALL that could
        flip the verdict to approve and auto-merge."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": []})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is False
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("b.py") == [stale]

    def test_out_of_range_dropped_ids_keep_all(self):
        """Ids that don't map to any candidate must not drop anything —
        treat the whole answer as unusable and keep everything."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [5]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_boolean_dropped_ids_keep_all(self):
        """JSON true/false are Python bools (int subclass) — they must
        not alias carry_ids 1/0. Server-side guard, independent of the
        reviewer-side validator (mocked out here)."""
        stale = {"severity": "high", "file": "b.py", "line": 10, "message": "stale demand"}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [False]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_gap_markers_excluded_from_revalidation_and_always_carried(self):
        """Coverage-gap ⚠️ markers keep their existing lifecycle: carried
        while the gap file is unchanged, dropped when it changes. They
        are NOT offered to the model for drop-or-keep — re-validation
        must not become a path to erase an active gap signal."""
        marker = {"severity": "medium", "file": "b.py",
                  "message": "⚠️ `b.py` skipped (too large: 99999 lines)"}
        self._seed_cache({"a.py": [], "b.py": [marker]},
                         coverage_gap_files=["b.py"])
        mc = self._make_provider()
        mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                                     "dropped_carried": []})
        # Marker never reached the model …
        assert mock_review.call_args.kwargs["carried_findings"] is None
        # … and still carries.
        submitted_body = mc.submit_review.call_args[0][2]
        assert "⚠️ `b.py` skipped" in submitted_body
        assert mc.submit_review.call_args.kwargs["approve"] is False
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        assert cached_after == [marker]

    def test_structural_gap_marker_flag_detected(self):
        """Markers created since the `gap_marker: True` flag exists are
        detected structurally — no reliance on the ⚠️-prefix shape
        heuristic (kept only as a fallback for pre-flag cached
        markers)."""
        marker = {"severity": "medium", "file": "b.py", "gap_marker": True,
                  "message": "review of b.py failed"}
        self._seed_cache({"a.py": [], "b.py": [marker]},
                         coverage_gap_files=["b.py"])
        mc = self._make_provider()
        mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        assert mock_review.call_args.kwargs["carried_findings"] is None
        submitted_body = mc.submit_review.call_args[0][2]
        assert "review of b.py failed" in submitted_body

    def test_resolved_filter_applies_before_revalidation(self):
        """User-resolved findings are filtered out BEFORE the review call
        so they never enter the drop-or-keep prompt (the model can't
        'keep' a finding the developer already dismissed)."""
        resolved = {"severity": "high", "file": "b.py", "line": 10,
                    "message": "developer dismissed this", "comment_id": 42}
        open_f = {"severity": "medium", "file": "b.py", "line": 20,
                  "message": "still open", "comment_id": 43}
        self._seed_cache({"a.py": [], "b.py": [resolved, open_f]})
        mc = self._make_provider()
        mc.get_resolved_comment_ids.return_value = {42}
        mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        carried_kwarg = mock_review.call_args.kwargs["carried_findings"]
        assert carried_kwarg == [open_f]

    def test_resolution_during_review_dropped_after_call(self):
        """A resolution landing DURING the multi-minute AI call is caught
        by the post-review re-fetch: the finding leaves the verdict and
        the cache even though the model kept it. Without this second
        pass, the carried copy would be re-posted and re-tagged with a
        fresh comment_id, permanently orphaning the user's resolution."""
        f42 = {"severity": "medium", "file": "b.py", "line": 10,
               "message": "kept finding", "comment_id": 42}
        f43 = {"severity": "high", "file": "b.py", "line": 20,
               "message": "resolved mid-review", "comment_id": 43}
        self._seed_cache({"a.py": [], "b.py": [f42, f43]})
        mc = self._make_provider()
        # Pre-review fetch sees nothing; post-review fetch sees 43.
        mc.get_resolved_comment_ids.side_effect = [set(), {43}]
        self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        assert mc.get_resolved_comment_ids.call_count == 2
        submitted_body = mc.submit_review.call_args[0][2]
        assert "kept finding" in submitted_body
        assert "resolved mid-review" not in submitted_body
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        assert [f["comment_id"] for f in cached_after] == [42]

    def test_submit_failure_leaves_cache_untouched(self):
        """ALL cache effects of a pass apply only after submit_review
        succeeds. If the submit fails, the standing platform review
        still shows the old findings — the cache must keep matching it
        (the comment-flow all-retracted backstop counts cached findings;
        a premature wipe could synthesize a flip-to-approve)."""
        stale = {"severity": "high", "file": "b.py", "line": 10,
                 "message": "stale demand", "comment_id": 77}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        mc.submit_review.side_effect = RuntimeError("502 from platform")
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert entry.findings.get("b.py") == [stale]
        mc.retract_finding.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_dropped_finding_thread_retracted(self):
        """A dropped finding's platform thread is resolved via
        provider.retract_finding (mirrors the comment-flow retraction).
        Without it the inline comment stays open forever — on BB DC
        (dismiss is a no-op there) an all-comments-resolved merge check
        would be permanently blocked by the very drop that enabled the
        merge."""
        stale = {"severity": "high", "file": "b.py", "line": 10,
                 "message": "stale demand", "comment_id": 77}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        mc.retract_finding.assert_called_once_with("owner/repo", 42, 77)
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("b.py") == []

    def test_dropped_finding_without_comment_id_no_retract(self):
        """Nothing to resolve when the finding never had an inline
        comment (summary-mode reviews, file-less findings)."""
        fileless = {"severity": "high", "message": "immortal observation"}
        self._seed_cache({"a.py": [], "b.py": [], "": [fileless]})
        mc = self._make_provider()
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        mc.retract_finding.assert_not_called()

    def test_retract_failure_does_not_block_drop(self):
        """Best-effort: a failed thread-resolve logs and continues — the
        drop itself (review body + cache) stands."""
        stale = {"severity": "high", "file": "b.py", "line": 10,
                 "message": "stale demand", "comment_id": 77}
        self._seed_cache({"a.py": [], "b.py": [stale]})
        mc = self._make_provider()
        mc.retract_finding.side_effect = RuntimeError("403")
        self._run(mc, {"severity": "low", "summary": "ok", "findings": [],
                       "dropped_carried": [0]})
        submitted_body = mc.submit_review.call_args[0][2]
        assert "stale demand" not in submitted_body
        assert _previous_diffs["gitea:owner/repo#42"].findings.get("b.py") == []

    def test_concurrent_retraction_not_resurrected(self):
        """The comment flow can retract a finding (under
        _previous_diffs_lock) while the push-flow review is in flight.
        The cache write must filter from the LIVE entry, not write back
        the pre-review snapshot — otherwise the retraction is silently
        undone and persisted."""
        f91 = {"severity": "high", "file": "b.py", "line": 10,
               "message": "race victim", "comment_id": 91}
        f92 = {"severity": "medium", "file": "b.py", "line": 20,
               "message": "survivor", "comment_id": 92}
        self._seed_cache({"a.py": [], "b.py": [f91, f92]})
        mc = self._make_provider()

        def review_and_concurrent_retract(*args, **kwargs):
            # Simulate the comment flow pruning comment 91 mid-review.
            entry = _previous_diffs["gitea:owner/repo#42"]
            entry.findings["b.py"] = [
                f for f in entry.findings["b.py"] if f.get("comment_id") != 91
            ]
            return {"severity": "low", "summary": "ok", "findings": []}

        self._run(mc, review_side_effect=review_and_concurrent_retract)
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        assert [f["comment_id"] for f in cached_after] == [92]

    def test_carried_cap_topk_by_severity(self):
        """The re-validation prompt set is capped server-side (top-K by
        severity); overflow findings are carried verbatim — they can't
        be dropped because the model never saw them, and the carry_id ↔
        candidate mapping stays aligned because the cap is applied
        before the call."""
        l1 = {"severity": "low", "file": "b.py", "line": 1, "message": "low one"}
        h1 = {"severity": "high", "file": "b.py", "line": 2, "message": "high one"}
        m1 = {"severity": "medium", "file": "b.py", "line": 3, "message": "medium one"}
        l2 = {"severity": "low", "file": "b.py", "line": 4, "message": "low two"}
        self._seed_cache({"a.py": [], "b.py": [l1, h1, m1, l2]})
        mc = self._make_provider()
        with patch("raven.server.RAVEN_CARRIED_REVALIDATION_MAX", 2):
            mock_review = self._run(mc, {"severity": "low", "summary": "ok",
                                         "findings": [], "dropped_carried": [0, 1]})
        # Top-2 by severity reached the model; ids 0/1 mapped onto them.
        assert mock_review.call_args.kwargs["carried_findings"] == [h1, m1]
        submitted_body = mc.submit_review.call_args[0][2]
        assert "high one" not in submitted_body
        assert "medium one" not in submitted_body
        # Overflow candidates carried verbatim.
        assert "low one" in submitted_body
        assert "low two" in submitted_body
        cached_after = _previous_diffs["gitea:owner/repo#42"].findings.get("b.py", [])
        assert [f["message"] for f in cached_after] == ["low one", "low two"]

    def test_carried_cap_ranks_unknown_severity_lowest(self):
        """Capping must NOT use scale.rank()'s model-emitted-severity fail
        CLOSED convention (unknown -> most severe) — these are already-
        validated cache entries, so the safe direction is the opposite:
        fail OPEN, ranking an unknown/empty severity as the LEAST severe,
        same as reviewer._cap_findings. Regression this pins: with
        scale.rank(), an unknown/empty-severity carried finding ties with
        (and, via stable sort, wins a cap slot ahead of) a genuine high
        finding instead of losing to it."""
        unknown = {"severity": "wat", "file": "b.py", "line": 1, "message": "unknown one"}
        empty = {"severity": "", "file": "b.py", "line": 2, "message": "empty one"}
        h1 = {"severity": "high", "file": "b.py", "line": 3, "message": "high one"}
        self._seed_cache({"a.py": [], "b.py": [unknown, empty, h1]})
        mc = self._make_provider()
        with patch("raven.server.RAVEN_CARRIED_REVALIDATION_MAX", 1):
            mock_review = self._run(mc, {"severity": "low", "summary": "ok", "findings": []})
        # Cap of 1: only the genuine high finding should reach the model —
        # unknown/empty severities must rank at the bottom, not tie for
        # the top with (and displace) a real high finding.
        assert mock_review.call_args.kwargs["carried_findings"] == [h1]

    def test_duplicate_fresh_restatement_deduped(self):
        """The prompt forbids copying carried findings into `findings`,
        but a model may restate one anyway. Verbatim duplicates are
        dropped in favor of the carried copy (it holds the comment_id
        retraction needs) — no duplicate inline comments, and no clone
        leaking into the '' cache bucket (the restated copy names an
        unchanged file, which _findings_by_file would bucket under '')."""
        valid = {"severity": "medium", "file": "b.py", "line": 20,
                 "message": "still valid", "comment_id": 43}
        self._seed_cache({"a.py": [], "b.py": [valid]})
        mc = self._make_provider()
        restated = {"severity": "medium", "file": "b.py", "line": 20,
                    "message": "still valid"}
        self._run(mc, {"severity": "medium", "summary": "ok",
                       "findings": [restated]})
        inline = mc.submit_review.call_args.kwargs["inline_comments"]
        assert sum("still valid" in c["body"] for c in inline) == 1
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert [f["message"] for f in entry.findings.get("b.py", [])] == ["still valid"]
        assert entry.findings.get("", []) == []


class TestReviewEvent:
    """Verify that review severity maps to the correct approve flag."""

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        return mc

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _run_with_severity(self, severity):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.side_effect = [
                [],                                                              # auto-add check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # gate check
                [{"user": {"login": "Raven"}, "state": "APPROVED"}],            # sole-reviewer check
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mock_review.return_value = {"severity": severity, "summary": "test", "findings": []}
            _process_pr(mc, self._normalized_payload())
        return mc

    def test_low_severity_approved(self):
        mc = self._run_with_severity("low")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is True

    def test_medium_severity_requests_changes(self):
        mc = self._run_with_severity("medium")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_high_severity_requests_changes(self):
        mc = self._run_with_severity("high")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_medium_approved_when_threshold_is_medium(self):
        with patch.dict(os.environ, {"REVIEW_APPROVE_MAX_SEVERITY": "medium"}):
            mc = self._run_with_severity("medium")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is True

    def test_high_still_rejected_when_threshold_is_medium(self):
        with patch.dict(os.environ, {"REVIEW_APPROVE_MAX_SEVERITY": "medium"}):
            mc = self._run_with_severity("high")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_medium_approved_when_threshold_is_capitalized(self):
        # REVIEW_APPROVE_MAX_SEVERITY is compared case-insensitively (matching
        # reviewer._coverage_gap_floor's .lower()); a capitalized value must
        # still approve a medium review rather than silently reading as 'low'
        # (SEVERITY_ORDER.get("Medium") -> default 0). Audit 07-02 #3.
        with patch.dict(os.environ, {"REVIEW_APPROVE_MAX_SEVERITY": "Medium"}):
            mc = self._run_with_severity("medium")
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is True


class TestParseErrorBlocksMerge:
    """Fix 1: _parse_error in review must block auto-merge."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        return mc

    def test_parse_error_posts_warning_and_skips_merge(self):
        mc = self._make_provider()
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.side_effect = [
            [],                                                        # auto-add check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],      # gate check
        ]
        mc.get_pr_requested_reviewers.return_value = []
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify") as mock_notify,
        ):
            mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
            mc.fetch_file.return_value = ""
            mc.post_pr_comment.return_value = {"id": 1}
            mock_review.return_value = {
                "severity": "high",
                "summary": "Review could not be parsed.",
                "findings": [],
                "_parse_error": True,
            }
            _process_pr(mc, self._normalized_payload())
        mc.merge_pr.assert_not_called()
        comment_body = mc.post_pr_comment.call_args[0][2]
        assert "Could not parse" in comment_body
        mock_notify.assert_called_once()
        assert mock_notify.call_args[1]["action"] == "review_failed"


class TestCoverageGapBlocksMerge:
    """Per-file coverage-gap tracking: a review carrying
    ``coverage_gap_files`` (oversized/failed chunks → part of the diff
    never reviewed) must post as needs_work — a formal APPROVE for code
    Raven never saw is externally visible (branch protection counts bot
    approvals; humans trust it) and must never happen. The gap is sticky
    only for files that haven't changed since they were skipped: once
    the named file is re-reviewed cleanly, the gap clears and approval
    is allowed again. The merge-dispatch gate stays as defense-in-depth.

    Implementation spans raven/reviewer.py (gap detection + file-keyed
    ⚠️ markers on the chunked paths) and raven/server.py (CacheEntry
    persistence, per-file sticky carry, verdict force, gates in both
    dispatch flows) — earlier commits on this branch; see CLAUDE.md
    "Coverage-gap tracking" for the lifecycle summary."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _normalized_payload(self, pr_number=42):
        return {
            "repo": "owner/repo",
            "sender": "alice",
            "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}",
            "pr_url": "https://git/pulls/42",
            "head_sha": "abc123",
            "head_ref": "feature",
            "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        return mc

    def _run(self, review, diff="diff --git a/f b/f\n+line\n"):
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server._safe_do_merge") as mock_merge,
        ):
            mc.fetch_pr_diff.return_value = diff
            mc.fetch_file.return_value = ""
            mc.submit_review.return_value = {"id": 1}
            mc.add_label_to_pr.return_value = None
            mc.get_commit_status.return_value = "success"
            mc.merge_pr.return_value = True
            mc.get_authenticated_user.return_value = "Raven"
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "Raven"}, "state": "APPROVED"},
            ]
            mc.get_pr_requested_reviewers.return_value = []
            mc.get_pr_head_sha.return_value = "abc123"
            mock_review.return_value = review
            _process_pr(mc, self._normalized_payload())
        return mc, mock_merge

    # Two-file diff used by the incremental tests below. a.py's hash is
    # computed from the real chunk text (unchanged); b.py gets a stale
    # hash (changed) — so each test controls which file is "changed".
    _TWO_FILE_DIFF = (
        "diff --git a/a.py b/a.py\n+aaa\n"
        "diff --git a/b.py b/b.py\n+bbb\n"
    )

    def _seed_cache(self, gap_files, changed="b.py"):
        """Seed a prior needs_work entry for owner/repo#42 where every
        file in _TWO_FILE_DIFF is unchanged except ``changed``.

        Realistic seed: the prior review's skip-marker findings are
        cached under their gap file's bucket (markers carry 'file', so
        _findings_by_file puts them there) — NOT findings={}. Seeding an
        empty findings map previously masked the ''-bucket carry bug:
        file-less markers landed under '' and were re-carried on every
        incremental pass, pinning severity forever."""
        import hashlib as _hashlib
        from raven.reviewer import split_diff_by_file as _split
        chunks = dict(_split(self._TWO_FILE_DIFF))
        hashes = {
            f: ("stale-hash" if f == changed
                else _hashlib.sha256(c.encode()).hexdigest())
            for f, c in chunks.items()
        }
        findings = {
            f: [{"severity": "medium", "file": f,
                 "message": f"⚠️ `{f}` skipped (too large: 9001 lines)"}]
            for f in gap_files
        }
        _previous_diffs["gitea:owner/repo#42"] = CacheEntry(
            timestamp=0.0,
            hashes=hashes,
            findings=findings,
            verdict="needs_work",
            summary="partial",
            coverage_gap_files=list(gap_files),
        )

    def test_review_with_coverage_gap_posts_needs_work_and_skips_merge(self):
        """A formal APPROVE must never post for partially-unreviewed
        code — even when the fresh severity is approvable (sticky gap on
        a clean incremental pass, or REVIEW_APPROVE_MAX_SEVERITY=high
        defeating the floor). The review still posts, as needs_work."""
        review = {
            "severity": "low",  # approvable by the default threshold
            "summary": "partial review",
            "findings": [{"severity": "medium",
                          "message": "⚠️ `big.py` skipped (too large: 9001 lines)"}],
            "chunked": True,
            "chunks_reviewed": 1,
            "coverage_gap": True,
            "coverage_gap_files": ["big.py"],
        }
        mc, mock_merge = self._run(review)
        # The review itself still posts with its findings …
        mc.submit_review.assert_called_once()
        # … but NOT as an externally-visible bot approval …
        assert mc.submit_review.call_args.kwargs["approve"] is False
        # … and the merge is gated.
        mock_merge.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_coverage_gap_files_persisted_to_cache_entry(self):
        """The gap-file list must land in CacheEntry so the comment-reply
        flow (which reads the cache, not the review dict) can see it."""
        review = {
            "severity": "medium",
            "summary": "partial review",
            "findings": [{"severity": "medium",
                          "message": "⚠️ `big.py` skipped (too large: 9001 lines)"}],
            "chunked": True,
            "chunks_reviewed": 1,
            "coverage_gap": True,
            "coverage_gap_files": ["big.py"],
        }
        self._run(review)
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert entry.coverage_gap_files == ["big.py"]

    def test_incremental_gap_persists_while_gap_file_unchanged(self):
        """Incremental re-reviews only run review_diff on changed files —
        an unchanged oversized file (a.py) is never re-reviewed, so a
        gap-free fresh review of the OTHER file (b.py) must not clear
        the gap: verdict stays needs_work, no merge, list persists."""
        self._seed_cache(gap_files=["a.py"], changed="b.py")
        review = {  # fresh review of b.py alone: clean, no gap
            "severity": "low", "summary": "ok", "findings": [],
            "chunked": False, "chunks_reviewed": 1,
        }
        mc, mock_merge = self._run(review, diff=self._TWO_FILE_DIFF)
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is False
        mock_merge.assert_not_called()
        assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == ["a.py"]
        # While the gap persists, the carried marker stays VISIBLE in
        # the posted review body (a.py is unchanged → its bucket carries).
        assert "skipped (too large" in mc.submit_review.call_args.args[2]

    def test_incremental_gap_clears_when_gap_file_rereviewed_clean(self):
        """Lifecycle: the author fixes the oversized file (b.py changes),
        the incremental pass re-reviews it cleanly → the gap clears,
        approval is allowed again and the merge dispatches. Without the
        per-file form, the old bool OR-ed back in forever and the PR
        could never auto-merge again."""
        self._seed_cache(gap_files=["b.py"], changed="b.py")
        review = {  # fresh review of the now-reasonable b.py: clean
            "severity": "low", "summary": "ok", "findings": [],
            "chunked": False, "chunks_reviewed": 1,
        }
        mc, mock_merge = self._run(review, diff=self._TWO_FILE_DIFF)
        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is True
        mock_merge.assert_called_once()
        assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == []
        # The STALE marker must not re-post once the gap file was
        # re-reviewed (b.py changed → its old bucket, marker included,
        # is replaced by the fresh review's findings).
        assert "skipped (too large" not in mc.submit_review.call_args.args[2]

    def test_end_to_end_marker_does_not_pin_severity_after_gap_clears(self, monkeypatch):
        """REGRESSION (PR #157 re-review, HIGH): markers used to be
        file-less, so _findings_by_file bucketed them under '' — and the
        carry loop carries the '' bucket on EVERY incremental pass. One
        gap event then pinned the merged severity at the marker's floor
        ('medium') forever: after the author fixed the oversized file
        and the fresh review came back clean, coverage_gap_files cleared
        but approve stayed False and the stale '⚠️ skipped' marker
        re-posted on every push.

        Run the REAL review_diff (backend mocked) through two
        _process_pr passes so the marker shape reviewer.py emits and the
        bucketing/carry logic server.py applies are exercised together
        — a hand-mocked review_diff can't catch a shape mismatch
        between the two."""
        from raven.ai.base import CompletionResult
        import raven.reviewer as rev

        fake_backend = MagicMock()
        fake_backend.name = "claude_cli"
        fake_backend.complete.return_value = CompletionResult(
            text='{"severity": "low", "summary": "ok", "findings": []}',
            input_tokens=0, output_tokens=0, cost_usd=None,
        )
        monkeypatch.setattr("raven.ai._cached_backend", fake_backend)
        monkeypatch.setattr(rev, "MAX_DIFF_LINES", 10)  # oversized: >30 lines

        diff_pass1 = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 5 +
            "diff --git a/b.py b/b.py\n" + "+big\n" * 40   # 41 lines → skipped
        )
        diff_pass2 = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 5 +  # unchanged
            "diff --git a/b.py b/b.py\n+fixed\n"            # fixed: small now
        )

        mc = self._make_provider()
        mc.fetch_file.return_value = ""
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"

        with (
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server._safe_do_merge") as mock_merge,
        ):
            # Pass 1: full review, b.py oversized → marker + needs_work.
            mc.fetch_pr_diff.return_value = diff_pass1
            _process_pr(mc, self._normalized_payload())
            assert mc.submit_review.call_args.kwargs["approve"] is False
            assert "skipped (too large" in mc.submit_review.call_args.args[2]
            assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == ["b.py"]
            mock_merge.assert_not_called()

            # Pass 2: author fixes b.py → incremental re-review is clean.
            mc.fetch_pr_diff.return_value = diff_pass2
            payload2 = self._normalized_payload()
            payload2["head_sha"] = "def456"  # new push, dodge dedup
            _process_pr(mc, payload2)

        body2 = mc.submit_review.call_args.args[2]
        # Approve again — severity is NOT pinned by a stale carried marker …
        assert mc.submit_review.call_args.kwargs["approve"] is True
        # … the stale marker does NOT re-post …
        assert "skipped (too large" not in body2
        # … the gap is cleared …
        assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == []
        # … and auto-merge dispatches again.
        mock_merge.assert_called_once()

    def test_end_to_end_gap_clears_when_gap_file_removed_from_pr(self, monkeypatch):
        """PIN (PR #157 re-review, finding B — code already correct):
        a gap file REMOVED from the PR (commit reverted / split out to
        another PR) is not in changed_files, so the incremental gap
        carry alone would never drop it. It doesn't have to: the file
        IS in removed_files (previous_hashes - current_hashes), and the
        removed-files gate forces a FULL re-review — is_incremental
        stays False, the gap carry contributes set(), and findings_map
        is rebuilt fresh-only. Both the gap and the stale marker clear.

        Two real review_diff passes (backend mocked) prove it
        end-to-end: pass 1 caches the real marker + gap for the
        oversized b.py; pass 2's diff contains ONLY a.py."""
        from raven.ai.base import CompletionResult
        import raven.reviewer as rev

        fake_backend = MagicMock()
        fake_backend.name = "claude_cli"
        fake_backend.complete.return_value = CompletionResult(
            text='{"severity": "low", "summary": "ok", "findings": []}',
            input_tokens=0, output_tokens=0, cost_usd=None,
        )
        monkeypatch.setattr("raven.ai._cached_backend", fake_backend)
        monkeypatch.setattr(rev, "MAX_DIFF_LINES", 10)  # oversized: >30 lines

        diff_pass1 = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 5 +
            "diff --git a/b.py b/b.py\n" + "+big\n" * 40   # 41 lines → skipped
        )
        # b.py is GONE from the PR entirely (reverted / split out).
        diff_pass2 = "diff --git a/a.py b/a.py\n" + "+line\n" * 5

        mc = self._make_provider()
        mc.fetch_file.return_value = ""
        mc.get_pr_description.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.list_directory.return_value = []
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"

        with (
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server._safe_do_merge") as mock_merge,
        ):
            # Pass 1: full review, b.py oversized → marker + needs_work.
            mc.fetch_pr_diff.return_value = diff_pass1
            _process_pr(mc, self._normalized_payload())
            assert mc.submit_review.call_args.kwargs["approve"] is False
            assert "skipped (too large" in mc.submit_review.call_args.args[2]
            assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == ["b.py"]
            mock_merge.assert_not_called()

            # Pass 2: b.py removed → removed_files gate → full re-review.
            mc.fetch_pr_diff.return_value = diff_pass2
            payload2 = self._normalized_payload()
            payload2["head_sha"] = "def456"  # new push, dodge dedup
            _process_pr(mc, payload2)

        body2 = mc.submit_review.call_args.args[2]
        # The full re-review approves — no gap, no severity pin …
        assert mc.submit_review.call_args.kwargs["approve"] is True
        # … the stale '⚠️ b.py' marker does NOT re-post …
        assert "skipped (too large" not in body2
        # … cache reflects the full path: gap cleared, b.py's hash and
        # findings buckets rebuilt fresh-only (b.py gone entirely) …
        entry = _previous_diffs["gitea:owner/repo#42"]
        assert entry.coverage_gap_files == []
        assert set(entry.hashes.keys()) == {"a.py"}
        assert not any(
            "skipped (too large" in f.get("message", "")
            for findings in entry.findings.values() for f in findings
        )
        # … and auto-merge dispatches again.
        mock_merge.assert_called_once()

    def test_incremental_gap_persists_when_gap_file_still_oversized(self):
        """The gap file changed but is STILL too large — the fresh
        review names it again, so the gap persists via the fresh list."""
        self._seed_cache(gap_files=["b.py"], changed="b.py")
        review = {
            "severity": "medium",
            "summary": "partial review",
            "findings": [{"severity": "medium",
                          "message": "⚠️ `b.py` skipped (too large: 9001 lines)"}],
            "chunked": True,
            "chunks_reviewed": 1,
            "coverage_gap": True,
            "coverage_gap_files": ["b.py"],
        }
        mc, mock_merge = self._run(review, diff=self._TWO_FILE_DIFF)
        assert mc.submit_review.call_args.kwargs["approve"] is False
        mock_merge.assert_not_called()
        assert _previous_diffs["gitea:owner/repo#42"].coverage_gap_files == ["b.py"]


class TestUnfetchableScaleBlocksMerge:
    """PR #216 review, Finding 1: a severities.json that EXISTS but could
    not be FETCHED (transient network/auth failure at the provider) must
    not be treated the same as "no file" — the repo may have a stricter
    scale than the built-in default, and silently reviewing under a
    guessed vocabulary risks auto-merging a PR the repo's real gate would
    have blocked. Mirrors TestCoverageGapBlocksMerge: the review still
    posts (the author gets feedback) but the verdict is forced to
    needs_work and auto-merge is refused."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _payload(self, pr_number=42):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": pr_number,
            "pr_title": f"PR #{pr_number}", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self, fetch_file_side_effect):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_file.side_effect = fetch_file_side_effect
        mc.fetch_pr_diff.return_value = "diff --git a/f b/f\n+line\n"
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"},
        ]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        return mc

    def _run_with_fetch(self, fetch_file_side_effect, review):
        mc = self._make_provider(fetch_file_side_effect)
        with (
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
            patch("raven.server._safe_do_merge") as mock_merge,
        ):
            _process_pr(mc, self._payload())
        return mc, mock_merge

    def test_fetch_failure_forces_needs_work_and_skips_merge(self):
        def fetch_file(repo, path, ref=None, **kw):
            if path.endswith("severities.json"):
                raise RuntimeError("boom: transient fetch failure")
            return ""

        review = {"severity": "low", "summary": "clean", "findings": []}
        mc, mock_merge = self._run_with_fetch(fetch_file, review)

        mc.submit_review.assert_called_once()
        assert mc.submit_review.call_args.kwargs["approve"] is False
        mock_merge.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_fetch_failure_increments_metric(self, mocker):
        import raven.server as server
        spy = mocker.patch.object(server, "inc")

        def fetch_file(repo, path, ref=None, **kw):
            if path.endswith("severities.json"):
                raise RuntimeError("boom")
            return ""

        review = {"severity": "low", "summary": "clean", "findings": []}
        self._run_with_fetch(fetch_file, review)

        spy.assert_any_call("raven_severity_scale_fetch_failed_total",
                            {"repo": "owner/repo"})

    def test_missing_file_still_approves(self):
        """Regression guard: the ordinary "no severities.json at all" case
        (most repos) must still approve exactly as before — only a real
        fetch EXCEPTION triggers the fail-closed behaviour."""
        def fetch_file(repo, path, ref=None, **kw):
            return ""  # every fetch cleanly reports "absent", none raise

        review = {"severity": "low", "summary": "clean", "findings": []}
        mc, mock_merge = self._run_with_fetch(fetch_file, review)

        assert mc.submit_review.call_args.kwargs["approve"] is True
        mock_merge.assert_called_once()


class TestDedup:
    """Fix 5: duplicate PR reviews within DEDUP_WINDOW are skipped."""

    def setup_method(self):
        _recent_prs.clear()

    def test_first_call_allowed(self):
        assert _should_skip_duplicate("owner/repo", 42) is False

    def test_second_call_within_window_skipped(self):
        _should_skip_duplicate("owner/repo", 42)
        assert _should_skip_duplicate("owner/repo", 42) is True

    def test_different_pr_allowed(self):
        _should_skip_duplicate("owner/repo", 42)
        assert _should_skip_duplicate("owner/repo", 43) is False

    def test_expired_entry_allowed(self):
        _should_skip_duplicate("owner/repo", 42)
        # Simulate time passing beyond the dedup window
        _recent_prs["owner/repo#42"] -= DEDUP_WINDOW + 1
        assert _should_skip_duplicate("owner/repo", 42) is False

    def test_different_head_sha_allowed(self):
        """A push with a different head SHA is a legitimate new event,
        not a redelivery — it must not be dropped as a duplicate."""
        _should_skip_duplicate("owner/repo", 42, head_sha="aaaaaaaa")
        assert _should_skip_duplicate("owner/repo", 42, head_sha="bbbbbbbb") is False

    def test_same_head_sha_skipped(self):
        """Redelivery of the same webhook (same SHA) is still deduped."""
        _should_skip_duplicate("owner/repo", 42, head_sha="aaaaaaaa")
        assert _should_skip_duplicate("owner/repo", 42, head_sha="aaaaaaaa") is True

    def test_head_sha_optional_preserves_legacy_key(self):
        """Calls without head_sha (e.g. comment dedup) keep the old key
        format so they don't collide with SHA-keyed entries."""
        _should_skip_duplicate("owner/repo", 42)
        assert "owner/repo#42" in _recent_prs
        _should_skip_duplicate("owner/repo", 42, head_sha="aaaaaaaa")
        assert "owner/repo#42@aaaaaaaa" in _recent_prs


class TestIssueComment:
    """Test conversational follow-up on PR comments."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        _recent_prs.clear()
        getattr(_server_mod, "_recent_pr_replies", {}).clear()
        yield
        _recent_prs.clear()
        getattr(_server_mod, "_recent_pr_replies", {}).clear()

    def _comment_payload(self, body="@Raven explain this", user="alice", is_pull=True):
        return {
            "action": "created",
            "is_pull": is_pull,
            "issue": {"number": 42},
            "comment": {
                "body": body,
                "user": {"login": user},
            },
            "repository": {"full_name": "owner/repo"},
        }

    def _normalized_comment_payload(self, body="@Raven explain this", user="alice",
                                    is_mention=True):
        return {
            "repo": "owner/repo",
            "sender": user,
            "pr_number": 42,
            "comment_body": body,
            "comment_user": user,
            "comment_id": 999,
            "file_path": "",
            "line": 0,
            "_is_mention": is_mention,
        }

    def test_mention_triggers_response(self, client):
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("@Raven why is this bad?"), event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_back_to_back_mentions_both_trigger(self, client):
        """Two users @mentioning in quick succession must both get a response —
        no per-PR cooldown dropping the second one silently."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            r1 = _post(client, self._comment_payload("@Raven first q", user="alice"), event="issue_comment")
            # Force a distinct comment id for the second delivery
            p2 = self._comment_payload("@Raven second q", user="bob")
            p2["comment"]["id"] = 9001
            r2 = _post(client, p2, event="issue_comment")
        assert r1.get_json()["status"] == "accepted"
        assert r2.get_json()["status"] == "accepted"
        assert mock_executor.submit.call_count == 2

    def test_own_comment_skipped(self, client):
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"):
            resp = _post(client, self._comment_payload(user="Raven"), event="issue_comment")
        assert resp.get_json()["status"] == "skipped"
        assert resp.get_json()["reason"] == "own comment"

    def test_bot_authored_comment_skipped(self, client):
        """A comment authored by a bot (e.g. a second auto-responder or
        dependabot) must not dispatch _process_comment — otherwise two
        bots in a thread create an unbounded paid reply loop. Uses the
        same _is_bot_author heuristic push/PR events use."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(
                client,
                self._comment_payload("@Raven take a look", user="dependabot"),
                event="issue_comment",
            )
        assert resp.get_json()["status"] == "skipped"
        assert resp.get_json()["reason"] == "bot author"
        mock_executor.submit.assert_not_called()

    def test_bot_suffix_authored_comment_skipped(self, client):
        """The ``foo[bot]`` GitHub-style suffix is also caught."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(
                client,
                self._comment_payload("@Raven thoughts?", user="other-reviewer[bot]"),
                event="issue_comment",
            )
        assert resp.get_json()["status"] == "skipped"
        assert resp.get_json()["reason"] == "bot author"
        mock_executor.submit.assert_not_called()

    def test_reply_budget_blocks_after_cap(self, client):
        """Backstop for loops the name heuristic misses: once a PR has
        received RAVEN_MAX_PR_REPLIES_PER_HOUR dispatched replies within
        the sliding window, the next qualifying @mention is skipped for
        budget rather than dispatched (every dispatch is a paid AI call)."""
        provider = _providers["gitea"]
        with patch.dict(os.environ, {"RAVEN_MAX_PR_REPLIES_PER_HOUR": "2"}), \
             patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            # First two qualifying mentions dispatch and consume the budget.
            for i, cid in enumerate((1001, 1002)):
                p = self._comment_payload(f"@Raven q{i}", user="alice")
                p["comment"]["id"] = cid
                r = _post(client, p, event="issue_comment")
                assert r.get_json()["status"] == "accepted"
            # Third qualifying mention is over budget.
            p3 = self._comment_payload("@Raven q3", user="alice")
            p3["comment"]["id"] = 1003
            r3 = _post(client, p3, event="issue_comment")
        assert r3.get_json()["status"] == "skipped"
        assert r3.get_json()["reason"] == "reply budget exceeded"
        assert mock_executor.submit.call_count == 2

    def test_reply_budget_increments_skip_metric(self, client):
        """An over-budget skip is counted in raven_responses_skipped_total
        with reason=rate_limit so reply-loop suppression is observable."""
        from raven.metrics import _counters
        _counters.clear()
        provider = _providers["gitea"]
        with patch.dict(os.environ, {"RAVEN_MAX_PR_REPLIES_PER_HOUR": "1"}), \
             patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor"):
            p1 = self._comment_payload("@Raven q1", user="alice")
            p1["comment"]["id"] = 2001
            _post(client, p1, event="issue_comment")
            p2 = self._comment_payload("@Raven q2", user="alice")
            p2["comment"]["id"] = 2002
            _post(client, p2, event="issue_comment")
        keys = [k for k in _counters if k.startswith("raven_responses_skipped_total")]
        assert len(keys) == 1
        assert 'reason="rate_limit"' in keys[0]

    def test_reply_budget_is_per_pr(self, client):
        """The budget is keyed per-PR, so traffic on PR #42 must not starve
        replies on PR #43."""
        provider = _providers["gitea"]
        with patch.dict(os.environ, {"RAVEN_MAX_PR_REPLIES_PER_HOUR": "1"}), \
             patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            # Exhaust PR #42's budget.
            p1 = self._comment_payload("@Raven q", user="alice")
            p1["comment"]["id"] = 3001
            _post(client, p1, event="issue_comment")
            p2 = self._comment_payload("@Raven q", user="alice")
            p2["comment"]["id"] = 3002
            r2 = _post(client, p2, event="issue_comment")
            # A different PR still gets its first reply.
            p3 = self._comment_payload("@Raven q", user="alice")
            p3["issue"]["number"] = 43
            p3["comment"]["id"] = 3003
            r3 = _post(client, p3, event="issue_comment")
        assert r2.get_json()["reason"] == "reply budget exceeded"
        assert r3.get_json()["status"] == "accepted"

    def test_human_mention_under_budget_dispatches(self, client):
        """Guard: a normal human @mention while under budget still
        dispatches as before — the new gates don't regress the happy path."""
        provider = _providers["gitea"]
        with patch.dict(os.environ, {"RAVEN_MAX_PR_REPLIES_PER_HOUR": "20"}), \
             patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("@Raven please explain"),
                         event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_unrelated_comment_ignored(self, client):
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"):
            resp = _post(client, self._comment_payload("looks good to me"), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"

    def test_not_a_pr_ignored(self, client):
        resp = _post(client, self._comment_payload(is_pull=False), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"

    def test_mention_word_boundary(self, client):
        """@Ravenous should not trigger, @Raven should."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor"):
            resp = _post(client, self._comment_payload("@Ravenous looks fine"), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"

    def test_quoted_mention_triggers_response(self, client):
        """BB DC wraps usernames containing dots in double quotes: @"jenkins.builder"."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(
                client,
                self._comment_payload('@"jenkins.builder" will you reply?'),
                event="issue_comment",
            )
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_identity_failure_ignores_comments(self, client):
        """When identity lookup fails, comments are silently ignored."""
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value=""):
            resp = _post(client, self._comment_payload("@Raven explain"), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"

    # ── RAVEN_REPLY_REQUIRE_MENTION (mention-only mode) ──────────────────── #
    # When set, Raven replies ONLY to comments that explicitly tag it — by the
    # literal product name "@Raven" OR its account username — and never to an
    # untagged in-thread reply. Default (unset) keeps the current behaviour:
    # an account-username @mention OR any in-thread reply triggers a reply.
    # A non-"Raven" account username (jenkins.builder) is used so "@Raven" is
    # distinguishable from the account @mention.

    def _threaded_payload(self):
        # A thread reply (parent set) with NO tag in the body. Gitea's parser
        # never sets parent_comment_id, so inject the normalized payload via a
        # patched parse_webhook to exercise the provider-agnostic hook gate.
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "comment_body": "thanks, that makes sense", "comment_user": "alice",
            "comment_id": 777, "parent_comment_id": 500, "file_path": "", "line": 0,
        }

    def test_require_mention_replies_to_at_raven_name(self, client, monkeypatch):
        monkeypatch.setenv("RAVEN_REPLY_REQUIRE_MENTION", "1")
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("@Raven take a look"), event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_default_recognizes_at_raven_name(self, client):
        # @Raven (the product display name) is recognized in BOTH modes, even
        # when the bot account is named something else — the mention-only
        # switch governs only the untagged-thread-reply behaviour, not which
        # tags count as directing a comment at Raven.
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("@Raven take a look"), event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_require_mention_replies_to_account_username(self, client, monkeypatch):
        monkeypatch.setenv("RAVEN_REPLY_REQUIRE_MENTION", "1")
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload('@"jenkins.builder" please reply'), event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_require_mention_ignores_untagged_comment(self, client, monkeypatch):
        monkeypatch.setenv("RAVEN_REPLY_REQUIRE_MENTION", "1")
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("looks good to me"), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"
        mock_executor.submit.assert_not_called()

    def test_require_mention_skips_untagged_thread_reply(self, client, monkeypatch):
        # The key effect: an in-thread reply with no tag is NOT answered in
        # mention-only mode (default WOULD auto-reply on thread membership).
        monkeypatch.setenv("RAVEN_REPLY_REQUIRE_MENTION", "1")
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch.object(provider, "parse_webhook", return_value=("comment", self._threaded_payload())), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, {"x": "y"}, event="issue_comment")
        assert resp.get_json()["status"] == "ignored"
        mock_executor.submit.assert_not_called()

    def test_default_dispatches_untagged_thread_reply(self, client):
        # Current behaviour preserved: a thread reply dispatches even untagged;
        # _process_comment then decides via thread membership.
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch.object(provider, "parse_webhook", return_value=("comment", self._threaded_payload())), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, {"x": "y"}, event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_email_like_string_does_not_trigger(self, client):
        # An @name inside an email / user@host must not false-trigger: the @
        # must not follow a word char.
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="ravenbot"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("ping deploy@ravenbot.io please"),
                         event="issue_comment")
        assert resp.get_json()["status"] == "ignored"
        mock_executor.submit.assert_not_called()

    def test_mention_inside_code_span_does_not_trigger(self, client):
        # A tag inside a `code span` (or fenced block) is not a real mention.
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("see `@jenkins.builder` in the config"),
                         event="issue_comment")
        assert resp.get_json()["status"] == "ignored"
        mock_executor.submit.assert_not_called()

    def test_configurable_mention_name_recognized(self, client, monkeypatch):
        # RAVEN_MENTION_NAMES customizes the recognized display name(s).
        monkeypatch.setenv("RAVEN_MENTION_NAMES", "CodeReviewer")
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("@CodeReviewer take a look"),
                         event="issue_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_mention_only_skip_increments_metric(self, client, monkeypatch):
        from raven.metrics import _counters
        monkeypatch.setenv("RAVEN_REPLY_REQUIRE_MENTION", "1")
        _counters.clear()
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="jenkins.builder"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload("looks good to me"), event="issue_comment")
        assert resp.get_json()["status"] == "ignored"
        mock_executor.submit.assert_not_called()
        keys = [k for k in _counters
                if k.startswith("raven_responses_skipped_total") and 'reason="no_mention"' in k]
        assert keys, "expected a no_mention skip metric"

    def test_process_comment_posts_response(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = [{"user": {"login": "alice"}, "body": "@Raven explain"}]
        mc.post_pr_comment.return_value = {"id": 1}
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "The issue is that...", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())
        mc.post_pr_comment.assert_called_once()
        posted_body = mc.post_pr_comment.call_args[0][2]
        assert "The issue is that..." in posted_body

    def test_process_comment_replies_in_thread(self):
        """The triggering comment's id is passed as parent_comment_id so
        providers that support threading (BB DC) post the reply in the same
        thread rather than as a top-level comment."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = [{"user": {"login": "alice"}, "body": "@Raven explain"}]
        mc.post_pr_comment.return_value = {"id": 1}
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "reply text", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())
        assert mc.post_pr_comment.call_args[1]["parent_comment_id"] == 999

    def test_process_comment_posts_error_on_exception(self):
        """When respond_to_comment raises, the user must see something —
        silent failures look like Raven ignored them."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.side_effect = RuntimeError("Claude timed out")
            _process_comment(mc, self._normalized_comment_payload())
        mc.post_pr_comment.assert_called_once()
        posted_body = mc.post_pr_comment.call_args[0][2]
        assert "\u26a0\ufe0f" in posted_body  # warning emoji

    def test_process_comment_failure_increments_classified_metric(self):
        """The comment-reply flow shares the failure-rate dashboards: an
        AIError that bubbles past the RespondParseError guard increments
        raven_review_failures_total{reason} so timeout/usage spikes on
        replies are visible too. (respond_to_comment already retried the
        transient classes inside reviewer.py.)"""
        from raven.metrics import _counters
        from raven.ai.base import AIError
        _counters.clear()
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment",
                   side_effect=AIError("timed out", reason="timeout")):
            _process_comment(mc, self._normalized_comment_payload())
        keys = [k for k in _counters if k.startswith("raven_review_failures_total")]
        assert len(keys) == 1
        assert 'reason="timeout"' in keys[0]

    def test_process_comment_posts_error_on_empty_response(self):
        """Empty response from Claude should still surface to the user."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())
        mc.post_pr_comment.assert_called_once()
        posted_body = mc.post_pr_comment.call_args[0][2]
        assert "\u26a0\ufe0f" in posted_body

    def test_process_comment_sends_reaction_ack(self):
        """Raven reacts to the triggering comment before starting the slow
        Claude call, so the user has immediate feedback."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "ok", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())
        mc.react_to_comment.assert_called_once_with("owner/repo", 42, 999)

    def test_process_comment_reaction_failure_does_not_break_flow(self):
        """If the reaction call raises, the main response still posts."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.react_to_comment.side_effect = RuntimeError("reactions down")
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "the answer", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())
        mc.post_pr_comment.assert_called_once()
        assert "the answer" in mc.post_pr_comment.call_args[0][2]

    def test_process_comment_reply_path_verifies_thread_in_background(self):
        """Reply-in-thread payloads reach _process_comment with
        _is_mention=False — the worker must call get_comment_thread
        to decide whether Raven should engage (authors derived from the
        thread dicts)."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_comment_thread.return_value = [
            {"id": 700, "parent_id": None, "user": {"login": "alice"},
             "body": "...", "file_path": None, "line": None, "resolved": False},
            {"id": 701, "parent_id": 700, "user": {"login": "Raven"},
             "body": "...", "file_path": None, "line": None, "resolved": False},
        ]
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "ok", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        mc.get_comment_thread.assert_called_once_with("owner/repo", 42, 700)
        mc.post_pr_comment.assert_called_once()

    def test_process_comment_reply_path_skips_when_raven_not_in_thread(self):
        """If the thread doesn't contain Raven, the worker exits quietly
        without posting anything."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_comment_thread.return_value = [
            {"id": 700, "parent_id": None, "user": {"login": "alice"},
             "body": "...", "file_path": None, "line": None, "resolved": False},
            {"id": 701, "parent_id": 700, "user": {"login": "bob"},
             "body": "...", "file_path": None, "line": None, "resolved": False},
        ]
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700
        _process_comment(mc, payload)
        mc.post_pr_comment.assert_not_called()
        mc.react_to_comment.assert_not_called()

    def test_process_comment_reply_path_skips_when_thread_lookup_raises(self):
        """Provider error during thread lookup — worker exits quietly."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_comment_thread.side_effect = RuntimeError("503")
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700
        _process_comment(mc, payload)
        mc.post_pr_comment.assert_not_called()

    def test_process_comment_mention_skips_thread_lookup(self):
        """When the handler marked the comment as an @mention, the worker
        trusts that signal and doesn't hit the provider thread API."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "the answer", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload(is_mention=True))
        mc.get_comment_thread.assert_not_called()
        mc.post_pr_comment.assert_called_once()

    # ── Comment-reply context recovery (root-anchor fallback + full file) ── #

    def test_reply_recovers_anchor_from_thread_root(self):
        """A threaded reply carries no anchor of its own (the anchor is on
        the thread ROOT). When the trigger's file_path/line are empty, the
        worker must derive them from the root comment so the diff
        truncation + code context are biased to the right file.

        Regression: replies inside an inline thread used to lose all
        file/line context (BB DC sends commentParentId but no anchor on
        the reply; the root carries the anchor)."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        # Root comment (id 700) carries the inline anchor; the reply does not.
        mc.get_comment_thread.return_value = [
            {"id": 700, "parent_id": None, "user": {"login": "Raven"},
             "body": "potential bug here", "file_path": "src/app.py", "line": 42,
             "resolved": False},
            {"id": 999, "parent_id": 700, "user": {"login": "alice"},
             "body": "why?", "file_path": None, "line": None, "resolved": False},
        ]
        mc.fetch_pr_diff.return_value = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n+++ b/src/app.py\n"
            "@@ -40,4 +40,4 @@\n line\n-old\n+new\n line\n"
        )
        mc.fetch_file.return_value = "\n".join(f"line {i}" for i in range(1, 60))
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700  # reply seeds from root id
        payload["file_path"] = ""  # reply has no anchor
        payload["line"] = 0
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "because X", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        mock_respond.assert_called_once()
        kwargs = mock_respond.call_args[1]
        # The recovered anchor flows to respond_to_comment.
        assert kwargs["file_path"] == "src/app.py"
        assert kwargs["line"] == 42
        # And the recovered path biases the diff truncation so the file
        # under discussion survives (positional diff arg, index 2).
        passed_diff = mock_respond.call_args[0][2]
        assert "src/app.py" in passed_diff

    def test_reply_recovers_anchor_fetches_full_file(self):
        """After recovering the anchor, the worker fetches the FULL modified
        file (not just a ±10-line snippet) so a question about code outside
        the snippet window is answerable."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_comment_thread.return_value = [
            {"id": 700, "parent_id": None, "user": {"login": "Raven"},
             "body": "finding", "file_path": "src/app.py", "line": 42,
             "resolved": False},
            {"id": 999, "parent_id": 700, "user": {"login": "alice"},
             "body": "why?", "file_path": None, "line": None, "resolved": False},
        ]
        mc.fetch_pr_diff.return_value = "diff --git a/src/app.py b/src/app.py\n+x\n"
        full_file = "\n".join(f"line {i}" for i in range(1, 60))
        mc.fetch_file.return_value = full_file
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700
        payload["file_path"] = ""
        payload["line"] = 0
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "r", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        kwargs = mock_respond.call_args[1]
        # The full file content is passed through (new param), not only the
        # narrow snippet.
        assert kwargs.get("file_content") == full_file

    def test_reply_over_cap_file_disclosed_not_passed(self):
        """A modified file exceeding MAX_FILE_LINES is not attached in full;
        instead the omission is disclosed to the model (file_truncated)."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.fetch_pr_diff.return_value = "diff --git a/big.py b/big.py\n+x\n"
        # Build a file well over the 500-line cap.
        big = "\n".join(f"line {i}" for i in range(1, 2000))
        mc.fetch_file.return_value = big
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=True)
        payload["file_path"] = "big.py"
        payload["line"] = 100
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "r", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        kwargs = mock_respond.call_args[1]
        # Over-cap: full content NOT attached, but the truncation is disclosed.
        assert not kwargs.get("file_content")
        assert kwargs.get("file_truncated") is True

    def test_reply_fetch_failure_disclosed(self):
        """When the file fetch raises, the worker tells the model the code
        context couldn't be fetched (context_fetch_failed) so it flags
        uncertainty instead of guessing."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.fetch_pr_diff.return_value = "diff --git a/a.py b/a.py\n+x\n"

        def fetch_file(repo, path, ref="HEAD"):
            if path == "CLAUDE.md":
                return ""  # CLAUDE.md absent — not the failure under test
            raise RuntimeError("500 fetching file")

        mc.fetch_file.side_effect = fetch_file
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=True)
        payload["file_path"] = "a.py"
        payload["line"] = 10
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "r", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        kwargs = mock_respond.call_args[1]
        assert kwargs.get("context_fetch_failed") is True

    def test_reply_no_anchor_and_root_anchorless_degrades(self):
        """A flat (non-inline) reply where the thread root ALSO has no
        anchor must degrade gracefully — no anchor recovered, no file
        content fetched, the reply still posts."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_comment_thread.return_value = [
            {"id": 700, "parent_id": None, "user": {"login": "Raven"},
             "body": "general note", "file_path": None, "line": None,
             "resolved": False},
            {"id": 999, "parent_id": 700, "user": {"login": "alice"},
             "body": "follow up", "file_path": None, "line": None, "resolved": False},
        ]
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        payload = self._normalized_comment_payload(is_mention=False)
        payload["parent_comment_id"] = 700
        payload["file_path"] = ""
        payload["line"] = 0
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "r", "revise": None, "retract_findings": []}
            _process_comment(mc, payload)
        kwargs = mock_respond.call_args[1]
        assert kwargs["file_path"] == ""
        assert not kwargs.get("file_content")
        assert not kwargs.get("file_truncated")
        assert not kwargs.get("context_fetch_failed")
        mc.post_pr_comment.assert_called_once()

    def test_respond_threads_prompt_override_from_base_branch(self):
        """When .claude/rules/raven/prompts/respond.md exists on the base
        branch, its contents are passed to respond_to_comment as
        prompt_override."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.get_pr_comments.return_value = [{"user": {"login": "alice"}, "body": "@Raven explain"}]
        mc.post_pr_comment.return_value = {"id": 1}
        mc.get_pr_base_ref.return_value = "main"

        def fetch_file(repo, path, ref="HEAD"):
            if path == ".claude/rules/raven/prompts/respond.md":
                assert ref == "main"
                return "REPO-SPECIFIC RESPOND PROMPT"
            return ""

        mc.fetch_file.side_effect = fetch_file

        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "reply body", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())

        kwargs = mock_respond.call_args.kwargs
        assert kwargs.get("prompt_override") == "REPO-SPECIFIC RESPOND PROMPT"

    def test_respond_no_override_passes_none(self):
        """When the override file doesn't exist, prompt_override is None."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.side_effect = FileNotFoundError()
        mc.get_pr_comments.return_value = [{"user": {"login": "alice"}, "body": "@Raven explain"}]
        mc.post_pr_comment.return_value = {"id": 1}
        mc.get_pr_base_ref.return_value = "main"

        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "reply body", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())

        kwargs = mock_respond.call_args.kwargs
        assert kwargs.get("prompt_override") is None

    def test_respond_tolerates_base_ref_fetch_failure(self):
        """If get_pr_base_ref raises, respond still runs with no override."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = [{"user": {"login": "alice"}, "body": "@Raven explain"}]
        mc.post_pr_comment.return_value = {"id": 1}
        mc.get_pr_base_ref.side_effect = RuntimeError("boom")

        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "reply body", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_comment_payload())

        assert mock_respond.called
        kwargs = mock_respond.call_args.kwargs
        assert kwargs.get("prompt_override") is None


class TestPullRequestComment:
    """Test handling of pull_request_comment events (inline diff comments)."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        _recent_prs.clear()
        yield
        _recent_prs.clear()

    def _comment_payload(self, body="@Raven explain", user="alice",
                         path="server.py", line=42):
        return {
            "action": "created",
            "comment": {
                "body": body,
                "user": {"login": user},
                "id": 999,
                "path": path,
                "line": line,
            },
            "pull_request": {"number": 42},
            "repository": {"full_name": "owner/repo"},
        }

    def _normalized_diff_comment(self, body="@Raven explain", user="alice",
                                  file_path="server.py", line=42,
                                  is_mention=True):
        return {
            "repo": "owner/repo",
            "sender": user,
            "pr_number": 42,
            "comment_body": body,
            "comment_user": user,
            "comment_id": 999,
            "file_path": file_path,
            "line": line,
            "_is_mention": is_mention,
        }

    def test_mention_in_diff_comment_triggers_response(self, client):
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"), \
             patch("raven.server.executor") as mock_executor:
            resp = _post(client, self._comment_payload(), event="pull_request_comment")
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_non_mention_diff_comment_ignored(self, client):
        provider = _providers["gitea"]
        with patch.object(provider, "get_authenticated_user", return_value="Raven"):
            resp = _post(client, self._comment_payload(body="looks fine"),
                         event="pull_request_comment")
        assert resp.get_json()["status"] == "ignored"

    def test_process_diff_comment_passes_file_context(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.post_pr_comment.return_value = {"id": 1}
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "The issue is...", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_diff_comment())
        # Verify file_path and line were passed to respond_to_comment
        call_kwargs = mock_respond.call_args[1]
        assert call_kwargs.get("file_path") == "server.py"
        assert call_kwargs.get("line") == 42

    def test_diff_comment_response_includes_location_header_on_flat_providers(self):
        """Gitea (no comment threading) keeps the Re: header for context."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.supports_comment_threads = False
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.post_pr_comment.return_value = {"id": 1}
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "Because of X.", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_diff_comment(body="@Raven why?"))
        posted_body = mc.post_pr_comment.call_args[0][2]
        assert "server.py" in posted_body
        assert "line 42" in posted_body
        assert "Because of X." in posted_body

    def test_diff_comment_includes_code_snippet_in_prompt(self):
        """Inline diff comments should have a line-numbered code window
        passed to respond_to_comment so Claude doesn't have to locate the
        line by parsing hunk headers."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.supports_comment_threads = False
        mc.get_pr_head_sha.return_value = "abc123"
        mc.fetch_pr_diff.return_value = "diff --git a/server.py\n+x\n"
        # Path-keyed rather than an ordered positional list: fetch_file is
        # also called for the repo's severities.json (base-ref provenance,
        # Task 14), and a fixed-position list breaks the instant another
        # base-ref fetch is added between two existing ones. Keyed by path
        # instead, so the fixture describes WHAT each path returns, not how
        # many calls happen or in what order.
        mc.fetch_file.side_effect = lambda repo, path, ref="HEAD": (
            "\n".join(f"row-{i}" for i in range(1, 101)) if path == "server.py" else ""
        )
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "ok", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_diff_comment(line=42))
        call_kwargs = mock_respond.call_args[1]
        snippet = call_kwargs.get("code_snippet", "")
        assert "→ row-42" in snippet
        # 10 lines of context on each side
        assert "row-32" in snippet
        assert "row-52" in snippet
        mc.get_pr_head_sha.assert_called_once_with("owner/repo", 42)

    def test_diff_comment_snippet_skipped_when_head_sha_fails(self):
        """If get_pr_head_sha raises, just skip the snippet — the response
        still gets generated using the diff alone."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.supports_comment_threads = False
        mc.get_pr_head_sha.side_effect = RuntimeError("no sha")
        mc.fetch_pr_diff.return_value = "diff --git a/server.py\n+x\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "ok", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_diff_comment(line=42))
        call_kwargs = mock_respond.call_args[1]
        assert call_kwargs.get("code_snippet", "") == ""

    def test_diff_comment_response_omits_location_header_when_threaded(self):
        """Threading providers (BB DC) render the thread at the file/line
        already, so the Re: header would duplicate context."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.supports_comment_threads = True
        mc.fetch_pr_diff.return_value = "diff --git a/f\n+line\n"
        mc.fetch_file.return_value = ""
        mc.get_pr_comments.return_value = []
        mc.post_pr_comment.return_value = {"id": 1}
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {"response": "Because of X.", "revise": None, "retract_findings": []}
            _process_comment(mc, self._normalized_diff_comment(body="@Raven why?"))
        posted_body = mc.post_pr_comment.call_args[0][2]
        assert "**Re:" not in posted_body
        assert "Because of X." in posted_body


class TestExtractCodeSnippet:
    def test_window_around_line(self):
        content = "\n".join(f"line-{i}" for i in range(1, 21))
        out = _extract_code_snippet(content, line=10, context=2)
        # Expected: lines 8..12, with 10 marked
        assert "8   line-8" in out
        assert "9   line-9" in out
        assert "10 → line-10" in out
        assert "11   line-11" in out
        assert "12   line-12" in out

    def test_marks_target_line(self):
        content = "a\nb\nc\n"
        out = _extract_code_snippet(content, line=2, context=5)
        assert "→ b" in out
        assert "  a" in out  # unmarked

    def test_clamps_to_file_bounds(self):
        content = "only-line"
        out = _extract_code_snippet(content, line=1, context=5)
        assert "only-line" in out

    def test_empty_content_returns_empty(self):
        assert _extract_code_snippet("", line=1) == ""

    def test_zero_line_returns_empty(self):
        assert _extract_code_snippet("a\nb\n", line=0) == ""

    def test_out_of_range_line_returns_empty(self):
        assert _extract_code_snippet("a\nb\n", line=10) == ""


class TestTruncateDiffForComment:
    """Relevance-biased truncation used when replying to diff comments."""

    def test_small_diff_unchanged(self):
        diff = "diff --git a/f b/f\n--- a/f\n+++ b/f\n+hi\n"
        assert _truncate_diff_for_comment(diff, "f") == diff

    def test_no_file_path_falls_back_to_head_truncation(self):
        from raven.server import _truncate_diff_for_comment as _t
        import raven.server as _srv
        with patch.object(_srv, "MAX_DIFF_LINES", 3):
            out = _t("a\nb\nc\nd\ne\n", file_path="")
        assert out.startswith("a\nb\nc\n")
        assert "truncated" in out

    def test_puts_named_file_first_when_too_large(self):
        """If the overall diff exceeds the limit but the named file fits,
        the named file's hunk must appear in the output even if it was
        originally at the end of the diff."""
        import raven.server as _srv
        early = "diff --git a/first b/first\n" + "+x\n" * 20
        late = "diff --git a/late b/late\n" + "+y\n" * 5
        diff = early + late
        with patch.object(_srv, "MAX_DIFF_LINES", 10):
            out = _srv._truncate_diff_for_comment(diff, file_path="late")
        assert "diff --git a/late b/late" in out
        # The oversized first-file chunk must be dropped.
        assert "diff --git a/first b/first" not in out

    def test_unknown_file_path_falls_back_to_head_truncation(self):
        """If the named file isn't in the diff, head-truncate as before."""
        import raven.server as _srv
        diff = "".join(f"line-{i}\n" for i in range(50))
        with patch.object(_srv, "MAX_DIFF_LINES", 5):
            out = _srv._truncate_diff_for_comment(diff, file_path="not-in-diff")
        assert out.startswith("line-0\nline-1\nline-2\nline-3\nline-4\n")

    def test_oversized_target_chunk_head_truncated_not_dropped(self):
        """If the target file's own chunk exceeds MAX_DIFF_LINES, it must
        still appear in the output (head-truncated) rather than being
        silently dropped — otherwise the function defeats its own purpose
        for the exact case it's meant to handle."""
        import raven.server as _srv
        big = "diff --git a/target b/target\n" + "+line\n" * 30
        other = "diff --git a/other b/other\n+x\n"
        diff = big + other
        with patch.object(_srv, "MAX_DIFF_LINES", 10):
            out = _srv._truncate_diff_for_comment(diff, file_path="target")
        assert "diff --git a/target b/target" in out
        # Unrelated file dropped when the target alone consumed the budget.
        assert "diff --git a/other b/other" not in out
        assert "truncated" in out

    def test_windows_around_commented_line_in_oversized_chunk(self):
        """When the target chunk is oversized and a hunk covers the
        commented-on line, the windower keeps that hunk so the line
        stays visible."""
        import raven.server as _srv
        header = "diff --git a/big b/big\n--- a/big\n+++ b/big\n"
        hunk_a = "@@ -1,0 +1,5 @@\n+early-1\n+early-2\n+early-3\n+early-4\n+early-5\n"
        hunk_b = "@@ -1,0 +50,5 @@\n+middle-50\n+middle-51\n+middle-52\n+middle-53\n+middle-54\n"
        hunk_c = "@@ -1,0 +100,5 @@\n+late-100\n+late-101\n+late-102\n+late-103\n+late-104\n"
        chunk = header + hunk_a + hunk_b + hunk_c
        diff = chunk + "diff --git a/other b/other\n" + "+x\n" * 50
        with patch.object(_srv, "MAX_DIFF_LINES", 10):
            out = _srv._truncate_diff_for_comment(diff, file_path="big", line=52)
        assert "middle-52" in out
        assert "+++ b/big" in out
        assert "truncated" in out

    def test_no_line_info_falls_back_to_head_truncation(self):
        """Without a line number, oversized target chunk is head-truncated."""
        import raven.server as _srv
        header = "diff --git a/big b/big\n--- a/big\n+++ b/big\n"
        chunk = header + "@@ -1,0 +1,3 @@\n" + "+body\n" * 40
        with patch.object(_srv, "MAX_DIFF_LINES", 8):
            out = _srv._truncate_diff_for_comment(chunk, file_path="big", line=0)
        assert out.startswith("diff --git a/big b/big")

    def test_line_outside_any_hunk_falls_back_to_head_truncation(self):
        """If the commented-on line sits outside any hunk, fall back."""
        import raven.server as _srv
        header = "diff --git a/big b/big\n--- a/big\n+++ b/big\n"
        hunk = "@@ -1,0 +1,3 @@\n+a\n+b\n+c\n"
        chunk = header + hunk + "\n" + "+filler\n" * 40
        with patch.object(_srv, "MAX_DIFF_LINES", 8):
            out = _srv._truncate_diff_for_comment(chunk, file_path="big", line=999)
        assert "diff --git a/big b/big" in out


class TestCachePersistence:
    """Test findings cache save/load and LRU eviction."""

    def setup_method(self):
        _previous_diffs.clear()

    def test_save_and_load_round_trip(self, tmp_path):
        from raven.server import CacheEntry
        cache_file = tmp_path / "raven" / "findings_cache.json"
        _previous_diffs["owner/repo#1"] = CacheEntry(
            timestamp=100.0,
            hashes={"a.py": "hash1"},
            findings={"a.py": [{"severity": "high", "message": "bug"}]},
        )
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path / "raven"):
            _save_cache()
            _previous_diffs.clear()
            _load_cache()
        assert "owner/repo#1" in _previous_diffs
        entry = _previous_diffs["owner/repo#1"]
        assert entry.timestamp == 100.0
        assert entry.hashes == {"a.py": "hash1"}
        assert entry.findings["a.py"][0]["message"] == "bug"

    def test_round_trip_restores_rebase_tolerance_state(self, tmp_path):
        """content_hashes/hunks must survive a restart. Dropping them
        silently reverts the PR to full re-review-on-rebase, and JSON has
        no tuples, so the (start, length) pairs need restoring by hand."""
        from raven.server import CacheEntry
        cache_file = tmp_path / "raven" / "findings_cache.json"
        _previous_diffs["owner/repo#1"] = CacheEntry(
            timestamp=100.0,
            hashes={"a.py": "raw1"},
            findings={"a.py": []},
            content_hashes={"a.py": "content1"},
            hunks={"a.py": [(7, 7), (40, 3)]},
        )
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path / "raven"):
            _save_cache()
            _previous_diffs.clear()
            _load_cache()
        entry = _previous_diffs["owner/repo#1"]
        assert entry.content_hashes == {"a.py": "content1"}
        assert entry.hunks == {"a.py": [(7, 7), (40, 3)]}

    def test_malformed_hunks_skip_only_that_entry(self, tmp_path):
        """A corrupt hunk row must fail into the per-entry guard, not
        reach _remap_carried_lines and blow up mid-review."""
        import json as _json
        from raven.reviewer import review_config_hash
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(_json.dumps({
            "_config_hash": review_config_hash(),
            "entries": {
                "owner/repo#1": {"timestamp": 1.0, "hashes": {}, "findings": {},
                                 "hunks": {"a.py": [[7]]}},
                "owner/repo#2": {"timestamp": 2.0, "hashes": {"b.py": "h"},
                                 "findings": {}},
            },
        }), encoding="utf-8")
        with patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        assert "owner/repo#1" not in _previous_diffs
        assert "owner/repo#2" in _previous_diffs

    def test_load_missing_file(self, tmp_path):
        cache_file = tmp_path / "nonexistent" / "cache.json"
        with patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()  # should not raise
        assert len(_previous_diffs) == 0

    def test_load_corrupt_file(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not json {{{", encoding="utf-8")
        with patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()  # should not raise
        assert len(_previous_diffs) == 0

    def test_lru_eviction(self):
        from raven.server import CacheEntry
        for i in range(_MAX_CACHED_PRS + 10):
            _previous_diffs[f"repo#{i}"] = CacheEntry(
                timestamp=float(i), hashes={}, findings={},
            )
        with patch("raven.server._save_cache"):
            _evict_cache()
        assert len(_previous_diffs) == _MAX_CACHED_PRS
        # Oldest entries (lowest timestamps) evicted first.
        for i in range(10):
            assert f"repo#{i}" not in _previous_diffs
        # Newest retained.
        assert f"repo#{_MAX_CACHED_PRS + 9}" in _previous_diffs

    # ── Migration & new-field tests (Task 1 of the comment-thread plan) ──

    def test_load_legacy_3tuple_entries_skipped(self, tmp_path):
        """Legacy 3-tuple cache entries (pre-2026-05-13) are no longer
        loadable — the loader treats them as malformed and skips them,
        re-warming from the next push. Operators with stale cache files
        on disk get a clean restart rather than an inconsistent state."""
        from raven.reviewer import review_config_hash
        cache_dir = tmp_path / "raven"
        cache_dir.mkdir()
        cache_file = cache_dir / "findings_cache.json"
        cache_file.write_text(json.dumps({
            "_config_hash": review_config_hash(),
            "entries": {
                "u/r#1": [1234567890.0, {"a.py": "h1"},
                          {"a.py": [{"severity": "low"}]}],
            },
        }))
        with patch("raven.server._CACHE_DIR", cache_dir), \
             patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        # Legacy entry skipped — cache empty after load.
        assert "u/r#1" not in _previous_diffs

    def test_load_new_dict_entries_round_trip(self, tmp_path):
        """New dict-shape entries round-trip with verdict + summary."""
        from raven.reviewer import review_config_hash
        cache_dir = tmp_path / "raven"
        cache_dir.mkdir()
        cache_file = cache_dir / "findings_cache.json"
        cache_file.write_text(json.dumps({
            "_config_hash": review_config_hash(),
            "entries": {"u/r#2": {
                "timestamp": 1700.0,
                "hashes": {"a.py": "h"},
                "findings": {"a.py": []},
                "verdict": "approve",
                "summary": "LGTM",
            }},
        }))
        with patch("raven.server._CACHE_DIR", cache_dir), \
             patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        entry = _previous_diffs["u/r#2"]
        assert entry.verdict == "approve"
        assert entry.summary == "LGTM"

    def test_coverage_gap_files_round_trip(self, tmp_path):
        """coverage_gap_files persists across save/load so the comment-
        flow gate survives a service restart."""
        from raven.server import CacheEntry
        cache_file = tmp_path / "raven" / "findings_cache.json"
        _previous_diffs["owner/repo#9"] = CacheEntry(
            timestamp=1.0, hashes={}, findings={},
            verdict="needs_work", summary="partial",
            coverage_gap_files=["big.py", "huge.py"],
        )
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path / "raven"):
            _save_cache()
            _previous_diffs.clear()
            _load_cache()
        assert _previous_diffs["owner/repo#9"].coverage_gap_files == ["big.py", "huge.py"]

    def test_load_entry_without_coverage_gap_files_defaults_empty(self, tmp_path):
        """Cache files written before the coverage_gap_files field exist
        on operator disks — entries WITHOUT the key must load cleanly as
        no-gap, not crash or block merges spuriously."""
        from raven.reviewer import review_config_hash
        cache_dir = tmp_path / "raven"
        cache_dir.mkdir()
        cache_file = cache_dir / "findings_cache.json"
        cache_file.write_text(json.dumps({
            "_config_hash": review_config_hash(),
            "entries": {"u/r#7": {
                "timestamp": 1700.0,
                "hashes": {"a.py": "h"},
                "findings": {"a.py": []},
                "verdict": "approve",
                "summary": "LGTM",
            }},
        }))
        with patch("raven.server._CACHE_DIR", cache_dir), \
             patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        entry = _previous_diffs["u/r#7"]
        assert entry.coverage_gap_files == []

    def test_config_hash_round_trips(self, tmp_path):
        """config_hash (Task 10 — per-entry cache invalidation) survives a
        save/load cycle so the config-hash gate still works after a
        service restart."""
        from raven.server import CacheEntry
        cache_file = tmp_path / "raven" / "findings_cache.json"
        _previous_diffs["owner/repo#10"] = CacheEntry(
            timestamp=1.0, hashes={}, findings={},
            verdict="approve", summary="ok",
            config_hash="abc123def4567890",
        )
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path / "raven"):
            _save_cache()
            _previous_diffs.clear()
            _load_cache()
        assert _previous_diffs["owner/repo#10"].config_hash == "abc123def4567890"

    def test_load_entry_without_config_hash_defaults_empty(self, tmp_path):
        """Cache files written before config_hash existed (every cache
        file on disk before this feature ships) must load cleanly with
        the '' legacy default, not crash or raise KeyError."""
        from raven.reviewer import review_config_hash
        cache_dir = tmp_path / "raven"
        cache_dir.mkdir()
        cache_file = cache_dir / "findings_cache.json"
        cache_file.write_text(json.dumps({
            "_config_hash": review_config_hash(),
            "entries": {"u/r#8": {
                "timestamp": 1700.0,
                "hashes": {"a.py": "h"},
                "findings": {"a.py": []},
                "verdict": "approve",
                "summary": "LGTM",
            }},
        }))
        with patch("raven.server._CACHE_DIR", cache_dir), \
             patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        entry = _previous_diffs["u/r#8"]
        assert entry.config_hash == ""

    def test_save_emits_new_dict_shape(self, tmp_path):
        """_save_cache serializes the new dict shape, not the legacy 3-tuple."""
        from raven.server import CacheEntry
        cache_dir = tmp_path / "raven"
        cache_dir.mkdir()
        cache_file = cache_dir / "findings_cache.json"
        _previous_diffs["u/r#3"] = CacheEntry(
            timestamp=1.0, hashes={}, findings={},
            verdict="needs_work", summary="see findings",
        )
        with patch("raven.server._CACHE_DIR", cache_dir), \
             patch("raven.server._CACHE_FILE", cache_file):
            _save_cache()
        data = json.loads(cache_file.read_text())
        entry = data["entries"]["u/r#3"]
        assert isinstance(entry, dict)
        assert entry["verdict"] == "needs_work"
        assert entry["summary"] == "see findings"

    def test_config_hash_match_loads_cache(self, tmp_path):
        """Cache loads when config hash matches."""
        cache_file = tmp_path / "cache.json"
        _previous_diffs["owner/repo#1"] = CacheEntry(timestamp=100.0, hashes={"a.py": "h"}, findings={"a.py": []})
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path):
            _save_cache()
            _previous_diffs.clear()
            _load_cache()
        assert "owner/repo#1" in _previous_diffs

    def test_config_hash_mismatch_wipes_cache(self, tmp_path):
        """Cache discarded when config hash differs (model/prompt change)."""
        cache_file = tmp_path / "cache.json"
        _previous_diffs["owner/repo#1"] = CacheEntry(timestamp=100.0, hashes={"a.py": "h"}, findings={"a.py": []})
        with patch("raven.server._CACHE_FILE", cache_file), \
             patch("raven.server._CACHE_DIR", tmp_path):
            _save_cache()
            _previous_diffs.clear()
            # Simulate config change by returning a different hash on load
            with patch("raven.server.review_config_hash", return_value="different_hash"):
                _load_cache()
        assert len(_previous_diffs) == 0

    def test_missing_config_hash_treated_as_mismatch(self, tmp_path):
        """Old-format cache file without _config_hash is discarded."""
        cache_file = tmp_path / "cache.json"
        # Write old format (no _config_hash, flat dict)
        import json as _json
        cache_file.write_text(_json.dumps({"owner/repo#1": [100.0, {}, {}]}), encoding="utf-8")
        with patch("raven.server._CACHE_FILE", cache_file):
            _load_cache()
        assert len(_previous_diffs) == 0

    def test_save_failure_increments_metric(self, tmp_path):
        """Disk write failures must surface as a metric — silent WARNING
        logging alone leaves operators unable to alert on persistent
        cache-persistence breakage."""
        _previous_diffs["owner/repo#1"] = CacheEntry(
            timestamp=100.0, hashes={"a.py": "h"}, findings={"a.py": []},
        )
        with patch("raven.server._CACHE_DIR", tmp_path), \
             patch("raven.server._CACHE_FILE", tmp_path / "cache.json"), \
             patch("raven.server.os.replace", side_effect=PermissionError("denied")), \
             patch("raven.server.inc") as mock_inc:
            _save_cache()
        # Verify the metric fired with the exception type as the reason.
        assert any(
            call.args[0] == "raven_cache_save_failures_total"
            and call.args[1].get("reason") == "PermissionError"
            for call in mock_inc.call_args_list
        ), f"Expected raven_cache_save_failures_total inc; got: {mock_inc.call_args_list}"


class TestBitbucketDCWebhook:
    """Integration tests for Bitbucket Data Center webhook endpoint."""

    BB_SECRET = "bb-test-secret"
    BB_URL = "https://bitbucket.example.com"
    BB_TOKEN = "bb-test-token"
    BB_USERNAME = "raven-bot"

    @pytest.fixture(autouse=True)
    def _setup(self):
        _providers.clear()
        _recent_prs.clear()
        env = {
            "BITBUCKET_DC_URL": self.BB_URL,
            "BITBUCKET_DC_TOKEN": self.BB_TOKEN,
            "BITBUCKET_DC_WEBHOOK_SECRET": self.BB_SECRET,
            "BITBUCKET_DC_USERNAME": self.BB_USERNAME,
            # Clear Gitea env vars so only BB DC is registered
            "GITEA_URL": "",
            "GITEA_TOKEN": "",
            "GITEA_WEBHOOK_SECRET": "",
            
        }
        with patch.dict(os.environ, env):
            app = create_app()
            app.config["TESTING"] = True
            self._app = app
            self._client = app.test_client()
            self._provider = _providers["bitbucket-dc"]
        yield
        _providers.clear()
        _recent_prs.clear()

    def _sign_bb(self, body: bytes, secret: str = None) -> str:
        secret = secret or self.BB_SECRET
        return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    def _post_bb_dc(self, payload_dict, event_key, secret=None):
        body = json.dumps(payload_dict).encode()
        sig = self._sign_bb(body, secret)
        return self._client.post(
            "/hook/bitbucket-dc",
            data=body,
            headers={
                "X-Hub-Signature": sig,
                "X-Event-Key": event_key,
                "Content-Type": "application/json",
            },
        )

    # -- PR opened --------------------------------------------------------- #

    def test_pr_opened_triggers_review(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "title": "Add feature",
                "fromRef": {
                    "displayId": "feature-branch",
                    "latestCommit": "aaa111",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "toRef": {
                    "displayId": "main",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "links": {"self": [{"href": "https://bb/pr/10"}]},
            },
        }
        with patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:opened")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    # -- Push triggers re-review ------------------------------------------- #

    def test_push_triggers_re_review(self):
        payload = {
            "actor": {"slug": "alice"},
            "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            "changes": [
                {
                    "ref": {"type": "BRANCH", "displayId": "feature-branch"},
                    "toHash": "bbb222",
                }
            ],
        }
        pr_dict = {
            "number": 10,
            "title": "Add feature",
            "html_url": "https://bb/pr/10",
            "head": {"sha": "bbb222", "ref": "feature-branch"},
            "base": {"ref": "main"},
        }
        with patch.object(self._provider, "find_open_pr_for_branch", return_value=pr_dict), \
             patch.object(self._provider, "_get_default_branch", return_value="main"), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "repo:refs_changed")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "accepted"
        assert data["reason"] == "re-review triggered"
        mock_executor.submit.assert_called_once()

    # -- Tag push ignored -------------------------------------------------- #

    def test_tag_push_ignored(self):
        payload = {
            "actor": {"slug": "alice"},
            "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            "changes": [
                {
                    "ref": {"type": "TAG", "displayId": "v1.0.0"},
                    "toHash": "ccc333",
                }
            ],
        }
        resp = self._post_bb_dc(payload, "repo:refs_changed")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ignored"

    # -- Comment with mention triggers response ---------------------------- #

    def test_comment_mention_triggers_response(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 555,
                "text": f"@{self.BB_USERNAME} explain this",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "accepted"
        assert data["reason"] == "responding to comment"
        mock_executor.submit.assert_called_once()

    # -- Comment without mention ignored ----------------------------------- #

    def test_comment_without_mention_ignored(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 556,
                "text": "looks good to me",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME):
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ignored"

    # -- Reply inside Raven's thread ---------------------------------------- #

    def test_reply_in_raven_thread_triggers_response_without_mention(self):
        """When a user replies to one of Raven's comments, Raven should
        evaluate/respond even without an @mention."""
        payload = {
            "actor": {"slug": "alice"},
            "commentParentId": 700,
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 701,
                "text": "ok but how?",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch.object(self._provider, "get_comment_thread",
                          return_value=["alice", self.BB_USERNAME]), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_reply_in_deep_thread_triggers_when_raven_replied_midway(self):
        """BB DC sets commentParentId to the thread root. If Raven replied
        inside that thread (not as the root), the auto-respond-without-mention
        feature must still trigger — needs a full thread walk, not just
        root-author check."""
        payload = {
            "actor": {"slug": "alice"},
            "commentParentId": 700,  # root = alice, not Raven
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 705,
                "text": "follow-up",
                "author": {"slug": "alice"},
            },
        }
        # Thread authors: alice (root) + raven-bot (reply) + alice (reply)
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch.object(self._provider, "get_comment_thread",
                          return_value=["alice", self.BB_USERNAME]), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_reply_in_other_users_thread_still_ignored_without_mention(self):
        payload = {
            "actor": {"slug": "alice"},
            "commentParentId": 800,
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 801,
                "text": "yeah agreed",
                "author": {"slug": "alice"},
            },
        }
        # Webhook always returns 200 accepted — the background worker does
        # the thread lookup and decides to skip when Raven isn't involved.
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    def test_webhook_returns_200_without_calling_thread_lookup(self):
        """Provider thread API must not be hit on the webhook hot path —
        the worker does that asynchronously so the webhook stays fast
        even when the provider is slow or unreachable."""
        payload = {
            "actor": {"slug": "alice"},
            "commentParentId": 900,
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 901,
                "text": "ping",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch.object(self._provider, "get_comment_thread") as mock_lookup, \
             patch("raven.server.executor"):
            resp = self._post_bb_dc(payload, "pr:comment:added")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "accepted"
        mock_lookup.assert_not_called()

    # -- Edited comment with bumped version triggers reprocessing ---------- #

    def test_edited_comment_dispatches_when_version_bumped(self):
        """pr:comment:edited carries comment.version (incremented per edit).
        The server's dedup key includes the version so a re-edit of the
        same comment id gets a distinct slot — letting a user add @raven
        to an existing comment and have the edit trigger a reply."""
        base_payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 901, "version": 1,
                "text": "looking good @raven-bot",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch("raven.server.executor") as mock_executor:
            resp_add = self._post_bb_dc(base_payload, "pr:comment:added")
            # Second delivery for the SAME comment_id but bumped version:
            edited = dict(base_payload)
            edited["comment"] = {
                "id": 901, "version": 2,
                "text": "looking good — @raven-bot please re-check",
                "author": {"slug": "alice"},
            }
            edited["previousComment"] = "looking good @raven-bot"
            resp_edit = self._post_bb_dc(edited, "pr:comment:edited")
        assert resp_add.status_code == 200
        assert resp_add.get_json()["status"] == "accepted"
        assert resp_edit.status_code == 200
        # Both deliveries dispatched — version-aware dedup keeps the edit
        # from collapsing onto the original add's slot.
        assert resp_edit.get_json()["status"] == "accepted"
        assert mock_executor.submit.call_count == 2

    def test_repeat_added_with_same_version_dedups(self):
        """Sanity: webhook retry with the SAME (id, version) hits the
        existing dedup slot. The version-aware key still collapses
        identical deliveries."""
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "toRef": {
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
            },
            "comment": {
                "id": 902, "version": 1,
                "text": "@raven-bot have a look",
                "author": {"slug": "alice"},
            },
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch("raven.server.executor") as mock_executor:
            first = self._post_bb_dc(payload, "pr:comment:added")
            second = self._post_bb_dc(payload, "pr:comment:added")
        assert first.get_json()["status"] == "accepted"
        assert second.get_json()["status"] == "skipped"
        assert second.get_json()["reason"] == "duplicate"
        assert mock_executor.submit.call_count == 1

    # -- Review-state events route to existing no-op consumers ------------- #

    def test_reviewer_approved_routes_to_no_action(self):
        """Parity with Gitea's pull_request_review_approved. Consumer is
        a no-op (route preserved so deliveries return a clean ignored
        response), but the response must NOT be "unhandled"."""
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10, "title": "x",
                "fromRef": {"displayId": "f", "latestCommit": "a",
                            "repository": {"slug": "my-repo", "project": {"key": "PROJ"}}},
                "toRef": {"displayId": "main",
                          "repository": {"slug": "my-repo", "project": {"key": "PROJ"}}},
                "links": {"self": [{"href": "https://bb/pr/10"}]},
            },
        }
        with patch("raven.server.executor"):
            resp = self._post_bb_dc(payload, "pr:reviewer:approved")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ignored"
        assert "no action" in data["reason"]

    def test_reviewer_changes_requested_routes_to_no_action(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10, "title": "x",
                "fromRef": {"displayId": "f", "latestCommit": "a",
                            "repository": {"slug": "my-repo", "project": {"key": "PROJ"}}},
                "toRef": {"displayId": "main",
                          "repository": {"slug": "my-repo", "project": {"key": "PROJ"}}},
                "links": {"self": [{"href": "https://bb/pr/10"}]},
            },
        }
        with patch("raven.server.executor"):
            resp = self._post_bb_dc(payload, "pr:reviewer:changes_requested")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ignored"

    # -- Reviewer updated triggers review ---------------------------------- #

    def test_reviewer_updated_triggers_review(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 10,
                "title": "Add feature",
                "fromRef": {
                    "displayId": "feature-branch",
                    "latestCommit": "aaa111",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "toRef": {
                    "displayId": "main",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "links": {"self": [{"href": "https://bb/pr/10"}]},
            },
            "addedReviewers": [{"slug": self.BB_USERNAME}],
        }
        with patch.object(self._provider, "get_authenticated_user", return_value=self.BB_USERNAME), \
             patch("raven.server.executor") as mock_executor:
            resp = self._post_bb_dc(payload, "pr:reviewer:updated")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "accepted"
        mock_executor.submit.assert_called_once()

    # -- Invalid signature rejected ---------------------------------------- #

    def test_invalid_signature_rejected(self):
        payload = {"actor": {"slug": "alice"}}
        resp = self._post_bb_dc(payload, "pr:opened", secret="wrong-secret")
        assert resp.status_code == 403


class TestBitbucketDCWebhookRouting:
    """Verify that BB DC pr:opened webhook is routed and dispatches a review."""

    BB_SECRET = "testsecret"
    BB_URL = "https://bitbucket.example.com"
    BB_TOKEN = "bb-test-token"
    BB_USERNAME = "raven-bot"

    @pytest.fixture(autouse=True)
    def _setup(self):
        _providers.clear()
        _recent_prs.clear()
        env = {
            "BITBUCKET_DC_URL": self.BB_URL,
            "BITBUCKET_DC_TOKEN": self.BB_TOKEN,
            "BITBUCKET_DC_WEBHOOK_SECRET": self.BB_SECRET,
            "BITBUCKET_DC_USERNAME": self.BB_USERNAME,
            "GITEA_URL": "",
            "GITEA_TOKEN": "",
            "GITEA_WEBHOOK_SECRET": "",
            
        }
        with patch.dict(os.environ, env):
            app = create_app()
            app.config["TESTING"] = True
            self._client = app.test_client()
        yield
        _providers.clear()
        _recent_prs.clear()

    def _sign_bb(self, body: bytes) -> str:
        return "sha256=" + hmac.new(
            self.BB_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()

    def test_bb_dc_pr_opened_dispatches_review(self):
        payload = {
            "actor": {"slug": "alice"},
            "pullRequest": {
                "id": 5,
                "title": "Implement widget",
                "fromRef": {
                    "displayId": "feature/widget",
                    "latestCommit": "deadbeef",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "toRef": {
                    "displayId": "main",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "links": {"self": [{"href": "https://bb/pr/5"}]},
            },
        }
        body = json.dumps(payload).encode()
        sig = self._sign_bb(body)
        with patch("raven.server._process_pr"):
            resp = self._client.post(
                "/hook/bitbucket-dc",
                data=body,
                headers={
                    "X-Hub-Signature": sig,
                    "X-Event-Key": "pr:opened",
                    "Content-Type": "application/json",
                },
            )
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "accepted"}


class TestShouldAutoAddReviewer:
    """Direct unit tests for the auto-add gate. Indirect coverage via
    _process_pr exists, but the helper's contract (case-insensitive
    match, empty-login tolerance, sole-Raven-counts-as-empty) is worth
    pinning down separately."""

    def _mc(self, raven_user="raven-bot", reviews=None, requested=None):
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = raven_user
        mc.get_pr_reviews.return_value = reviews or []
        mc.get_pr_requested_reviewers.return_value = requested or []
        return mc

    def test_no_reviewers_returns_true(self):
        from raven.server import _should_auto_add_reviewer
        mc = self._mc()
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is True

    def test_advisory_mode_never_auto_adds(self, mocker):
        """Advisory mode short-circuits to False before any provider call.
        Auto-adding Raven would itself block the merge (Raven listed as
        reviewer without formal approval = blocked) — defeats advisory."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "advisory")
        from raven.server import _should_auto_add_reviewer
        mc = self._mc()
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is False
        # Short-circuit before any API hit.
        mc.get_authenticated_user.assert_not_called()
        mc.get_pr_reviews.assert_not_called()
        mc.get_pr_requested_reviewers.assert_not_called()

    def test_human_reviewer_returns_false_in_fill_gap_mode(self, mocker):
        """Fill-gap mode: human reviewer present → don't add Raven."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "gap")
        from raven.server import _should_auto_add_reviewer
        mc = self._mc(reviews=[{"user": {"login": "alice"}, "state": "COMMENT"}])
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is False

    def test_human_requested_returns_false_in_fill_gap_mode(self, mocker):
        """Fill-gap mode: human requested reviewer present → don't add Raven."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "gap")
        from raven.server import _should_auto_add_reviewer
        mc = self._mc(requested=["alice"])
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is False

    def test_case_insensitive_raven_match(self):
        """Some providers normalize login casing differently (BB DC
        lowercases slugs; Gitea preserves case). The gate must detect
        Raven's own entry regardless of case and not re-add."""
        from raven.server import _should_auto_add_reviewer
        mc = self._mc(
            raven_user="Raven-Bot",
            reviews=[{"user": {"login": "raven-bot"}, "state": "APPROVED"}],
            requested=["RAVEN-BOT"],
        )
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is False

    def test_empty_login_entries_ignored(self):
        """A review with an empty/null login is neither Raven nor a
        human — ignore it rather than short-circuiting to False."""
        from raven.server import _should_auto_add_reviewer
        mc = self._mc(
            reviews=[{"user": {"login": ""}, "state": "COMMENT"},
                     {"user": {"login": None}, "state": "COMMENT"},
                     {"user": None, "state": "COMMENT"}],
            requested=["", None],
        )
        assert _should_auto_add_reviewer(mc, "owner/repo", 1) is True

    def test_auto_add_true_when_all_prs_flag_set_and_not_reviewer(self, mocker):
        """In RAVEN_REVIEW_MODE="all" mode, Raven auto-adds even when
        a human reviewer is listed, so long as Raven itself isn't."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "all")
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "alice"}, "state": "COMMENTED"}]
        mc.get_pr_requested_reviewers.return_value = ["bob"]
        from raven.server import _should_auto_add_reviewer
        assert _should_auto_add_reviewer(mc, "owner/repo", 42) is True

    def test_auto_add_false_when_all_prs_flag_set_and_raven_already_reviewer(self, mocker):
        """RAVEN_REVIEW_MODE="all" must still be idempotent — don't
        re-add if Raven is already a reviewer."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "all")
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        from raven.server import _should_auto_add_reviewer
        assert _should_auto_add_reviewer(mc, "owner/repo", 42) is False

    def test_auto_add_false_when_all_prs_flag_set_and_raven_requested(self, mocker):
        """Idempotent: if Raven is already in requested reviewers, no add."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "all")
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = []
        mc.get_pr_requested_reviewers.return_value = ["Raven"]
        from raven.server import _should_auto_add_reviewer
        assert _should_auto_add_reviewer(mc, "owner/repo", 42) is False

    def test_auto_add_false_in_fill_gap_mode_with_other_reviewer(self, mocker):
        """Fill-gap mode preserves the PR #101 behaviour — decline when
        any human reviewer is present."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "gap")
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "alice"}, "state": "COMMENTED"}]
        mc.get_pr_requested_reviewers.return_value = []
        from raven.server import _should_auto_add_reviewer
        assert _should_auto_add_reviewer(mc, "owner/repo", 42) is False

    def test_auto_add_true_in_fill_gap_mode_with_no_others(self, mocker):
        """Fill-gap mode: no humans, Raven is welcome."""
        mocker.patch("raven.server.RAVEN_REVIEW_MODE", "gap")
        mc = MagicMock(spec=GitProvider)
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = []
        mc.get_pr_requested_reviewers.return_value = []
        from raven.server import _should_auto_add_reviewer
        assert _should_auto_add_reviewer(mc, "owner/repo", 42) is True


class TestDoMerge:
    """Test _do_merge SHA re-check and head_sha pass-through."""

    def setup_method(self):
        _recent_prs.clear()

    def test_sha_recheck_blocks_merge_when_changed(self):
        """Provider-agnostic SHA re-check prevents merge after force-push during CI wait."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_commit_status.return_value = "success"
        mc.get_pr_head_sha.return_value = "newsha456"  # Changed during CI wait
        review = {"severity": "low", "summary": "ok", "findings": []}
        with patch("raven.server.time.sleep"):
            _do_merge(mc, "owner/repo", 42, "My PR", "http://x", review, "abc123", "squash")
        mc.merge_pr.assert_not_called()

    def test_sha_recheck_fails_closed_on_api_error(self):
        """If SHA re-check API call fails, skip merge (fail closed)."""
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_commit_status.return_value = "success"
        mc.get_pr_head_sha.side_effect = Exception("connection refused")
        review = {"severity": "low", "summary": "ok", "findings": []}
        with patch("raven.server.time.sleep"):
            _do_merge(mc, "owner/repo", 42, "My PR", "http://x", review, "abc123", "squash")
        mc.merge_pr.assert_not_called()

    def test_sha_recheck_allows_merge_when_unchanged(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_commit_status.return_value = "success"
        mc.get_pr_head_sha.return_value = "abc123"  # Same as original
        mc.merge_pr.return_value = True
        review = {"severity": "low", "summary": "ok", "findings": []}
        with patch("raven.server.time.sleep"):
            _do_merge(mc, "owner/repo", 42, "My PR", "http://x", review, "abc123", "squash")
        mc.merge_pr.assert_called_once()


class TestGiteaAutoMerge:
    """Test RAVEN_GITEA_AUTO_MERGE option."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def test_auto_merge_passes_merge_when_checks_succeed(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.merge_pr.return_value = True
        review = {"severity": "low", "summary": "ok", "findings": []}
        with patch("raven.server._GITEA_AUTO_MERGE", True):
            _do_merge(mc, "owner/repo", 42, "My PR", "http://x", review, "abc123", "squash")
        mc.merge_pr.assert_called_once_with(
            "owner/repo", 42, commit_title="My PR", strategy="squash",
            head_sha="abc123", merge_when_checks_succeed=True,
        )

    def test_non_gitea_provider_polls_ci(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "bitbucket-dc"
        mc.get_commit_status.return_value = "success"
        mc.get_pr_head_sha.return_value = "abc123"  # Must match for SHA re-check
        mc.merge_pr.return_value = True
        review = {"severity": "low", "summary": "ok", "findings": []}
        with patch("raven.server._GITEA_AUTO_MERGE", True), \
             patch("raven.server.time.sleep"):
            _do_merge(mc, "owner/repo", 42, "My PR", "http://x", review, "abc123", "squash")
        # BB DC should use regular merge (not merge_when_checks_succeed)
        mc.merge_pr.assert_called_once()
        assert mc.merge_pr.call_args.kwargs.get("merge_when_checks_succeed") is not True


class TestFetchPromptOverride:
    """The per-repo prompt-override fetch helper.

    Returns the override string on success; None on missing file, fetch
    error, empty or whitespace-only content, or when RULES_DIR is empty.
    """

    def _make_provider(self, fetch_file_impl):
        mp = MagicMock()
        mp.fetch_file.side_effect = fetch_file_impl
        return mp

    def test_returns_override_on_success(self, mocker):
        mocker.patch("raven.server.RULES_DIR", ".claude/rules")
        from raven.server import _fetch_prompt_override
        provider = self._make_provider(
            lambda repo, path, ref=None: "OVERRIDE PROMPT BODY"
        )
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result == "OVERRIDE PROMPT BODY"

    def test_returns_none_on_missing_file(self, mocker):
        mocker.patch("raven.server.RULES_DIR", ".claude/rules")
        from raven.server import _fetch_prompt_override
        provider = self._make_provider(
            lambda repo, path, ref=None: (_ for _ in ()).throw(FileNotFoundError())
        )
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result is None

    def test_returns_none_on_generic_fetch_error(self, mocker):
        mocker.patch("raven.server.RULES_DIR", ".claude/rules")
        from raven.server import _fetch_prompt_override
        provider = self._make_provider(
            lambda repo, path, ref=None: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result is None

    def test_returns_none_on_empty_content(self, mocker):
        mocker.patch("raven.server.RULES_DIR", ".claude/rules")
        from raven.server import _fetch_prompt_override
        provider = self._make_provider(lambda repo, path, ref=None: "")
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result is None

    def test_returns_none_on_whitespace_only_content(self, mocker):
        mocker.patch("raven.server.RULES_DIR", ".claude/rules")
        from raven.server import _fetch_prompt_override
        provider = self._make_provider(lambda repo, path, ref=None: "  \n\t\n  ")
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result is None

    def test_returns_none_when_both_config_dirs_empty(self, mocker):
        mocker.patch("raven.server.CONFIG_DIR", "")
        mocker.patch("raven.server.RULES_DIR", "")
        from raven.server import _fetch_prompt_override
        provider = MagicMock()
        provider.fetch_file.side_effect = AssertionError("fetch_file should not be called")
        result = _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert result is None
        provider.fetch_file.assert_not_called()

    def test_constructs_correct_path_for_review(self, mocker):
        mocker.patch("raven.server.CONFIG_DIR", ".raven")
        from raven.server import _fetch_prompt_override
        captured = {}
        def fetch_file(repo, path, ref=None):
            captured["path"] = path
            captured["ref"] = ref
            return "body"
        provider = self._make_provider(fetch_file)
        _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert captured["path"] == ".raven/prompts/review.md"
        assert captured["ref"] == "main"

    def test_constructs_correct_path_for_respond(self, mocker):
        mocker.patch("raven.server.CONFIG_DIR", ".raven")
        from raven.server import _fetch_prompt_override
        captured = {}
        def fetch_file(repo, path, ref=None):
            captured["path"] = path
            return "body"
        provider = self._make_provider(fetch_file)
        _fetch_prompt_override(provider, "owner/repo", "main", "respond")
        assert captured["path"] == ".raven/prompts/respond.md"

    def test_legacy_fallback_honours_custom_rules_dir(self, mocker):
        mocker.patch("raven.server.CONFIG_DIR", "")
        mocker.patch("raven.server.RULES_DIR", ".custom/dir")
        from raven.server import _fetch_prompt_override
        captured = {}
        def fetch_file(repo, path, ref=None):
            captured["path"] = path
            return "body"
        provider = self._make_provider(fetch_file)
        _fetch_prompt_override(provider, "owner/repo", "main", "review")
        assert captured["path"] == ".custom/dir/raven/prompts/review.md"


# ------------------------------------------------------------------ #
#  .raven/ config dir + legacy .claude/rules/raven/ fallback          #
# ------------------------------------------------------------------ #


class TestRepoConfigDirResolution:
    """Raven's per-repo config (prompt overrides, severities.json) lives
    under ``RAVEN_CONFIG_DIR`` (default ``.raven``), NOT under
    ``.claude/`` — everything below ``.claude/`` is swept into every
    other agent's context in that repo, and a Raven prompt override is
    noise to all of them.

    The pre-move ``{RULES_DIR}/raven/`` home is still read as a fallback
    so existing repos keep working. A hit there is reported back through
    ``on_legacy_path`` so the review body can nag about it.
    """

    NEW_REVIEW = ".raven/prompts/review.md"
    NEW_RESPOND = ".raven/prompts/respond.md"
    NEW_SCALE = ".raven/severities.json"
    OLD_REVIEW = ".claude/rules/raven/prompts/review.md"
    OLD_RESPOND = ".claude/rules/raven/prompts/respond.md"
    OLD_SCALE = ".claude/rules/raven/severities.json"

    SCALE_BODY = '{"severities": {"nit": 1, "bad": 2}}'

    def _provider(self, files):
        """Provider whose fetch_file serves ``files`` and returns "" (the
        providers' real 404 behaviour) for everything else."""
        mp = MagicMock()
        mp.fetch_file.side_effect = (
            lambda repo, path, ref=None: files.get(path, ""))
        return mp

    def _paths(self, mp):
        return [c.args[1] for c in mp.fetch_file.call_args_list]

    def _std_dirs(self, mocker, config_dir=".raven", rules_dir=".claude/rules"):
        import raven.server as server
        mocker.patch.object(server, "CONFIG_DIR", config_dir)
        mocker.patch.object(server, "RULES_DIR", rules_dir)

    # ── prompt overrides ─────────────────────────────────────────── #

    def test_review_override_reads_the_new_path_first(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.NEW_REVIEW: "NEW BODY"})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") == "NEW BODY"
        assert self._paths(mp)[0] == self.NEW_REVIEW

    def test_respond_override_reads_the_new_path_first(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.NEW_RESPOND: "NEW BODY"})
        assert _fetch_prompt_override(mp, "o/r", "main", "respond") == "NEW BODY"
        assert self._paths(mp)[0] == self.NEW_RESPOND

    def test_new_path_hit_never_probes_the_legacy_path(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.NEW_REVIEW: "NEW", self.OLD_REVIEW: "OLD"})
        seen = []
        result = _fetch_prompt_override(mp, "o/r", "main", "review",
                                        on_legacy_path=seen.append)
        assert result == "NEW"
        assert self.OLD_REVIEW not in self._paths(mp)
        assert seen == []

    def test_falls_back_to_the_legacy_path(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.OLD_REVIEW: "OLD BODY"})
        seen = []
        result = _fetch_prompt_override(mp, "o/r", "main", "review",
                                        on_legacy_path=seen.append)
        assert result == "OLD BODY"
        assert self._paths(mp) == [self.NEW_REVIEW, self.OLD_REVIEW]

    def test_legacy_hit_reports_the_relative_path(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.OLD_RESPOND: "OLD BODY"})
        seen = []
        _fetch_prompt_override(mp, "o/r", "main", "respond",
                               on_legacy_path=seen.append)
        assert seen == ["prompts/respond.md"]

    def test_whitespace_only_new_path_falls_through_to_legacy(self, mocker):
        """An empty override is "no override" on either path — a repo that
        blanked the new file must not lose its legacy one silently."""
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.NEW_REVIEW: "  \n\t ", self.OLD_REVIEW: "OLD"})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") == "OLD"

    def test_new_path_fetch_error_still_tries_legacy(self, mocker):
        self._std_dirs(mocker)
        from raven.server import _fetch_prompt_override

        def fetch(repo, path, ref=None):
            if path == self.NEW_REVIEW:
                raise RuntimeError("transport blew up")
            return "OLD BODY"

        mp = MagicMock()
        mp.fetch_file.side_effect = fetch
        assert _fetch_prompt_override(mp, "o/r", "main", "review") == "OLD BODY"

    def test_config_dir_empty_disables_the_new_path_only(self, mocker):
        self._std_dirs(mocker, config_dir="")
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.OLD_REVIEW: "OLD"})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") == "OLD"
        assert self._paths(mp) == [self.OLD_REVIEW]

    def test_rules_dir_empty_disables_the_legacy_path_only(self, mocker):
        self._std_dirs(mocker, rules_dir="")
        from raven.server import _fetch_prompt_override
        mp = self._provider({self.OLD_REVIEW: "OLD"})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") is None
        assert self._paths(mp) == [self.NEW_REVIEW]

    def test_both_dirs_empty_fetches_nothing(self, mocker):
        self._std_dirs(mocker, config_dir="", rules_dir="")
        from raven.server import _fetch_prompt_override
        mp = self._provider({})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") is None
        mp.fetch_file.assert_not_called()

    def test_honours_a_custom_config_dir(self, mocker):
        self._std_dirs(mocker, config_dir=".ravenconf")
        from raven.server import _fetch_prompt_override
        mp = self._provider({".ravenconf/prompts/review.md": "BODY"})
        assert _fetch_prompt_override(mp, "o/r", "main", "review") == "BODY"

    # ── severity scale ───────────────────────────────────────────── #

    def test_scale_reads_the_new_path_first(self, mocker):
        self._std_dirs(mocker)
        import raven.server as server
        mp = self._provider({self.NEW_SCALE: self.SCALE_BODY})
        scale = server._fetch_severity_scale(mp, "o/r", "main")
        assert scale.ordered() == ["bad", "nit"]
        assert self._paths(mp)[0] == self.NEW_SCALE

    def test_scale_falls_back_to_the_legacy_path_and_reports_it(self, mocker):
        self._std_dirs(mocker)
        import raven.server as server
        mp = self._provider({self.OLD_SCALE: self.SCALE_BODY})
        seen = []
        scale = server._fetch_severity_scale(mp, "o/r", "main",
                                             on_legacy_path=seen.append)
        assert scale.ordered() == ["bad", "nit"]
        assert seen == ["severities.json"]

    def test_scale_new_path_wins_over_legacy(self, mocker):
        self._std_dirs(mocker)
        import raven.server as server
        mp = self._provider({
            self.NEW_SCALE: '{"severities": {"new": 1, "newer": 2}}',
            self.OLD_SCALE: self.SCALE_BODY,
        })
        assert server._fetch_severity_scale(mp, "o/r", "main").ordered() == \
            ["newer", "new"]

    def test_rules_dir_empty_no_longer_disables_the_scale(self, mocker):
        """Behaviour change on upgrade: RAVEN_RULES_DIR="" used to be the
        kill switch for the scale file too. It now only disables the
        legacy path — RAVEN_CONFIG_DIR="" is the new kill switch."""
        self._std_dirs(mocker, rules_dir="")
        import raven.server as server
        mp = self._provider({self.NEW_SCALE: self.SCALE_BODY})
        assert server._fetch_severity_scale(mp, "o/r", "main").ordered() == \
            ["bad", "nit"]

    def test_both_dirs_empty_skips_the_scale_fetch(self, mocker):
        self._std_dirs(mocker, config_dir="", rules_dir="")
        import raven.server as server
        from raven.severity import default_scale
        mp = self._provider({})
        assert server._fetch_severity_scale(mp, "o/r", "main").ranks == \
            default_scale().ranks
        mp.fetch_file.assert_not_called()

    def test_scale_fetch_failure_on_both_paths_still_fails_the_merge_closed(self, mocker):
        """The on_fetch_failed contract predates the move and must survive
        it: a scale Raven merely failed to READ must not silently
        substitute the looser built-in gate on a merge-capable path."""
        self._std_dirs(mocker)
        import raven.server as server
        mocker.patch.object(server, "inc")
        mp = MagicMock()
        mp.fetch_file.side_effect = RuntimeError("boom")
        failed = []
        server._fetch_severity_scale(mp, "o/r", "main",
                                     on_fetch_failed=lambda: failed.append(True))
        assert failed == [True]

    def test_new_path_read_failure_fails_the_merge_closed_even_with_a_legacy_scale(self, mocker):
        """The non-obvious half of the fail-closed contract: a review that
        got a perfectly usable scale off the legacy path STILL refuses to
        merge, because the `.raven` file it could not read may be a
        stricter scale that supersedes it. Nothing about the returned
        object shows this — only on_fetch_failed does — so a refactor
        that dropped the flag once any path succeeded would look correct
        and silently re-open the gate."""
        self._std_dirs(mocker)
        import raven.server as server
        mocker.patch.object(server, "inc")

        def fetch(repo, path, ref=None):
            if path == self.NEW_SCALE:
                raise RuntimeError("transport blew up")
            return self.SCALE_BODY

        mp = MagicMock()
        mp.fetch_file.side_effect = fetch
        failed = []
        scale = server._fetch_severity_scale(
            mp, "o/r", "main", on_fetch_failed=lambda: failed.append(True))

        # The legacy scale governs the review that still posts...
        assert scale.ordered() == ["bad", "nit"]
        # ...but the merge gate is closed regardless.
        assert failed == [True]

    def test_scale_absent_on_both_paths_does_not_fail_closed(self, mocker):
        """"No file anywhere" is the normal case for most repos and must
        stay a silent default — only a READ FAILURE fails closed."""
        self._std_dirs(mocker)
        import raven.server as server
        mp = self._provider({})
        failed = []
        server._fetch_severity_scale(mp, "o/r", "main",
                                     on_fetch_failed=lambda: failed.append(True))
        assert failed == []


class TestLegacyConfigPathNote:
    """Every review that read config from the deprecated path carries a
    migration note in its body — the in-band nag that replaces a
    dashboard metric."""

    def _dirs(self, mocker):
        import raven.server as server
        mocker.patch.object(server, "CONFIG_DIR", ".raven")
        mocker.patch.object(server, "RULES_DIR", ".claude/rules")

    def test_no_note_when_nothing_legacy_was_read(self, mocker):
        self._dirs(mocker)
        import raven.server as server
        assert server._legacy_config_path_lines({}) == []
        assert server._legacy_config_path_lines({"legacy_config_paths": []}) == []

    def test_note_names_the_old_and_the_new_path(self, mocker):
        self._dirs(mocker)
        import raven.server as server
        lines = server._legacy_config_path_lines(
            {"legacy_config_paths": ["severities.json"]})
        body = "\n".join(lines)
        assert ".claude/rules/raven/severities.json" in body
        assert ".raven/severities.json" in body

    def test_note_covers_every_legacy_file_read(self, mocker):
        self._dirs(mocker)
        import raven.server as server
        body = "\n".join(server._legacy_config_path_lines(
            {"legacy_config_paths": ["prompts/review.md", "severities.json"]}))
        assert ".raven/prompts/review.md" in body
        assert ".raven/severities.json" in body

    def test_note_suppressed_when_the_legacy_dir_is_disabled(self, mocker):
        """RULES_DIR="" means the legacy path can't have been read, so a
        stale field on a cached review dict must not render a note
        pointing at a path that doesn't resolve."""
        import raven.server as server
        mocker.patch.object(server, "CONFIG_DIR", ".raven")
        mocker.patch.object(server, "RULES_DIR", "")
        assert server._legacy_config_path_lines(
            {"legacy_config_paths": ["severities.json"]}) == []

    def test_markdown_breaking_paths_are_dropped_not_escaped(self, mocker):
        """The note renders paths inside code spans in a PR comment. Only
        Raven's own literals ever reach this field today, but the review
        dict it rides on is model-shaped — a backtick must not be able to
        close the span and start emitting free markdown."""
        self._dirs(mocker)
        import raven.server as server
        body = "\n".join(server._legacy_config_path_lines({
            "legacy_config_paths": ["ev`il.md", "with\nnewline", "severities.json"],
        }))
        assert "ev`il" not in body
        assert "newline" not in body
        assert ".raven/severities.json" in body

    def test_summary_body_carries_the_note(self, mocker):
        self._dirs(mocker)
        import raven.server as server
        body = server._format_comment({
            "severity": "low", "summary": "s", "findings": [],
            "legacy_config_paths": ["prompts/review.md"],
        })
        assert ".raven/prompts/review.md" in body

    def test_inline_body_carries_the_note(self, mocker):
        """RAVEN_REVIEW_OUTPUT=inline never calls _format_comment — the
        nag must not vanish for those operators (same trap the severity
        mismatch line already fell into once)."""
        self._dirs(mocker)
        import raven.server as server
        body = server._format_inline_leftovers(
            [], review={"legacy_config_paths": ["severities.json"]})
        assert ".raven/severities.json" in body

    def test_severity_mismatch_line_names_the_path_actually_read(self, mocker):
        """The mismatch note points the operator at a file to go fix — it
        must name where the scale really came from, not a hardcoded
        location the repo may no longer use."""
        self._dirs(mocker)
        import raven.server as server
        review = {
            "unknown_severities": ["critical"],
            "severity_scale_names": ["blocker", "bug", "nit"],
        }
        assert ".raven/severities.json" in "\n".join(
            server._severity_mismatch_lines(review))

        review["legacy_config_paths"] = ["severities.json"]
        assert ".claude/rules/raven/severities.json" in "\n".join(
            server._severity_mismatch_lines(review))


class TestProcessPrReportsLegacyConfigPaths:
    """Call-site guard: the on_legacy_path callback must actually be wired
    into _process_pr's fetches and reach the posted body. A unit test on
    the helper alone would pass with the callback never threaded."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _provider(self, files):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.fetch_pr_diff.return_value = "diff --git a/f.py b/f.py\n+line\n"
        mc.fetch_file.side_effect = (
            lambda repo, path, ref=None: files.get(path, ""))
        mc.list_directory.return_value = []
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        return mc

    def _submitted_body(self, mc):
        call = mc.submit_review.call_args
        return call.args[2] if len(call.args) >= 3 else call.kwargs["body"]

    def _run(self, mc, mocker):
        mocker.patch.object(_server_mod, "CONFIG_DIR", ".raven")
        mocker.patch.object(_server_mod, "RULES_DIR", ".claude/rules")
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "summary"),
            patch("raven.server.review_diff",
                  return_value={"severity": "low", "summary": "s", "findings": []}),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())

    def test_legacy_scale_read_is_nagged_in_the_review_body(self, mocker):
        mc = self._provider({
            ".claude/rules/raven/severities.json":
                '{"severities": {"nit": 1, "bad": 2}}',
        })
        self._run(mc, mocker)
        assert ".raven/severities.json" in self._submitted_body(mc)

    def test_legacy_review_override_is_nagged_in_the_review_body(self, mocker):
        mc = self._provider({
            ".claude/rules/raven/prompts/review.md": "OVERRIDE",
        })
        self._run(mc, mocker)
        assert ".raven/prompts/review.md" in self._submitted_body(mc)

    def test_no_nag_when_config_lives_at_the_new_path(self, mocker):
        mc = self._provider({
            ".raven/severities.json": '{"severities": {"nit": 1, "bad": 2}}',
        })
        self._run(mc, mocker)
        assert "Deprecated Raven config path" not in self._submitted_body(mc)

    def test_no_nag_when_the_repo_has_no_config_at_all(self, mocker):
        mc = self._provider({})
        self._run(mc, mocker)
        assert "Deprecated Raven config path" not in self._submitted_body(mc)


class TestProcessCommentReportsLegacyConfigPaths:
    """The comment-reply flow is the only path that reads the RESPOND
    override, so it owns the nag for that file — _process_pr never
    fetches it and would never notice."""

    def _payload(self):
        return {
            "repo": "u/r", "pr_number": 1,
            "comment_body": "@raven what about this?", "comment_id": 12,
            "parent_comment_id": None, "_is_mention": True,
        }

    def _provider(self, files):
        mp = MagicMock(spec=GitProvider)
        mp.name = "gitea"
        mp.fetch_pr_diff.return_value = "diff..."
        mp.get_pr_comments.return_value = []
        mp.get_comment_thread.return_value = []
        mp.get_pr_state.return_value = "open"
        mp.get_pr_head_sha.return_value = "abc123"
        mp.get_pr_base_ref.return_value = "main"
        mp.get_authenticated_user.return_value = "raven"
        mp.supports_comment_threads = True
        mp.fetch_file.side_effect = (
            lambda repo, path, ref=None: files.get(path, ""))
        return mp

    def _posted_body(self, mp):
        return mp.post_pr_comment.call_args.args[2]

    def _run(self, mp, mocker):
        mocker.patch.object(_server_mod, "CONFIG_DIR", ".raven")
        mocker.patch.object(_server_mod, "RULES_DIR", ".claude/rules")
        with patch("raven.server.respond_to_comment",
                   return_value={"response": "here you go"}):
            _process_comment(mp, self._payload())

    def test_legacy_respond_override_is_nagged_in_the_reply(self, mocker):
        mp = self._provider({
            ".claude/rules/raven/prompts/respond.md": "OVERRIDE",
        })
        self._run(mp, mocker)
        body = self._posted_body(mp)
        assert "here you go" in body
        assert ".raven/prompts/respond.md" in body

    def test_no_nag_when_the_respond_override_is_at_the_new_path(self, mocker):
        mp = self._provider({".raven/prompts/respond.md": "OVERRIDE"})
        self._run(mp, mocker)
        assert "Deprecated Raven config path" not in self._posted_body(mp)


# ------------------------------------------------------------------ #
#  Comment-thread-context feature: retract + revise + auto-merge      #
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_provider_for_comment_flow():
    """Module-level fixture shared across the comment-flow tests below
    (TestProcessCommentRetraction, TestProcessCommentRevision,
    TestProcessCommentRaceGuard). Sibling test classes can't share
    class-scoped fixtures."""
    mp = MagicMock(spec=GitProvider)
    mp.name = "gitea"
    mp.fetch_pr_diff.return_value = "diff..."
    mp.get_pr_comments.return_value = [
        {"id": 50, "user": {"login": "carol"}, "body": "global note"},
    ]
    mp.get_comment_thread.return_value = [
        {"id": 10, "parent_id": None, "user": {"login": "raven"},
         "body": "Original finding", "file_path": "a.py", "line": 5,
         "resolved": False},
        {"id": 11, "parent_id": 10, "user": {"login": "alice"},
         "body": "Why is this bad?", "file_path": "a.py", "line": 5,
         "resolved": False},
    ]
    mp.get_pr_state.return_value = "open"
    mp.get_pr_head_sha.return_value = "abc123"
    mp.get_pr_metadata.return_value = {"title": "Test PR", "html_url": "https://x/u/r/pulls/1"}
    mp.fetch_file.side_effect = lambda r, p, ref="HEAD": "" if p == "CLAUDE.md" else "code"
    mp.get_pr_base_ref.return_value = "main"
    mp.get_authenticated_user.return_value = "raven"
    mp.supports_comment_threads = True
    return mp


@pytest.fixture
def cached_needs_work(mock_provider_for_comment_flow):
    """Seed cache with a prior 'needs_work' entry under the prefixed key
    format _process_pr uses: f'{provider.name}:{repo}#{pr}'."""
    from raven.server import CacheEntry, _previous_diffs
    pr_key = "gitea:u/r#1"
    _previous_diffs[pr_key] = CacheEntry(
        timestamp=0.0, hashes={}, findings={},
        verdict="needs_work", summary="see findings",
    )
    yield
    _previous_diffs.pop(pr_key, None)


class TestProcessCommentRetraction:
    def _payload(self, comment_id=12, parent=10):
        return {
            "repo": "u/r", "pr_number": 1,
            "comment_body": "?", "comment_id": comment_id,
            "parent_comment_id": parent, "_is_mention": True,
            "file_path": "a.py", "line": 5,
        }

    def test_retraction_preserves_coverage_gap_files_in_cache(self, mock_provider_for_comment_flow):
        """PIN (PR #157 re-review, finding A — code already correct):
        the comment flow never RECONSTRUCTS CacheEntry. Only two
        constructions exist (_load_cache and _process_pr's cache
        write); retractions mutate entry.findings in place and
        revisions assign entry.verdict/entry.summary on the existing
        object, so coverage_gap_files survives comment activity
        untouched. If a comment-flow write ever rebuilt the entry
        without the field, it would silently reset to [] and a SECOND
        comment-driven flip-to-approve would pass both the
        flip-suppression guard and the merge-dispatch gate. Lock the
        invariant: a retraction-only flow on a gap-carrying entry
        filters the retracted finding but leaves the gap list (and
        verdict) intact."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        marker = {"severity": "medium", "file": "b.py",
                  "message": "⚠️ `b.py` skipped (too large: 9001 lines)"}
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0,
            hashes={},
            findings={
                "a.py": [{"severity": "low", "message": "nit", "comment_id": 10}],
                "b.py": [marker],
            },
            verdict="needs_work",
            summary="partial",
            coverage_gap_files=["b.py"],
        )
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "fair point", "revise": None,
                    "retract_findings": [10],
                }
                _process_comment(mock_provider_for_comment_flow, self._payload())
            entry = _previous_diffs[pr_key]
            # The retraction itself happened: provider call made and the
            # comment-linked finding dropped from its bucket …
            mock_provider_for_comment_flow.retract_finding.assert_called_once()
            assert entry.findings["a.py"] == []
            # … the marker finding (no comment_id) survives …
            assert entry.findings["b.py"] == [marker]
            # … and the gap list + verdict are untouched — no silent
            # reset that would unlock a later flip-to-approve.
            assert entry.coverage_gap_files == ["b.py"]
            assert entry.verdict == "needs_work"
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_retracts_filtered_to_raven_authored_thread_ids(self, mock_provider_for_comment_flow):
        """IDs the AI lists are filtered TWO ways:
          - dropped if not in the fetched thread (defense vs hallucination), AND
          - dropped if the thread entry wasn't authored by Raven (defense
            against the AI/prompt-injection resolving a developer's
            comment).
        Mock thread has id=10 (raven) and id=11 (alice). AI returns
        [10, 11, 9999]. After filtering, only Raven's own comment 10
        survives. 10 has no parent so root == 10."""
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [10, 11, 9999],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        called_ids = sorted(
            call.args[2] for call in mock_provider_for_comment_flow.retract_finding.call_args_list
        )
        assert called_ids == [10]

    def test_retract_walks_to_thread_root_from_reply(self, mock_provider_for_comment_flow):
        """When the AI picks a Raven-authored REPLY id (not the original
        finding's id), the server walks up the in-memory thread to the
        root and resolves that. Thread resolution is a thread-root
        operation; the BB DC GET response has no parent field so the
        provider can't walk up via API — the caller does it using the
        thread tree we already fetched.

        Mock thread:
          id=10 (raven, root, no parent)  <-- THE finding
            id=20 (alice, reply)
              id=30 (raven, reply)         <-- AI picks this
        Expected: retract_finding called with id=10, not 30.
        """
        mock_provider_for_comment_flow.get_comment_thread.return_value = [
            {"id": 10, "parent_id": None, "user": {"login": "raven"},
             "body": "Original finding", "resolved": False},
            {"id": 20, "parent_id": 10, "user": {"login": "alice"},
             "body": "Not a bug", "resolved": False},
            {"id": 30, "parent_id": 20, "user": {"login": "raven"},
             "body": "Acknowledged — finding doesn't apply", "resolved": False},
        ]
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [30],  # AI picks its own reply, not the root
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        called_ids = [
            call.args[2] for call in mock_provider_for_comment_flow.retract_finding.call_args_list
        ]
        # Walked up: 30 → 20 → 10. Resolve root.
        assert called_ids == [10]

    def test_retract_dedupes_when_multiple_replies_share_root(self, mock_provider_for_comment_flow):
        """If the AI lists multiple Raven-authored replies in the same
        thread, all walk to the same root — call retract_finding once,
        not N times."""
        mock_provider_for_comment_flow.get_comment_thread.return_value = [
            {"id": 10, "parent_id": None, "user": {"login": "raven"}, "body": "X", "resolved": False},
            {"id": 11, "parent_id": 10, "user": {"login": "raven"}, "body": "Y", "resolved": False},
            {"id": 12, "parent_id": 11, "user": {"login": "raven"}, "body": "Z", "resolved": False},
        ]
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [10, 11, 12],  # all three
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        called_ids = [
            call.args[2] for call in mock_provider_for_comment_flow.retract_finding.call_args_list
        ]
        # Three seeds collapse to a single root.
        assert called_ids == [10]

    def test_retract_drops_when_root_not_raven_authored(self, mock_provider_for_comment_flow):
        """Defense: Raven joined a developer-rooted thread (e.g. answered
        a @mention on a top-level discussion comment). The AI sees its
        own reply marked [YOU], lists it for retract — but the thread
        root is the developer's comment. We never resolve threads we
        didn't originate; drop with a warning."""
        mock_provider_for_comment_flow.get_comment_thread.return_value = [
            {"id": 10, "parent_id": None, "user": {"login": "alice"},
             "body": "Discussion starter", "resolved": False},
            {"id": 20, "parent_id": 10, "user": {"login": "raven"},
             "body": "@mention reply", "resolved": False},
        ]
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [20],  # Raven's own reply
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        # 20 walks to root=10 (alice). Root not Raven-authored → drop.
        mock_provider_for_comment_flow.retract_finding.assert_not_called()

    def test_retracts_skipped_when_pr_not_open(self, mock_provider_for_comment_flow):
        mock_provider_for_comment_flow.get_pr_state.return_value = "merged"
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [10],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        mock_provider_for_comment_flow.retract_finding.assert_not_called()

    def test_retract_failure_does_not_block_subsequent(self, mock_provider_for_comment_flow):
        # Two independent Raven-rooted threads so they don't dedupe to
        # a single root via the in-memory walk-up. (Within one thread,
        # multiple [YOU]-marked entries collapse to the same root.)
        mock_provider_for_comment_flow.get_comment_thread.return_value = [
            {"id": 10, "parent_id": None, "user": {"login": "raven"},
             "body": "Finding A", "file_path": "a.py", "line": 5,
             "resolved": False},
            {"id": 20, "parent_id": None, "user": {"login": "raven"},
             "body": "Finding B", "file_path": "b.py", "line": 5,
             "resolved": False},
        ]
        mock_provider_for_comment_flow.retract_finding.side_effect = [False, True]
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None,
                "retract_findings": [10, 20],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        assert mock_provider_for_comment_flow.retract_finding.call_count == 2

    def test_no_retract_when_list_empty(self, mock_provider_for_comment_flow):
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok", "revise": None, "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        mock_provider_for_comment_flow.retract_finding.assert_not_called()

    def test_successful_retract_drops_matching_finding_from_cache(self, mock_provider_for_comment_flow):
        """End-to-end: when a cached finding carries comment_id=42 and
        retract_finding(42) succeeds, the cache cleanup drops that
        finding so the next push-driven incremental review doesn't
        carry it forward and re-post."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={"a.py": "h"},
            findings={"a.py": [
                {"file": "a.py", "line": 5, "severity": "medium",
                 "message": "the flagged thing", "comment_id": 42},
                {"file": "a.py", "line": 10, "severity": "low",
                 "message": "another finding", "comment_id": 43},
            ]},
            verdict="needs_work", summary="see findings",
        )
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            mock_provider_for_comment_flow.get_comment_thread.return_value = [
                {"id": 42, "parent_id": None, "user": {"login": "raven"},
                 "body": "the flagged thing", "file_path": "a.py", "line": 5,
                 "resolved": False},
            ]
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "retracting",
                    "revise": None,
                    "retract_findings": [42],
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 99,
                    "parent_comment_id": 42, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            remaining = _previous_diffs[pr_key].findings["a.py"]
            assert all(f.get("comment_id") != 42 for f in remaining)
            assert any(f.get("comment_id") == 43 for f in remaining)
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_all_findings_retracted_synthesizes_revise_to_approve(self, mock_provider_for_comment_flow):
        """Defense in depth: when the AI retracts every cached finding
        but doesn't set `revise`, and prior verdict was `needs_work`,
        the server synthesizes a flip to `approve`. Without this
        backstop, a conservative AI's "I acknowledge" response leaves
        the PR blocked despite the basis for blocking being gone."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={"a.py": "h"},
            findings={"a.py": [
                {"file": "a.py", "line": 5, "severity": "high",
                 "message": "the only finding", "comment_id": 42},
            ]},
            verdict="needs_work", summary="single concern",
        )
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            mock_provider_for_comment_flow.get_comment_thread.return_value = [
                {"id": 42, "parent_id": None, "user": {"login": "raven"},
                 "body": "the only finding", "file_path": "a.py", "line": 5,
                 "resolved": False},
            ]
            mock_provider_for_comment_flow.submit_review.return_value = {"id": 1234}
            with patch("raven.server.respond_to_comment") as mock_respond, \
                 patch("raven.server._safe_do_merge"):
                mock_respond.return_value = {
                    "response": "you're right, retracting",
                    "revise": None,                # AI did NOT set revise
                    "retract_findings": [42],      # but retracted the only finding
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 99,
                    "parent_comment_id": 42, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            # The backstop fired: a new formal review was submitted with
            # approve=True and the synthesized body.
            mock_provider_for_comment_flow.submit_review.assert_called_once()
            kwargs = mock_provider_for_comment_flow.submit_review.call_args.kwargs
            assert kwargs["approve"] is True
            assert "Revised to approve" in kwargs["body"]
            # Cache verdict flipped accordingly.
            assert _previous_diffs[pr_key].verdict == "approve"
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_partial_retract_does_not_synthesize_revise(self, mock_provider_for_comment_flow):
        """Backstop fires only when the cache is empty after retract.
        Retracting 1 of 2 findings leaves the verdict unchanged — the
        remaining finding still justifies `needs_work`."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={"a.py": "h"},
            findings={"a.py": [
                {"file": "a.py", "line": 5, "severity": "high",
                 "message": "retract me", "comment_id": 42},
                {"file": "a.py", "line": 9, "severity": "high",
                 "message": "still valid", "comment_id": 43},
            ]},
            verdict="needs_work", summary="two concerns",
        )
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            mock_provider_for_comment_flow.get_comment_thread.return_value = [
                {"id": 42, "parent_id": None, "user": {"login": "raven"},
                 "body": "retract me", "file_path": "a.py", "line": 5,
                 "resolved": False},
            ]
            with patch("raven.server.respond_to_comment") as mock_respond, \
                 patch("raven.server._safe_do_merge"):
                mock_respond.return_value = {
                    "response": "ack",
                    "revise": None,
                    "retract_findings": [42],
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 99,
                    "parent_comment_id": 42, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            # No new formal review submitted — cache still has a finding.
            mock_provider_for_comment_flow.submit_review.assert_not_called()
            assert _previous_diffs[pr_key].verdict == "needs_work"
        finally:
            _previous_diffs.pop(pr_key, None)


class TestProcessCommentRevision:
    def _payload(self):
        return {
            "repo": "u/r", "pr_number": 1,
            "comment_body": "?", "comment_id": 12,
            "parent_comment_id": None, "_is_mention": True,
        }

    def test_revise_needs_work_to_approve_submits_review(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        # Patch _safe_do_merge so the inline-executor autouse fixture
        # doesn't run real CI-wait polling (CI_WAIT_TIMEOUT defaults
        # to 300s and time.sleep is real inside _wait_for_ci).
        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge"):
            mock_respond.return_value = {
                "response": "you're right",
                "revise": {"verdict": "approve", "body": "Revised: LGTM"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        kwargs = mock_provider_for_comment_flow.submit_review.call_args.kwargs
        assert kwargs["approve"] is True
        assert kwargs["body"] == "Revised: LGTM"

    def test_revise_in_advisory_mode_uses_comment_only_and_advisory_body(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """In advisory mode, verdict revision posts via
        submit_review(comment_only=True) with the advisory_update body
        header, and auto-merge dispatch is suppressed."""
        monkeypatch.setattr("raven.server.RAVEN_REVIEW_MODE", "advisory")
        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge") as mock_merge:
            mock_respond.return_value = {
                "response": "ack",
                "revise": {"verdict": "approve", "body": "Revised: LGTM"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())

        kwargs = mock_provider_for_comment_flow.submit_review.call_args.kwargs
        # comment_only path
        assert kwargs.get("comment_only") is True
        # Body is wrapped via _format_comment(mode="advisory_update").
        assert "Raven Updated Recommendation" in kwargs["body"]
        # Even on a flip-to-approve, auto-merge is suppressed in advisory.
        mock_merge.assert_not_called()

    def test_revise_unchanged_verdict_skips_submit(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "x",
                "revise": {"verdict": "needs_work", "body": "still needs work"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        mock_provider_for_comment_flow.submit_review.assert_not_called()

    def test_cache_updated_after_successful_submit(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        from raven.server import _previous_diffs
        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge"):
            mock_respond.return_value = {
                "response": "yes",
                "revise": {"verdict": "approve", "body": "Revised"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        entry = _previous_diffs["gitea:u/r#1"]
        assert entry.verdict == "approve"
        assert entry.summary == "Revised"

    def test_submit_failure_leaves_cache_unchanged(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        from raven.server import _previous_diffs
        mock_provider_for_comment_flow.submit_review.side_effect = RuntimeError("api down")
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "x",
                "revise": {"verdict": "approve", "body": "Revised"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        entry = _previous_diffs["gitea:u/r#1"]
        assert entry.verdict == "needs_work"

    def test_auto_merge_dispatched_on_flip_to_approve(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        from concurrent.futures import Future
        submitted = []

        class _Capture:
            def submit(self, fn, *args, **kwargs):
                submitted.append((fn, args, kwargs))
                fut = Future(); fut.set_result(None); return fut
            def shutdown(self, **kwargs): pass

        monkeypatch.setattr("raven.server.ci_wait_executor", _Capture())
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "ok",
                "revise": {"verdict": "approve", "body": "Revised"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, self._payload())
        assert submitted, "Expected ci_wait_executor.submit on flip-to-approve"

    def test_no_auto_merge_on_flip_to_needs_work(self, mock_provider_for_comment_flow, monkeypatch):
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={}, findings={},
            verdict="approve", summary="LGTM",
        )
        try:
            submitted = []
            from concurrent.futures import Future

            class _Capture:
                def submit(self, fn, *args, **kwargs):
                    submitted.append((fn, args, kwargs))
                    fut = Future(); fut.set_result(None); return fut
                def shutdown(self, **kwargs): pass

            monkeypatch.setattr("raven.server.ci_wait_executor", _Capture())
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "wait",
                    "revise": {"verdict": "needs_work", "body": "Found another issue"},
                    "retract_findings": [],
                }
                _process_comment(mock_provider_for_comment_flow, self._payload())
            assert not submitted
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_auto_merge_dispatched_on_retract_only_when_prior_approve(
        self, mock_provider_for_comment_flow, monkeypatch,
    ):
        """Goal 3 regression guard: BB DC scenario where prior verdict
        was 'approve' but auto-merge was blocked by unresolved comments.
        Retraction succeeds → auto-merge MUST retry."""
        from raven.server import CacheEntry, _previous_diffs
        from concurrent.futures import Future
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={}, findings={},
            verdict="approve", summary="LGTM",
        )
        submitted = []

        class _Capture:
            def submit(self, fn, *args, **kwargs):
                submitted.append((fn, args, kwargs))
                fut = Future(); fut.set_result(None); return fut
            def shutdown(self, **kwargs): pass

        monkeypatch.setattr("raven.server.ci_wait_executor", _Capture())
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "you're right",
                    "revise": None,
                    "retract_findings": [10],
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 99,
                    "parent_comment_id": 10, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            mock_provider_for_comment_flow.retract_finding.assert_called_once()
            mock_provider_for_comment_flow.submit_review.assert_not_called()
            assert submitted, "Expected auto-merge dispatch on retraction-only with prior=approve"
        finally:
            _previous_diffs.pop(pr_key, None)

    # ---------------------------------------------------------------- #
    #  Sole-reviewer gate on the comment flow. The push path only       #
    #  auto-merges when Raven is the sole reviewer; a comment-driven    #
    #  verdict flip must respect the same gate — otherwise a '@raven'   #
    #  reply can merge a PR a human reviewer was still blocking.        #
    # ---------------------------------------------------------------- #

    def _capture_executor(self, monkeypatch):
        from concurrent.futures import Future
        submitted = []

        class _Capture:
            def submit(self, fn, *args, **kwargs):
                submitted.append((fn, args, kwargs))
                fut = Future(); fut.set_result(None); return fut
            def shutdown(self, **kwargs): pass

        monkeypatch.setattr("raven.server.ci_wait_executor", _Capture())
        return submitted

    def _flip_to_approve(self, mock_respond):
        mock_respond.return_value = {
            "response": "ok",
            "revise": {"verdict": "approve", "body": "Revised"},
            "retract_findings": [],
        }

    def test_no_auto_merge_dispatch_when_other_reviewer_exists(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Flip-to-approve with a non-Raven review present must NOT
        dispatch auto-merge — same gate as _process_pr."""
        submitted = self._capture_executor(monkeypatch)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = [
            {"user": {"login": "alice"}, "state": "COMMENTED"},
        ]
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = []
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())
        # The verdict revision itself still posts — only the merge is gated.
        mock_provider_for_comment_flow.submit_review.assert_called_once()
        assert not submitted, "Comment-flow auto-merge must respect the sole-reviewer gate"

    def test_no_auto_merge_dispatch_when_other_reviewer_requested(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        submitted = self._capture_executor(monkeypatch)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = []
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = ["bob"]
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())
        assert not submitted, "Pending requested reviewer must block comment-flow auto-merge"

    def test_auto_merge_dispatched_when_raven_is_sole_reviewer(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Raven's own review and self-request must not count as 'other
        reviewers' (case-insensitive), mirroring the push-path filter."""
        submitted = self._capture_executor(monkeypatch)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"},
        ]
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = ["raven"]
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())
        assert submitted, "Raven-only reviewer state must still dispatch auto-merge"

    def test_no_auto_merge_dispatch_when_reviewer_check_fails(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Fail closed: if reviewer state can't be verified, don't merge."""
        submitted = self._capture_executor(monkeypatch)
        mock_provider_for_comment_flow.get_pr_reviews.side_effect = RuntimeError("api down")
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())
        assert not submitted, "Reviewer-state fetch failure must fail closed (no merge dispatch)"

    def test_flip_to_approve_suppressed_when_cached_coverage_gap(
        self, mock_provider_for_comment_flow, monkeypatch,
    ):
        """Coverage-gap gate on the comment flow: the prior review had
        unreviewed files, and the respond model never saw them either —
        a comment-induced flip-to-approve (e.g. the author talking the
        AI into retracting the skip marker) must NOT post a formal
        APPROVE, must NOT flip the cached verdict, and must NOT dispatch
        auto-merge. The conversational reply itself still posts."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={}, findings={},
            verdict="needs_work", summary="partial review",
            coverage_gap_files=["big.py"],
        )
        submitted = self._capture_executor(monkeypatch)
        try:
            with patch("raven.server.respond_to_comment") as mock_respond:
                self._flip_to_approve(mock_respond)
                _process_comment(mock_provider_for_comment_flow, self._payload())
            # The reply text posted …
            assert mock_provider_for_comment_flow.post_pr_comment.called
            # … but no formal review (the APPROVE was suppressed) …
            mock_provider_for_comment_flow.submit_review.assert_not_called()
            # … the cached verdict stays needs_work …
            assert _previous_diffs[pr_key].verdict == "needs_work"
            # … and nothing was dispatched to merge.
            assert not submitted, "Cached coverage gap must block comment-driven auto-merge"
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_flip_to_approve_suppressed_when_cache_entry_missing(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Fail-closed parity (PR #157 re-review, low): the merge-
        dispatch gate treats a missing cache entry as unverifiable gap
        state and blocks, but the flip-to-approve guard used to fail
        OPEN on the same state (gap_files=[] → formal APPROVE posts).
        The eviction window is real: several provider HTTP round-trips
        sit between the TOCTOU verdict re-check and the guard, during
        which a concurrent _process_pr's _evict_cache() can LRU-evict
        the entry. Simulate it by evicting during get_pr_state (which
        runs after the TOCTOU check): the formal APPROVE must be
        suppressed, mirroring the merge gate."""
        from raven.server import _previous_diffs
        pr_key = "gitea:u/r#1"
        submitted = self._capture_executor(monkeypatch)

        def _evict_then_open(repo, pr):
            _previous_diffs.pop(pr_key, None)
            return "open"

        mock_provider_for_comment_flow.get_pr_state.side_effect = _evict_then_open
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())
        # The conversational reply still posts …
        assert mock_provider_for_comment_flow.post_pr_comment.called
        # … but no formal review (fail-closed suppression) …
        mock_provider_for_comment_flow.submit_review.assert_not_called()
        # … and nothing was dispatched to merge.
        assert not submitted

    def test_retract_only_dispatch_blocked_by_other_reviewer(
        self, mock_provider_for_comment_flow, monkeypatch,
    ):
        """The retraction-on-prior-approve dispatch path (BB DC unblock)
        must respect the same gate as the flip-to-approve path."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={}, findings={},
            verdict="approve", summary="LGTM",
        )
        submitted = self._capture_executor(monkeypatch)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = [
            {"user": {"login": "alice"}, "state": "CHANGES_REQUESTED"},
        ]
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = []
        try:
            mock_provider_for_comment_flow.retract_finding.return_value = True
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "you're right",
                    "revise": None,
                    "retract_findings": [10],
                }
                # In-thread reply payload (parent set) — same shape as
                # test_auto_merge_dispatched_on_retract_only_when_prior_approve,
                # which proves this payload reaches the dispatch site.
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 99,
                    "parent_comment_id": 10, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            assert not submitted, "Retract-only dispatch must respect the sole-reviewer gate"
        finally:
            _previous_diffs.pop(pr_key, None)

    def test_flip_to_approve_suppressed_when_severity_scale_fetch_fails(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Fail-closed parity with _process_pr (audit 2026-08-14 MED).

        When severities.json cannot be READ, _fetch_severity_scale falls
        back to default_scale() — which for a repo that reuses the
        built-in low/medium/high names but sets a tighter
        blocks_at_or_above is a LOOSER gate than its real policy.
        _process_pr already refuses to approve on that state via its
        on_fetch_failed closure; the comment-driven flip-to-approve is
        the one remaining merge-capable path that did not, so a
        transient provider error could auto-merge past the repo's own
        blocking tier. The reply itself still posts — only the merge is
        blocked."""
        submitted = self._capture_executor(monkeypatch)

        def _fail_severities(repo, path, ref=None, *a, **kw):
            if path.endswith("severities.json"):
                raise RuntimeError("transient provider failure")
            return ""

        mock_provider_for_comment_flow.fetch_file.side_effect = _fail_severities
        with patch("raven.server.respond_to_comment") as mock_respond:
            self._flip_to_approve(mock_respond)
            _process_comment(mock_provider_for_comment_flow, self._payload())

        assert mock_provider_for_comment_flow.post_pr_comment.called, (
            "The conversational reply must still post — only the merge is gated"
        )
        assert not submitted, (
            "severities.json unreadable means the repo's real merge gate is "
            "unknown; a comment-driven flip-to-approve must not auto-merge"
        )


class TestProcessCommentRaceGuard:
    def test_comment_flow_does_not_add_itself_to_in_progress(
        self, mock_provider_for_comment_flow, cached_needs_work, monkeypatch,
    ):
        """Asymmetric semantics: comment-flow gives push priority by
        checking _in_progress_prs, but must NOT add itself — otherwise
        a fresh push webhook arriving during the comment-flow's
        ~30-60s synchronous sequence would be dropped at server.py:553
        and the new commits never get reviewed.

        It DOES add itself to _comment_mutating_prs so a concurrent
        second comment-flow on the same PR serializes.
        """
        from raven.server import (
            _in_progress_prs, _comment_mutating_prs, _in_progress_lock,
        )
        pr_key = "gitea:u/r#1"

        # Capture set membership at submit_review call time.
        captured = {}
        def _capture_then_return(*args, **kwargs):
            with _in_progress_lock:
                captured["in_progress_prs"] = pr_key in _in_progress_prs
                captured["comment_mutating"] = pr_key in _comment_mutating_prs
            return {"id": 1, "inline_comments": []}
        mock_provider_for_comment_flow.submit_review.side_effect = _capture_then_return
        # Also patch _safe_do_merge so the inline-executor autouse fixture
        # doesn't run real CI polling.
        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge"):
            mock_respond.return_value = {
                "response": "ok",
                "revise": {"verdict": "approve", "body": "Revised"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })
        # Mid-flight: pr_key was NOT in _in_progress_prs (push set),
        # but IS in _comment_mutating_prs (concurrent-comment exclusion).
        assert captured["in_progress_prs"] is False
        assert captured["comment_mutating"] is True
        # And after: comment-mutation slot released.
        with _in_progress_lock:
            assert pr_key not in _comment_mutating_prs
            assert pr_key not in _in_progress_prs

    def test_in_progress_skips_revision_and_retraction(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        """If the PR is already in _in_progress_prs (push re-review),
        skip mutations but still post the reply."""
        from raven.server import _in_progress_prs, _in_progress_lock
        pr_key = "gitea:u/r#1"
        with _in_progress_lock:
            _in_progress_prs.add(pr_key)
        try:
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "ok",
                    "revise": {"verdict": "approve", "body": "Revised"},
                    "retract_findings": [10],
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 12,
                    "parent_comment_id": 10, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            mock_provider_for_comment_flow.submit_review.assert_not_called()
            mock_provider_for_comment_flow.retract_finding.assert_not_called()
            mock_provider_for_comment_flow.post_pr_comment.assert_called()  # reply still went out
        finally:
            with _in_progress_lock:
                _in_progress_prs.discard(pr_key)

    def test_concurrent_comment_flow_skips_mutations(
        self, mock_provider_for_comment_flow, cached_needs_work,
    ):
        """When another comment-flow is mid-mutation for the same PR
        (entry in _comment_mutating_prs), the second one bails before
        submit_review — protects against both flows submitting opposing
        reviews + dismissing each other's. Reply still posts."""
        from raven.server import _comment_mutating_prs, _in_progress_lock
        pr_key = "gitea:u/r#1"
        with _in_progress_lock:
            _comment_mutating_prs.add(pr_key)
        try:
            with patch("raven.server.respond_to_comment") as mock_respond:
                mock_respond.return_value = {
                    "response": "ok",
                    "revise": {"verdict": "approve", "body": "Revised"},
                    "retract_findings": [10],
                }
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 12,
                    "parent_comment_id": 10, "_is_mention": True,
                    "file_path": "a.py", "line": 5,
                })
            mock_provider_for_comment_flow.submit_review.assert_not_called()
            mock_provider_for_comment_flow.retract_finding.assert_not_called()
            mock_provider_for_comment_flow.post_pr_comment.assert_called()  # reply still went out
        finally:
            with _in_progress_lock:
                _comment_mutating_prs.discard(pr_key)

    def test_verdict_none_skips_revise_server_side(self, mock_provider_for_comment_flow):
        """Server enforces 'no revise without prior verdict' regardless
        of AI behaviour (defense in depth)."""
        # No cache entry → prior_verdict is None
        with patch("raven.server.respond_to_comment") as mock_respond:
            mock_respond.return_value = {
                "response": "...",
                "revise": {"verdict": "approve", "body": "AI ignored the rule"},
                "retract_findings": [],
            }
            _process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })
        mock_provider_for_comment_flow.submit_review.assert_not_called()
        mock_provider_for_comment_flow.post_pr_comment.assert_called()

    def test_prior_verdict_changed_under_guard_skips_mutations(
        self, mock_provider_for_comment_flow,
    ):
        """If a concurrent _process_pr changed the cache verdict between
        the AI call and the under-guard re-check, mutations are skipped."""
        from raven.server import CacheEntry, _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key] = CacheEntry(
            timestamp=0.0, hashes={}, findings={},
            verdict="approve", summary="LGTM",
        )

        def _flip_cache_during_ai(*args, **kwargs):
            _previous_diffs[pr_key].verdict = "needs_work"
            _previous_diffs[pr_key].summary = "Push found issues"
            return {
                "response": "ok",
                "revise": {"verdict": "needs_work", "body": "Reconsidered"},
                "retract_findings": [],
            }

        try:
            with patch("raven.server.respond_to_comment",
                       side_effect=_flip_cache_during_ai):
                _process_comment(mock_provider_for_comment_flow, {
                    "repo": "u/r", "pr_number": 1,
                    "comment_body": "?", "comment_id": 12,
                    "parent_comment_id": None, "_is_mention": True,
                })
            mock_provider_for_comment_flow.post_pr_comment.assert_called()  # reply still out
            mock_provider_for_comment_flow.submit_review.assert_not_called()
        finally:
            _previous_diffs.pop(pr_key, None)


class TestCommentFlowUsesTheRepoScale:
    """Task 14: the comment-reply flow is the last path that rendered on
    the built-in low/medium/high vocabulary regardless of the repo's own
    severities.json — the same defect class Task 12b fixed for
    _process_pr's renderers (nine total across the feature; none caught
    by a green suite)."""

    def test_max_severity_uses_the_supplied_scale(self):
        import raven.server as server
        from raven.severity import SeverityScale
        s = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                          blocks_at_or_above="bug")
        findings = [{"severity": "nit"}, {"severity": "bug"}]
        assert server._max_severity_from_findings(findings, s) == "bug"

    def test_custom_tier_not_rendered_as_default_vocabulary(self):
        """Without the scale, a repo's tier name that isn't in the
        built-in vocabulary is unranked against it, and
        _max_severity_from_findings (deliberately, matching the
        pre-scale SEVERITY_ORDER.get(name, 0) behaviour preserved
        through Phase A — see its own docstring) ties an unranked name
        with the scale's LEAST severe tier, not the most severe one.
        So a repo's 'blocker' — its own most severe tier — silently
        renders and would notify as 'low' if the scale were never
        threaded through: an under-representation, the opposite of
        fail-closed, and exactly the silent-wrong-answer failure this
        task exists to prevent.

        (Note: the task brief's own draft of this test asserted
        '== "high"' for the unscoped call — verified empirically wrong
        against the shipped _max_severity_from_findings, whose docstring
        and Phase-A history both establish "unrecognised -> least
        severe" as the deliberate, preserved behaviour. Corrected here;
        see task-14-report.md.)
        """
        import raven.server as server
        from raven.severity import SeverityScale
        s = SeverityScale(ranks={"nit": 10, "blocker": 30},
                          blocks_at_or_above="blocker")
        assert server._max_severity_from_findings([{"severity": "blocker"}], s) == "blocker"
        assert server._max_severity_from_findings([{"severity": "blocker"}]) == "low"

    def test_process_comment_fetches_the_scale_from_base_ref(
            self, mock_provider_for_comment_flow, mocker):
        """_process_comment must resolve the repo scale from the PR's base
        ref, like it already does for CLAUDE.md — not from the head SHA."""
        import raven.server as server
        from raven.severity import default_scale

        spy = mocker.patch.object(server, "_fetch_severity_scale",
                                  return_value=default_scale())
        mock_provider_for_comment_flow.get_pr_base_ref.return_value = "base-sha"

        with patch("raven.server.respond_to_comment",
                   return_value={"response": "ok", "revise": None, "retract_findings": []}):
            server._process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })

        assert spy.called
        assert spy.call_args[0][2] == "base-sha"

    def test_advisory_update_body_uses_the_fetched_scale_not_default(
            self, mock_provider_for_comment_flow, cached_needs_work,
            mocker, monkeypatch):
        """Integration-level pin (mirrors Task 12b's approach — a
        renderer-only unit test can't catch a call-site regression):
        the advisory_update body posted by _process_comment must render
        with the REPO'S scale in BOTH places _format_comment needs it —
        the computed severity string AND the ``scale=`` kwarg that
        drives its emoji.

        Deliberately uses the MIDDLE tier ('bug'), not the most-severe
        one: a most-severe example is a weak mutation target here,
        because ``SeverityScale.emoji()`` fails closed to most-severe
        on an unrecognised name, so an unthreaded ``scale=`` kwarg would
        *coincidentally* still render red for a most-severe finding —
        this test would have passed even with that call site's
        ``scale=comment_scale`` reverted (caught only by mutation
        testing, see task-14-report.md). 'bug' can't coincide: under
        the real scale it's neither most nor least severe (🟠); if
        _max_severity_from_findings is unthreaded, 'bug' is unranked
        against default_scale() and collapses to its least-severe tier
        ('low'); if _format_comment's scale= is unthreaded, 'bug'
        stays the right STRING but its emoji fails closed to
        default_scale()'s most-severe (🔴, since 'bug' is unranked
        there too). Either mutation alone is now visibly wrong."""
        import raven.server as server
        from raven.severity import SeverityScale

        scale = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                              blocks_at_or_above="bug")
        mocker.patch.object(server, "_fetch_severity_scale", return_value=scale)
        monkeypatch.setattr("raven.server.RAVEN_REVIEW_MODE", "advisory")

        from raven.server import _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key].findings = {
            "a.py": [{"severity": "bug", "message": "m"}],
        }

        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge"):
            mock_respond.return_value = {
                "response": "ack",
                "revise": {"verdict": "approve", "body": "Revised: LGTM"},
                "retract_findings": [],
            }
            server._process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })

        kwargs = mock_provider_for_comment_flow.submit_review.call_args.kwargs
        assert "🟠" in kwargs["body"]
        assert "BUG" in kwargs["body"]
        assert "🔴" not in kwargs["body"]

    def test_synthetic_merge_review_uses_the_fetched_scale_not_default(
            self, mock_provider_for_comment_flow, cached_needs_work, mocker):
        """Integration-level pin for the OTHER _max_severity_from_findings
        call site (~:2777): the synthetic review dict built for the
        comment-driven auto-merge dispatch must compute 'severity' from
        the repo's own scale too, not default_scale()."""
        import raven.server as server
        from raven.severity import SeverityScale

        scale = SeverityScale(ranks={"nit": 10, "blocker": 30},
                              blocks_at_or_above="blocker")
        mocker.patch.object(server, "_fetch_severity_scale", return_value=scale)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = []
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = []

        from raven.server import _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key].findings = {
            "a.py": [{"severity": "blocker", "message": "m"}],
        }

        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge") as mock_merge:
            mock_respond.return_value = {
                "response": "ok",
                "revise": {"verdict": "approve", "body": "Revised: LGTM"},
                "retract_findings": [],
            }
            server._process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })

        mock_merge.assert_called_once()
        synthetic_review = mock_merge.call_args.args[5]
        assert synthetic_review["severity"] == "blocker"

    def test_synthetic_merge_review_carries_scale_fields(
            self, mock_provider_for_comment_flow, cached_needs_work, mocker):
        """Finding 1 (PR #216 review, 2nd pass): the synthetic review dict
        built for comment-driven auto-merge dispatch must carry
        severity_scale_names / severity_blocks_at — every REAL review dict
        (review_diff's output) carries them, and notifier._scale_from_review
        needs them to reconstruct the repo's actual scale. Without these
        two fields the notifier silently reconstructs default_scale(), so
        a custom-scale repo's merge_failed/ci_failed notification would
        mis-colour (fail closed to most-severe or wrong emoji) and
        _passes_threshold would mis-filter against built-in ranks."""
        import raven.server as server
        from raven.severity import SeverityScale

        scale = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                              blocks_at_or_above="bug")
        mocker.patch.object(server, "_fetch_severity_scale", return_value=scale)
        mock_provider_for_comment_flow.get_pr_reviews.return_value = []
        mock_provider_for_comment_flow.get_pr_requested_reviewers.return_value = []

        from raven.server import _previous_diffs
        pr_key = "gitea:u/r#1"
        _previous_diffs[pr_key].findings = {
            "a.py": [{"severity": "bug", "message": "m"}],
        }

        with patch("raven.server.respond_to_comment") as mock_respond, \
             patch("raven.server._safe_do_merge") as mock_merge:
            mock_respond.return_value = {
                "response": "ok",
                "revise": {"verdict": "approve", "body": "Revised: LGTM"},
                "retract_findings": [],
            }
            server._process_comment(mock_provider_for_comment_flow, {
                "repo": "u/r", "pr_number": 1,
                "comment_body": "?", "comment_id": 12,
                "parent_comment_id": None, "_is_mention": True,
            })

        mock_merge.assert_called_once()
        synthetic_review = mock_merge.call_args.args[5]
        assert synthetic_review["severity_scale_names"] == ["blocker", "bug", "nit"]
        assert synthetic_review["severity_blocks_at"] == "bug"


# ------------------------------------------------------------------ #
#  RAVEN_REVIEW_MODE resolver                                          #
# ------------------------------------------------------------------ #

class TestReviewMode:
    """RAVEN_REVIEW_MODE resolution: explicit flag, defaults, validation.

    Tests the resolver function directly via ``_resolve_review_mode()``
    rather than reloading the module. The resolver reads ``os.environ``
    at call time, so ``monkeypatch.setenv`` is sufficient — no reload,
    no ThreadPoolExecutor leaks, no stale dict references for sibling
    test classes.
    """

    def test_default_mode_is_all(self, monkeypatch):
        from raven.server import _resolve_review_mode
        monkeypatch.delenv("RAVEN_REVIEW_MODE", raising=False)
        assert _resolve_review_mode() == "all"

    def test_explicit_mode_advisory(self, monkeypatch):
        from raven.server import _resolve_review_mode
        monkeypatch.setenv("RAVEN_REVIEW_MODE", "advisory")
        assert _resolve_review_mode() == "advisory"

    def test_explicit_mode_gap(self, monkeypatch):
        from raven.server import _resolve_review_mode
        monkeypatch.setenv("RAVEN_REVIEW_MODE", "gap")
        assert _resolve_review_mode() == "gap"

    def test_invalid_mode_raises_systemexit(self, monkeypatch):
        from raven.server import _resolve_review_mode
        monkeypatch.setenv("RAVEN_REVIEW_MODE", "bogus")
        with pytest.raises(SystemExit):
            _resolve_review_mode()

    def test_legacy_env_var_is_ignored(self, monkeypatch):
        """RAVEN_REVIEW_ALL_PRS was removed entirely — setting it has no
        effect. Clean break, not a soft migration."""
        from raven.server import _resolve_review_mode
        monkeypatch.delenv("RAVEN_REVIEW_MODE", raising=False)
        monkeypatch.setenv("RAVEN_REVIEW_ALL_PRS", "false")  # would have meant 'gap'
        # Default still 'all' because RAVEN_REVIEW_ALL_PRS is no longer read.
        assert _resolve_review_mode() == "all"

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        """docker-compose `${RAVEN_REVIEW_MODE:-}` passes "" into the
        container when the host var is unset. Must not crash the
        validator — empty string is treated as unset and defaults to
        'all'."""
        from raven.server import _resolve_review_mode
        monkeypatch.setenv("RAVEN_REVIEW_MODE", "")
        assert _resolve_review_mode() == "all"

    def test_whitespace_only_env_var_falls_back_to_default(self, monkeypatch):
        """``   `` is functionally unset (same as empty); resolver strips
        before validating."""
        from raven.server import _resolve_review_mode
        monkeypatch.setenv("RAVEN_REVIEW_MODE", "   ")
        assert _resolve_review_mode() == "all"


class TestReviewOutput:
    """RAVEN_REVIEW_OUTPUT resolution: default, explicit values, validation.
    Resolver reads os.environ at call time, so setenv is sufficient."""

    def test_default_output_is_both(self, monkeypatch):
        from raven.server import _resolve_review_output
        monkeypatch.delenv("RAVEN_REVIEW_OUTPUT", raising=False)
        assert _resolve_review_output() == "both"

    def test_explicit_summary(self, monkeypatch):
        from raven.server import _resolve_review_output
        monkeypatch.setenv("RAVEN_REVIEW_OUTPUT", "summary")
        assert _resolve_review_output() == "summary"

    def test_explicit_inline(self, monkeypatch):
        from raven.server import _resolve_review_output
        monkeypatch.setenv("RAVEN_REVIEW_OUTPUT", "inline")
        assert _resolve_review_output() == "inline"

    def test_case_insensitive(self, monkeypatch):
        from raven.server import _resolve_review_output
        monkeypatch.setenv("RAVEN_REVIEW_OUTPUT", "  Summary  ")
        assert _resolve_review_output() == "summary"

    def test_invalid_output_raises_systemexit(self, monkeypatch):
        from raven.server import _resolve_review_output
        monkeypatch.setenv("RAVEN_REVIEW_OUTPUT", "bogus")
        with pytest.raises(SystemExit):
            _resolve_review_output()

    def test_empty_env_var_falls_back_to_both(self, monkeypatch):
        """docker-compose `${RAVEN_REVIEW_OUTPUT:-}` passes "" when unset —
        must default to 'both', not crash the validator."""
        from raven.server import _resolve_review_output
        monkeypatch.setenv("RAVEN_REVIEW_OUTPUT", "")
        assert _resolve_review_output() == "both"


class TestReviewOutputChannels:
    """End-to-end: RAVEN_REVIEW_OUTPUT controls whether the summary body
    and/or inline comments reach submit_review."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.fetch_pr_diff.return_value = "diff --git a/f.py b/f.py\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        return mc

    _REVIEW = {
        "severity": "medium",
        "summary": "two issues",
        "findings": [
            {"severity": "high", "file": "f.py", "line": 3, "message": "inline-able"},
            {"severity": "low", "message": "no file/line — body-only"},
        ],
    }

    def _run(self, output_mode):
        mc = self._make_provider()
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", output_mode),
            patch("raven.server.review_diff", return_value=dict(self._REVIEW)),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        # submit_review(repo, pr, body, approve=, inline_comments=, commit_id=, ...)
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        inline = call.kwargs["inline_comments"]
        return body, inline

    def test_both_posts_body_and_inline(self):
        body, inline = self._run("both")
        assert "Findings:" in body
        assert "inline-able" in body          # findings list present in body
        assert len(inline) == 1               # the file/line finding posted inline
        assert inline[0]["file"] == "f.py"

    def test_summary_posts_body_no_inline(self):
        body, inline = self._run("summary")
        assert "Findings:" in body
        assert "inline-able" in body
        assert inline == []                   # no inline comments

    def test_inline_posts_inline_and_bodyless_findings_only(self):
        body, inline = self._run("inline")
        # inline-able finding goes to the line, NOT the body
        assert len(inline) == 1
        assert inline[0]["file"] == "f.py"
        assert "inline-able" not in body
        # the file-less finding has nowhere inline to go → kept in a MINIMAL
        # body so it isn't dropped...
        assert "no file/line — body-only" in body
        # ...but with NO recommendation: no review-summary prose, no footer.
        assert "two issues" not in body
        assert "Reviewed by Raven" not in body

    def test_inline_all_anchored_posts_no_body(self):
        """Every finding has a line → all post inline, body is empty: inline
        mode shows no summary/recommendation comment at all."""
        mc = self._make_provider()
        review = {
            "severity": "high", "summary": "one issue",
            "findings": [{"severity": "high", "file": "f.py", "line": 3,
                          "message": "boom"}],
        }
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        assert call.kwargs["inline_comments"][0]["file"] == "f.py"
        assert body == ""

    def test_inline_clean_pr_posts_no_body(self):
        """No findings → inline mode posts NO recommendation body (empty
        string); the formal verdict is still recorded via submit_review."""
        mc = self._make_provider()
        clean = {"severity": "low", "summary": "all clear", "findings": []}
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=clean),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        assert body == ""
        assert call.kwargs["inline_comments"] == []

    def test_inline_blocking_no_findings_posts_minimal_body(self):
        """needs_work verdict but the model returned no findings → a one-line
        body is still posted so the blocking review isn't content-less (Gitea
        rejects an empty body with no inline comments for a non-approve)."""
        mc = self._make_provider()
        review = {"severity": "high", "summary": "blocked", "findings": []}
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        assert body != ""
        assert "blocked" in body
        assert call.kwargs["approve"] is False

    def test_inline_surfaces_severity_mismatch_note(self):
        """Finding 2 (PR #216 review): under RAVEN_REVIEW_OUTPUT=inline,
        _process_pr builds the body via _format_inline_leftovers and never
        calls _format_comment — so the config-error line naming the
        unrecognised severities (the feature's whole detection story) was
        invisible to the operator, even though the merge still failed
        closed. Must surface the same note the summary/both path gets from
        _format_comment."""
        mc = self._make_provider()
        review = dict(self._REVIEW)
        review["unknown_severities"] = ["critical"]
        review["severity_scale_names"] = ["blocker", "bug", "nit"]
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        assert "Severity config mismatch" in body
        assert "`critical`" in body

    def test_inline_mismatch_note_survives_with_no_leftover_findings(self):
        """A mismatch can occur with zero non-postable findings (every
        finding is inline-anchored, or there are none at all) — the note
        must still surface rather than being swallowed by the "nothing to
        show inline -> empty body" shortcut."""
        mc = self._make_provider()
        review = {
            "severity": "medium", "summary": "ok", "findings": [],
            "unknown_severities": ["critical"],
            "severity_scale_names": ["blocker", "bug", "nit"],
        }
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        call = mc.submit_review.call_args
        body = call.args[2] if len(call.args) >= 3 else call.kwargs["body"]
        assert "Severity config mismatch" in body


class TestCachedMergeDispatch:
    """_maybe_dispatch_cached_merge: dispatch auto-merge from a cached
    approve verdict without a fresh AI review pass (TODO review-ops item,
    wedges from the PR #160-162 rollout). Safety invariants — every
    decline reason fails closed, and a dispatch reuses the SAME
    _safe_do_merge path (CI gate + force-push recheck) the review flows
    use."""

    DIFF = "diff --git a/f.py b/f.py\n+line\n"
    PR_KEY = "gitea:owner/repo#42"

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _diff_hashes(self, diff=None):
        from raven.reviewer import split_diff_by_file as _split
        return {f: hashlib.sha256(c.encode()).hexdigest()
                for f, c in _split(diff or self.DIFF)}

    def _seed_cache(self, verdict="approve", gap=(), hashes=None,
                    findings=None, summary="cached body"):
        import time as _time
        _previous_diffs[self.PR_KEY] = CacheEntry(
            timestamp=_time.time(),
            hashes=self._diff_hashes() if hashes is None else hashes,
            findings={"f.py": []} if findings is None else findings,
            verdict=verdict,
            summary=summary,
            coverage_gap_files=list(gap),
        )

    def _make_provider(self, sole=True):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = self.DIFF
        mc.get_authenticated_user.return_value = "Raven"
        if sole:
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "Raven"}, "state": "APPROVED"}]
        else:
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "alice"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_state.return_value = "open"
        mc.get_pr_head_sha.return_value = "abc123"
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        return mc

    def _call(self, mc, **kwargs):
        from raven.server import _maybe_dispatch_cached_merge
        kwargs.setdefault("head_sha", "abc123")
        return _maybe_dispatch_cached_merge(
            mc, "owner/repo", 42, "PR #42", "http://x", **kwargs)

    @staticmethod
    def _outcomes(mock_inc):
        return [c.args[1]["outcome"] for c in mock_inc.call_args_list
                if c.args[0] == "raven_cached_merge_dispatch_total"]

    # ── dispatch path ──────────────────────────────────────────────── #

    def test_dispatches_on_cached_approve_matching_head(self):
        self._seed_cache()
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            mock_exec.submit.return_value = MagicMock()
            result = self._call(mc)
        assert result is True
        mock_exec.submit.assert_called_once()
        args = mock_exec.submit.call_args[0]
        assert args[0] is _safe_do_merge
        assert args[2] == "owner/repo"
        assert args[3] == 42
        review_arg = args[6]
        assert review_arg["approve"] is True
        assert review_arg["summary"] == "cached body"
        assert args[7] == "abc123"          # head-SHA pinned for _do_merge
        assert self._outcomes(mock_inc) == ["dispatched"]

    def test_dispatch_synthetic_review_carries_scale_fields(self):
        """Finding 1 (PR #216 review, 2nd pass): the synthesized review
        dict must carry severity_scale_names / severity_blocks_at, like
        every REAL review dict does — without them notifier._scale_from_review
        silently reconstructs default_scale() for the merge_failed/
        ci_failed notification this dict eventually reaches, mis-colouring
        (and mis-filtering) a custom-scale repo."""
        from raven.severity import SeverityScale
        scale = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                              blocks_at_or_above="bug")
        self._seed_cache()
        mc = self._make_provider()
        with patch("raven.server.ci_wait_executor") as mock_exec:
            mock_exec.submit.return_value = MagicMock()
            self._call(mc, scale=scale)
        review_arg = mock_exec.submit.call_args[0][6]
        assert review_arg["severity_scale_names"] == ["blocker", "bug", "nit"]
        assert review_arg["severity_blocks_at"] == "bug"

    def test_dispatch_recomputes_hashes_when_not_supplied(self):
        """Without precomputed hashes the helper must fetch the CURRENT
        diff and hash it — the cached approval must describe the head."""
        self._seed_cache()
        mc = self._make_provider()
        with patch("raven.server.ci_wait_executor") as mock_exec:
            mock_exec.submit.return_value = MagicMock()
            result = self._call(mc, current_hashes=None)
        assert result is True
        mc.fetch_pr_diff.assert_called_once_with("owner/repo", 42)

    def test_dispatch_fetches_head_sha_when_not_supplied(self):
        self._seed_cache()
        mc = self._make_provider()
        with patch("raven.server.ci_wait_executor") as mock_exec:
            mock_exec.submit.return_value = MagicMock()
            result = self._call(mc, head_sha=None)
        assert result is True
        args = mock_exec.submit.call_args[0]
        assert args[7] == "abc123"

    def test_dispatch_synthetic_review_carries_cached_findings(self):
        finding = {"severity": "medium", "file": "f.py", "line": 3,
                   "message": "kept finding"}
        self._seed_cache(findings={"f.py": [finding]})
        mc = self._make_provider()
        with patch("raven.server.ci_wait_executor") as mock_exec:
            mock_exec.submit.return_value = MagicMock()
            assert self._call(mc) is True
        review_arg = mock_exec.submit.call_args[0][6]
        assert review_arg["findings"] == [finding]
        assert review_arg["severity"] == "medium"

    def test_dispatch_synthetic_review_uses_the_supplied_scale(self):
        """Task 14 (found outside its assigned file scope, fixed as part
        of it per team-lead ruling — same file, same defect class):
        _process_pr's no-changes-skip branch already resolves the repo's
        scale (``no_changes_scale``) to compute ``expected_config_hash``,
        but never threaded it into this call, so the synthesized notify
        payload's severity silently fell back to default_scale() and
        misreported any cached finding using a custom tier name. A
        'bug'-severity finding (unranked in the built-in vocabulary)
        must report 'bug' — not collapse to default_scale()'s
        least-severe reading ('low')."""
        from raven.severity import SeverityScale
        finding = {"severity": "bug", "file": "f.py", "line": 3, "message": "m"}
        self._seed_cache(findings={"f.py": [finding]})
        mc = self._make_provider()
        scale = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                              blocks_at_or_above="bug")
        with patch("raven.server.ci_wait_executor") as mock_exec:
            mock_exec.submit.return_value = MagicMock()
            assert self._call(mc, scale=scale) is True
        review_arg = mock_exec.submit.call_args[0][6]
        assert review_arg["severity"] == "bug"

    # ── safety invariants: every gate fails closed ─────────────────── #

    def _assert_declined(self, mc, mock_exec, mock_inc, result, reason):
        assert result is False
        mock_exec.submit.assert_not_called()
        mc.merge_pr.assert_not_called()
        assert self._outcomes(mock_inc) == [f"declined_{reason}"]

    def test_declines_when_cache_entry_missing(self):
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "no_cache_entry")

    def test_declines_when_verdict_needs_work(self):
        self._seed_cache(verdict="needs_work")
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "verdict_not_approve")

    def test_declines_when_verdict_none(self):
        self._seed_cache(verdict=None)
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "verdict_not_approve")

    def test_declines_on_coverage_gap(self):
        self._seed_cache(gap=["big.py"])
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "coverage_gap")

    def test_declines_on_hash_mismatch(self):
        """Cached approval pinned to different content than the current
        head — wedge 1's stale approval must never dispatch."""
        self._seed_cache(hashes={"f.py": "0" * 64})
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "hash_mismatch")

    def test_declines_on_extra_cached_file(self):
        """Hash state must match exactly — a cached file absent from the
        current diff is a mismatch, not a subset pass."""
        hashes = self._diff_hashes()
        hashes["gone.py"] = "1" * 64
        self._seed_cache(hashes=hashes)
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "hash_mismatch")

    def test_declines_when_not_sole_reviewer(self):
        self._seed_cache()
        mc = self._make_provider(sole=False)
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "not_sole_reviewer")

    def test_declines_in_advisory_mode(self, monkeypatch):
        monkeypatch.setattr(_server_mod, "RAVEN_REVIEW_MODE", "advisory")
        self._seed_cache()
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "advisory_mode")

    def test_declines_when_pr_not_open(self):
        self._seed_cache()
        mc = self._make_provider()
        mc.get_pr_state.return_value = "merged"
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "pr_not_open")

    def test_declines_when_pr_state_unverifiable(self):
        self._seed_cache()
        mc = self._make_provider()
        mc.get_pr_state.side_effect = RuntimeError("api down")
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc)
        self._assert_declined(mc, mock_exec, mock_inc, result, "pr_state_unverifiable")

    def test_declines_when_diff_fetch_fails(self):
        self._seed_cache()
        mc = self._make_provider()
        mc.fetch_pr_diff.side_effect = RuntimeError("api down")
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc, current_hashes=None)
        self._assert_declined(mc, mock_exec, mock_inc, result, "diff_fetch_failed")

    def test_declines_when_head_sha_unavailable(self):
        self._seed_cache()
        mc = self._make_provider()
        mc.get_pr_head_sha.side_effect = RuntimeError("api down")
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc, head_sha=None)
        self._assert_declined(mc, mock_exec, mock_inc, result, "no_head_sha")

    def test_declines_on_head_sentinel_with_precomputed_hashes(self):
        """head_sha='HEAD' + precomputed hashes: the helper must NOT
        re-fetch the sha (it would postdate the hashed diff — stale-
        approval race) and must NOT pin the 'HEAD' sentinel. Decline."""
        self._seed_cache()
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc, head_sha="HEAD",
                                current_hashes=self._diff_hashes())
        self._assert_declined(mc, mock_exec, mock_inc, result, "no_head_sha")
        mc.get_pr_head_sha.assert_not_called()

    # ── reused merge path keeps its own gates ──────────────────────── #

    def test_dispatched_merge_enforces_ci_gate(self):
        """CI failure inside the reused _safe_do_merge path blocks the
        merge — the cached dispatch adds no bypass."""
        self._seed_cache()
        mc = self._make_provider()
        mc.get_commit_status.return_value = "failure"
        with patch("raven.server.notify") as mock_notify:
            result = self._call(mc)       # inline ci_wait_executor fixture
        assert result is True             # dispatched — gate fired downstream
        mc.merge_pr.assert_not_called()
        assert mock_notify.call_args.kwargs["action"] == "ci_failed"

    def test_dispatched_merge_enforces_force_push_protection(self):
        """Head SHA drift between dispatch and merge (push during CI
        wait) — the reused _do_merge recheck must skip the merge."""
        self._seed_cache()
        mc = self._make_provider()
        mc.get_pr_head_sha.return_value = "fff999"   # drifted vs pinned abc123
        with patch("raven.server.notify"):
            result = self._call(mc, head_sha="abc123")
        assert result is True
        mc.merge_pr.assert_not_called()

    def test_dispatched_merge_merges_when_gates_pass(self):
        self._seed_cache()
        mc = self._make_provider()
        with patch("raven.server.notify"):
            result = self._call(mc)
        assert result is True
        mc.merge_pr.assert_called_once()
        assert mc.merge_pr.call_args.kwargs["head_sha"] == "abc123"


class TestNoChangesSkipCachedMergeDispatch:
    """Wedge 2 (PR #161): a retrigger push with zero changed files hits
    the no-changes skip, which used to return before any merge logic —
    a standing cached approval could never dispatch. The skip path now
    attempts the cached merge dispatch."""

    DIFF = "diff --git a/f.py b/f.py\n+line\n"
    PR_KEY = "gitea:owner/repo#42"

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _seed_cache(self, verdict="approve", gap=(), config_hash=None):
        # config_hash defaults to the REAL hash _maybe_dispatch_cached_merge
        # will expect for this class's provider fixture (mc.fetch_file
        # returns "" -> default_scale(), no prompt override) — not "".
        # Before the config-hash mismatch fix, "" silently skipped that
        # comparison, so an entry left at its old zero-value default
        # dispatched anyway; after the fix "" is treated as an active
        # mismatch (see TestCachedMergeRespectsConfigHash), so a fixture
        # meant to represent a CURRENTLY VALID cached approve — which is
        # what test_no_changes_skip_dispatches_cached_approve needs — must
        # carry a real, matching hash like any review completed under
        # this feature would.
        import time as _time
        if config_hash is None:
            from raven.server import _entry_config_hash
            from raven.severity import default_scale
            config_hash = _entry_config_hash(default_scale(), None)
        _previous_diffs[self.PR_KEY] = CacheEntry(
            timestamp=_time.time(),
            hashes={"f.py": hashlib.sha256(self.DIFF.encode()).hexdigest()},
            findings={"f.py": []},
            verdict=verdict,
            summary="cached body",
            coverage_gap_files=list(gap),
            config_hash=config_hash,
        )

    def _payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "http://x",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self, sole=True):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = self.DIFF
        mc.fetch_file.return_value = ""
        mc.get_authenticated_user.return_value = "Raven"
        if sole:
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "Raven"}, "state": "APPROVED"}]
            mc.get_pr_requested_reviewers.return_value = []
        else:
            mc.get_pr_reviews.return_value = [
                {"user": {"login": "alice"}, "state": "APPROVED"}]
            mc.get_pr_requested_reviewers.return_value = ["Raven"]
        mc.get_pr_state.return_value = "open"
        mc.get_pr_head_sha.return_value = "abc123"
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        return mc

    def test_no_changes_skip_dispatches_cached_approve(self):
        """The headline wedge-2 fix: cached approve + no changes →
        merge dispatches WITHOUT a fresh AI review pass."""
        self._seed_cache()
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()        # no AI pass
        mc.merge_pr.assert_called_once()       # merge still dispatched
        assert mc.merge_pr.call_args.kwargs["head_sha"] == "abc123"

    def test_no_changes_skip_dispatch_uses_the_repos_scale(self):
        """Task 14 (found outside its assigned scope, fixed as part of it
        per team-lead ruling — same file, same defect class): this branch
        already fetches ``no_changes_scale`` (to compute
        ``expected_config_hash``) but never threaded it into the
        dispatch call — the synthesized notify payload's severity
        silently fell back to default_scale(). A cached finding using
        the repo's own vocabulary ('bug') must report 'bug', not the
        default scale's fallback ('low')."""
        import json
        from raven.server import _entry_config_hash

        scale_json = json.dumps({
            "severities": {"nit": 10, "bug": 20, "blocker": 30},
            "blocks_at_or_above": "bug",
        })
        from raven.severity import from_json
        scale = from_json(scale_json)

        self._seed_cache(config_hash=_entry_config_hash(scale, None))
        _previous_diffs[self.PR_KEY].findings = {
            "f.py": [{"severity": "bug", "file": "f.py", "line": 1, "message": "m"}],
        }
        mc = self._make_provider()
        mc.fetch_file.side_effect = lambda repo, path, ref="HEAD": (
            scale_json if path.endswith("severities.json") else ""
        )
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server._safe_do_merge") as mock_merge,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()
        mock_merge.assert_called_once()
        synthetic_review = mock_merge.call_args.args[5]
        assert synthetic_review["severity"] == "bug"

    def test_no_changes_skip_does_not_merge_needs_work(self):
        self._seed_cache(verdict="needs_work")
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_no_changes_skip_does_not_merge_with_other_reviewer(self):
        self._seed_cache()
        mc = self._make_provider(sole=False)
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_no_changes_skip_does_not_merge_with_coverage_gap(self):
        self._seed_cache(gap=["big.py"])
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()
        mc.merge_pr.assert_not_called()

    def test_no_changes_skip_does_not_merge_in_advisory_mode(self, monkeypatch):
        monkeypatch.setattr(_server_mod, "RAVEN_REVIEW_MODE", "advisory")
        self._seed_cache()
        mc = self._make_provider()
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, self._payload())
        mock_review.assert_not_called()
        mc.merge_pr.assert_not_called()


class TestFetchSeverityScale:
    def _provider(self, mocker, body=None, exc=None):
        p = mocker.Mock()
        if exc is not None:
            p.fetch_file.side_effect = exc
        else:
            p.fetch_file.return_value = body
        return p

    def test_reads_from_the_base_ref(self, mocker):
        import raven.server as server
        p = self._provider(mocker, body='{"severities": {"nit": 1, "bad": 2}}')
        scale = server._fetch_severity_scale(p, "acme/repo", "base-sha")
        assert scale.ordered() == ["bad", "nit"]
        args, kwargs = p.fetch_file.call_args
        assert "severities.json" in args[1]
        assert kwargs.get("ref") == "base-sha"

    def test_missing_file_falls_back_to_default(self, mocker):
        import raven.server as server
        from raven.severity import default_scale
        p = self._provider(mocker, body=None)
        assert server._fetch_severity_scale(p, "acme/repo", "r").ranks == default_scale().ranks

    def test_invalid_file_falls_back_and_counts(self, mocker):
        import raven.server as server
        from raven.severity import default_scale

        inc = mocker.patch.object(server, "inc")
        p = self._provider(mocker, body='{"severities": {"only": 1}}')
        scale = server._fetch_severity_scale(p, "acme/repo", "r")

        assert scale.ranks == default_scale().ranks
        inc.assert_any_call("raven_severity_scale_invalid_total", {"repo": "acme/repo"})

    def test_fetch_error_falls_back(self, mocker):
        import raven.server as server
        from raven.severity import default_scale
        p = self._provider(mocker, exc=RuntimeError("boom"))
        assert server._fetch_severity_scale(p, "acme/repo", "r").ranks == default_scale().ranks

    def test_both_config_dirs_disabled_skips_the_fetch(self, mocker):
        import raven.server as server
        mocker.patch.object(server, "CONFIG_DIR", "")
        mocker.patch.object(server, "RULES_DIR", "")
        p = self._provider(mocker, body='{"severities": {"a": 1, "b": 2}}')
        server._fetch_severity_scale(p, "acme/repo", "r")
        p.fetch_file.assert_not_called()


class TestSeverityMismatchComment:
    def test_comment_names_offending_and_known_tiers(self):
        import raven.server as server
        review = {"severity": "blocker", "summary": "s", "findings": [],
                  "unknown_severities": ["critical", "major"],
                  "severity_scale_names": ["blocker", "bug", "nit"]}
        body = server._format_comment(review)
        assert "Severity config mismatch" in body
        assert "`critical`" in body and "`major`" in body
        assert "blocker" in body and "nit" in body

    def test_no_line_when_vocabulary_matches(self):
        import raven.server as server
        review = {"severity": "bug", "summary": "s", "findings": [],
                  "unknown_severities": [], "severity_scale_names": ["bug"]}
        assert "Severity config mismatch" not in server._format_comment(review)

    def test_absent_key_is_safe(self):
        """Cached/legacy reviews have no such key."""
        import raven.server as server
        body = server._format_comment({"severity": "low", "summary": "s", "findings": []})
        assert "Severity config mismatch" not in body

    def test_custom_scale_mismatch_names_the_file(self):
        """When a repo scale is actually in effect (names differ from the
        built-in default), the message should still point at the file
        that governs it — this is the useful, actionable case."""
        import raven.server as server
        review = {"severity": "blocker", "summary": "s", "findings": [],
                  "unknown_severities": ["critical"],
                  "severity_scale_names": ["blocker", "bug", "nit"]}
        body = server._format_comment(review)
        assert "severities.json" in body

    def test_default_scale_mismatch_does_not_name_a_nonexistent_file(self):
        """Finding 2 (PR #216 review, 2nd pass): unknown_severities fires
        just as often for the ~100% of repos with NO severities.json at
        all (default low/medium/high in effect) — where the file the old
        message pointed at was never read and likely doesn't exist. Detect
        this by comparing severity_scale_names against
        default_scale().ordered(); when they match, the message must not
        name that file, and should instead point at the real likely
        causes (a prompt override, or the model not honouring the
        vocabulary)."""
        import raven.server as server
        from raven.severity import default_scale
        review = {"severity": "high", "summary": "s", "findings": [],
                  "unknown_severities": ["critical"],
                  "severity_scale_names": default_scale().ordered()}
        body = server._format_comment(review)
        assert "Severity config mismatch" in body
        assert "severities.json" not in body
        assert "active severity scale" in body
        assert "prompt override" in body

    def test_inline_leftovers_renders_the_same_note(self):
        """Finding 2 (PR #216 review): _format_inline_leftovers must
        render the identical mismatch note _format_comment does, given the
        same review dict — RAVEN_REVIEW_OUTPUT=inline never calls
        _format_comment, so without this the note was invisible."""
        import raven.server as server
        review = {"severity": "blocker", "summary": "s", "findings": [],
                  "unknown_severities": ["critical", "major"],
                  "severity_scale_names": ["blocker", "bug", "nit"]}
        body = server._format_inline_leftovers([], review=review)
        assert "Severity config mismatch" in body
        assert "`critical`" in body and "`major`" in body
        assert "blocker" in body and "nit" in body

    def test_inline_leftovers_no_line_when_vocabulary_matches(self):
        import raven.server as server
        review = {"severity": "bug", "summary": "s", "findings": [],
                  "unknown_severities": [], "severity_scale_names": ["bug"]}
        assert server._format_inline_leftovers([], review=review) == ""

    def test_inline_leftovers_default_scale_mismatch_does_not_name_a_file(self):
        """Same fix, shared helper, other renderer."""
        import raven.server as server
        from raven.severity import default_scale
        review = {"severity": "high", "summary": "s", "findings": [],
                  "unknown_severities": ["critical"],
                  "severity_scale_names": default_scale().ordered()}
        body = server._format_inline_leftovers([], review=review)
        assert "severities.json" not in body
        assert "active severity scale" in body

    def test_inline_leftovers_review_omitted_is_safe(self):
        """Existing callers that don't pass review= (none left in
        production, but direct unit callers/tests) must keep working."""
        import raven.server as server
        assert server._format_inline_leftovers([]) == ""


class TestFormatCommentUsesTheRepoScale:
    """_format_comment / _format_inline_leftovers must colour findings from
    the SCALE THEY'RE GIVEN, not silently default_scale(). A real bug found
    in review: with a nit/bug/blocker repo scale, every custom tier name is
    unknown to default_scale(), and emoji()/normalize() fail closed to
    most-severe for an unrecognised name — so EVERY finding rendered 🔴
    regardless of its actual tier, defeating Phase A's position-based
    colour system entirely (the emoji conveyed nothing)."""

    def _scale(self):
        from raven.severity import SeverityScale
        return SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                             blocks_at_or_above="bug")

    def test_format_comment_colours_each_custom_tier_correctly(self):
        import raven.server as server
        scale = self._scale()
        review = {
            "severity": "blocker", "summary": "s",
            "findings": [
                {"severity": "blocker", "message": "m1"},
                {"severity": "bug", "message": "m2"},
                {"severity": "nit", "message": "m3"},
            ],
        }
        body = server._format_comment(review, scale=scale)
        assert "🔴 [blocker]" in body
        assert "🟠 [bug]" in body
        assert "🟡 [nit]" in body

    def test_format_comment_default_scale_renders_as_today(self):
        import raven.server as server
        review = {
            "severity": "high", "summary": "s",
            "findings": [
                {"severity": "high", "message": "m1"},
                {"severity": "medium", "message": "m2"},
                {"severity": "low", "message": "m3"},
            ],
        }
        body = server._format_comment(review)
        assert "🔴 [high]" in body
        assert "🟠 [medium]" in body
        assert "🟡 [low]" in body

    def test_inline_leftovers_colours_each_custom_tier_correctly(self):
        import raven.server as server
        scale = self._scale()
        findings = [
            {"severity": "blocker", "message": "m1"},
            {"severity": "bug", "message": "m2"},
            {"severity": "nit", "message": "m3"},
        ]
        body = server._format_inline_leftovers(findings, scale)
        assert "🔴 **[blocker]**" in body
        assert "🟠 **[bug]**" in body
        assert "🟡 **[nit]**" in body

    def test_inline_leftovers_default_scale_renders_as_today(self):
        import raven.server as server
        findings = [
            {"severity": "high", "message": "m1"},
            {"severity": "medium", "message": "m2"},
            {"severity": "low", "message": "m3"},
        ]
        body = server._format_inline_leftovers(findings)
        assert "🔴 **[high]**" in body
        assert "🟠 **[medium]**" in body
        assert "🟡 **[low]**" in body


class TestProcessPrThreadsScaleIntoRenderers:
    """Integration-level guard on the _process_pr call sites: the resolved
    repo scale must reach _format_comment / _format_inline_leftovers, not
    fall back to their default_scale(). Exercises the real dispatch path
    (not a direct unit call to the renderer) so a regression at the CALL
    SITE — not just the renderer signature — is caught."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.fetch_pr_diff.return_value = "diff --git a/f.py b/f.py\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        return mc

    def _scale(self):
        from raven.severity import SeverityScale
        return SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                             blocks_at_or_above="bug")

    def _submitted_body(self, mc):
        call = mc.submit_review.call_args
        return call.args[2] if len(call.args) >= 3 else call.kwargs["body"]

    def test_summary_mode_body_uses_the_repo_scale(self):
        mc = self._make_provider()
        review = {
            "severity": "bug", "summary": "s",
            "findings": [
                {"severity": "bug", "message": "the middle tier"},
                {"severity": "nit", "message": "the least severe tier"},
            ],
        }
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "summary"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server._fetch_severity_scale", return_value=self._scale()),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        body = self._submitted_body(mc)
        assert "🟠 [bug]" in body
        assert "🟡 [nit]" in body
        # The bug this guards against: both tiers rendering 🔴 identically.
        assert "🔴 [nit]" not in body

    def test_inline_mode_body_uses_the_repo_scale(self):
        mc = self._make_provider()
        review = {
            "severity": "bug", "summary": "s",
            "findings": [
                {"severity": "nit", "message": "no file/line — body-only"},
            ],
        }
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "inline"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server._fetch_severity_scale", return_value=self._scale()),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        body = self._submitted_body(mc)
        assert "🟡" in body
        assert "🔴" not in body


class TestPerEntryConfigHash:
    def test_entry_written_under_one_scale_misses_under_another(self):
        import raven.server as server
        from raven.severity import SeverityScale

        a = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b")
        b = SeverityScale(ranks={"x": 1, "y": 2}, blocks_at_or_above="y")

        entry = server.CacheEntry(timestamp=0, hashes={}, findings={},
                                  config_hash=server._entry_config_hash(a, None))
        assert server._entry_config_hash(b, None) != entry.config_hash

    def test_same_scale_and_override_hits(self):
        import raven.server as server
        from raven.severity import SeverityScale
        s = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b")
        assert server._entry_config_hash(s, "OVERRIDE") == server._entry_config_hash(s, "OVERRIDE")

    def test_override_change_alone_changes_the_hash(self):
        import raven.server as server
        from raven.severity import SeverityScale
        s = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b")
        assert server._entry_config_hash(s, "A") != server._entry_config_hash(s, "B")

    def test_legacy_entry_without_hash_is_a_miss(self):
        import raven.server as server
        from raven.severity import default_scale
        entry = server.CacheEntry(timestamp=0, hashes={}, findings={})
        assert entry.config_hash == ""
        assert entry.config_hash != server._entry_config_hash(default_scale(), None)


class TestCachedMergeRespectsConfigHash:
    """Merge-safety half of the per-entry config hash: _maybe_dispatch_cached_merge
    re-dispatches an auto-merge from a cached approve verdict WITHOUT a fresh
    AI pass, so a scale/prompt-override change on base_ref since that verdict
    was cached must not silently keep dispatching merges under the old
    config. A caller that doesn't supply ``expected_config_hash`` at all
    skips the comparison entirely — see TestPerEntryConfigHash and the
    existing TestCachedMergeDispatch / TestNoChangesSkipCachedMergeDispatch
    suites, which rely on exactly that skip and must keep passing unmodified.

    A caller that DOES supply ``expected_config_hash`` gets a strict
    equality check against ``entry.config_hash`` — INCLUDING a legacy
    entry whose ``config_hash == ""`` (written before this feature
    shipped). That is a deliberate asymmetry with `_process_pr`'s read
    path (see CacheEntry.config_hash's docstring): there, a "" mismatch
    triggers a fresh review that records a real hash and re-warms the
    entry, so treating it as "skip, not a miss" costs one review, once.
    Here, on the NO-review cached-merge-dispatch path, there is no write
    to re-warm from — an original implementation that special-cased ""
    as "skip the comparison" left a legacy entry able to auto-merge under
    *any* future scale change forever, defeating the entire point of this
    task. That hole was found by review, not by the original test suite
    (a fully green 1200+ run never caught it), which is why this class
    pins the strict behaviour explicitly."""

    DIFF = "diff --git a/f.py b/f.py\n+line\n"
    PR_KEY = "gitea:owner/repo#42"

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _diff_hashes(self):
        from raven.reviewer import split_diff_by_file as _split
        return {f: hashlib.sha256(c.encode()).hexdigest()
                for f, c in _split(self.DIFF)}

    def _seed_cache(self, config_hash=""):
        import time as _time
        _previous_diffs[self.PR_KEY] = CacheEntry(
            timestamp=_time.time(),
            hashes=self._diff_hashes(),
            findings={"f.py": []},
            verdict="approve",
            summary="cached body",
            config_hash=config_hash,
        )

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = self.DIFF
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [
            {"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_state.return_value = "open"
        mc.get_pr_head_sha.return_value = "abc123"
        mc.get_commit_status.return_value = "success"
        mc.merge_pr.return_value = True
        return mc

    def _call(self, mc, **kwargs):
        from raven.server import _maybe_dispatch_cached_merge
        kwargs.setdefault("head_sha", "abc123")
        return _maybe_dispatch_cached_merge(
            mc, "owner/repo", 42, "PR #42", "http://x", **kwargs)

    def test_matching_expected_hash_still_dispatches(self):
        """Positive control: a cached entry whose recorded config_hash
        matches what the caller expects right now is not blocked by this
        gate."""
        import raven.server as server
        self._seed_cache(config_hash="samehash1234")
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            mock_exec.submit.return_value = MagicMock()
            result = self._call(mc, expected_config_hash="samehash1234")
        assert result is True
        outcomes = [c.args[1]["outcome"] for c in mock_inc.call_args_list
                    if c.args[0] == "raven_cached_merge_dispatch_total"]
        assert outcomes == ["dispatched"]

    def test_mismatched_expected_hash_declines(self):
        """The gate this task exists for: a cached approve computed under
        one config_hash must never dispatch under a different one."""
        self._seed_cache(config_hash="oldhash1234")
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc, expected_config_hash="newhash5678")
        assert result is False
        mock_exec.submit.assert_not_called()
        mc.merge_pr.assert_not_called()
        outcomes = [c.args[1]["outcome"] for c in mock_inc.call_args_list
                    if c.args[0] == "raven_cached_merge_dispatch_total"]
        assert outcomes == ["declined_config_hash_mismatch"]

    def test_legacy_entry_with_no_recorded_hash_is_refused(self):
        """CRITICAL: an entry cached before this feature shipped has
        config_hash="". Unlike _process_pr's read path — where a ""
        mismatch triggers a fresh review that records a real hash and
        re-warms the entry — this is the NO-review cached-merge-dispatch
        path: there is no write here to re-warm from. Skipping the
        comparison for a hash-less entry would let it auto-merge under
        ANY future severity-scale / prompt-override change forever,
        which is exactly the hole this task exists to close. The entry
        must be refused (declined_config_hash_mismatch) whenever the
        caller supplies a real expected hash, the same as any other
        config_hash inequality — "" is not a wildcard."""
        self._seed_cache(config_hash="")
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            result = self._call(mc, expected_config_hash="anyrealhash999")
        assert result is False
        mock_exec.submit.assert_not_called()
        mc.merge_pr.assert_not_called()
        outcomes = [c.args[1]["outcome"] for c in mock_inc.call_args_list
                    if c.args[0] == "raven_cached_merge_dispatch_total"]
        assert outcomes == ["declined_config_hash_mismatch"]

    def test_no_expected_hash_supplied_skips_the_check(self):
        """A caller that doesn't participate (omits the kwarg entirely)
        gets today's behaviour — the default used by every pre-existing
        call site/test."""
        self._seed_cache(config_hash="somehash")
        mc = self._make_provider()
        with (
            patch("raven.server.ci_wait_executor") as mock_exec,
            patch("raven.server.inc") as mock_inc,
        ):
            mock_exec.submit.return_value = MagicMock()
            result = self._call(mc)
        assert result is True

    def test_no_changes_skip_declines_dispatch_on_scale_change(self):
        """End-to-end via _process_pr's no-changes-skip path (wedge 2's
        recovery branch): a cached approve recorded under one severity
        scale must not auto-merge once the repo's severities.json (read
        from base_ref) has changed, even though the diff itself didn't."""
        import raven.server as server
        from raven.severity import SeverityScale

        old_scale = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above="b")
        self._seed_cache(config_hash=server._entry_config_hash(old_scale, None))
        mc = self._make_provider()
        mc.fetch_file.return_value = '{"severities": {"x": 1, "y": 2}, "blocks_at_or_above": "y"}'
        payload = {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "http://x",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, payload)
        mock_review.assert_not_called()          # still no fresh AI pass
        mc.merge_pr.assert_not_called()           # but the scale changed — no merge

    def test_no_changes_skip_declines_dispatch_for_legacy_hashless_entry(self):
        """CRITICAL — the hole found in Task 10's review: a cached approve
        of UNKNOWN provenance (config_hash="", written before this feature
        shipped — or simply never validated against any scale) must not
        auto-merge via the no-changes-skip recovery path just because it
        has no recorded hash to compare against. Reproduces the exact
        live sequence the reviewer found: legacy entry -> repo's
        severities.json changes on base_ref -> _process_pr's no-changes
        path -> review_diff is NOT called (no fresh AI pass) but merge_pr
        must ALSO not be called. Distinct from
        test_no_changes_skip_declines_dispatch_on_scale_change above
        (real-hash vs different-real-hash) — this is "" vs a real hash,
        the specific case the original implementation special-cased into
        skipping the comparison instead of refusing."""
        self._seed_cache(config_hash="")
        mc = self._make_provider()
        mc.fetch_file.return_value = '{"severities": {"x": 1, "y": 2}, "blocks_at_or_above": "y"}'
        payload = {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "http://x",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }
        with (
            patch("raven.server.review_diff") as mock_review,
            patch("raven.server.notify"),
        ):
            _process_pr(mc, payload)
        mock_review.assert_not_called()          # still no fresh AI pass
        mc.merge_pr.assert_not_called()           # hash-less entry — no merge either


class TestCacheWriteRecordsConfigHash:
    """The write side of Task 10: a completed review's cache entry must
    record the config_hash it was computed under, or every future
    config-hash comparison is vacuous (stored "" forever vs a real hash,
    a permanent one-way mismatch instead of the intended one-time miss)."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.fetch_pr_diff.return_value = "diff --git a/f.py b/f.py\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.side_effect = [
            [],                                                     # auto-add check
            [{"user": {"login": "Raven"}, "state": "APPROVED"}],   # gate check
        ]
        return mc

    def test_fresh_review_records_the_current_config_hash(self):
        import raven.server as server
        from raven.severity import default_scale

        mc = self._make_provider()
        payload = {
            "repo": "owner/repo", "sender": "alice", "pr_number": 99,
            "pr_title": "PR #99", "pr_url": "http://x",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }
        with (
            patch("raven.server.review_diff",
                  return_value={"severity": "low", "summary": "ok", "findings": []}),
            patch("raven.server.notify"),
        ):
            _process_pr(mc, payload)
        entry = _previous_diffs["gitea:owner/repo#99"]
        assert entry.config_hash == server._entry_config_hash(default_scale(), None)
        assert entry.config_hash != ""


class TestApproveGateUsesTheRepoScale:
    """Task 13 — THE CORE OF THE FEATURE. The approve decision must read
    the repo's SeverityScale, not the built-in SEVERITY_ORDER vocabulary
    via severity_gte. Every custom tier name is unknown to SEVERITY_ORDER,
    ranks 0, ties with the threshold, and approves — including the repo's
    most severe tier. See PR #211's disarmed-gate bug, one layer up."""

    def _scale(self):
        from raven.severity import SeverityScale
        return SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                             blocks_at_or_above="bug")

    def test_below_blocking_tier_approves(self):
        import raven.server as server
        assert server._approve_from_severity("nit", self._scale()) is True

    def test_at_blocking_tier_blocks(self):
        import raven.server as server
        assert server._approve_from_severity("bug", self._scale()) is False

    def test_above_blocking_tier_blocks(self):
        import raven.server as server
        assert server._approve_from_severity("blocker", self._scale()) is False

    def test_unknown_tier_blocks(self):
        """Fail closed — an unrecognised name must never approve."""
        import raven.server as server
        assert server._approve_from_severity("wat", self._scale()) is False

    def test_nothing_blocks_scale_approves_everything(self):
        from raven.severity import SeverityScale
        import raven.server as server
        s = SeverityScale(ranks={"a": 1, "b": 2}, blocks_at_or_above=None)
        assert server._approve_from_severity("b", s) is True

    def test_default_scale_reproduces_todays_decisions(self, monkeypatch):
        """The env-configured default must decide exactly as severity_gte does."""
        import raven.server as server
        from raven.severity import default_scale
        from raven.reviewer import severity_gte

        for env in ("low", "medium", "high"):
            monkeypatch.setenv("REVIEW_APPROVE_MAX_SEVERITY", env)
            scale = default_scale()
            for sev in ("low", "medium", "high"):
                assert server._approve_from_severity(sev, scale) is severity_gte(env, sev), \
                    f"divergence at REVIEW_APPROVE_MAX_SEVERITY={env}, severity={sev}"


class TestInlineCommentBodyDefaultsToScaleNotLiteralLow:
    """The eighth instance of this feature's recurring defect class: a
    default-vocabulary literal ('low') sitting next to a SeverityScale.
    9d2d478 fixed three .get("severity", "low") sites in _process_pr's
    inline-comment construction and missed the actual inline comment
    body line — reachable because CacheEntry.findings loads straight
    from JSON with no per-finding validation, so a legacy/malformed
    carried finding with no 'severity' key hits this. Under a custom
    scale it must render that scale's least-severe tier and colour, not
    the literal 'low' / 🔴."""

    def setup_method(self):
        _recent_prs.clear()
        _previous_diffs.clear()

    def _payload(self):
        return {
            "repo": "owner/repo", "sender": "alice", "pr_number": 42,
            "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
            "head_sha": "abc123", "head_ref": "feature", "base_ref": "main",
        }

    def _make_provider(self):
        mc = MagicMock(spec=GitProvider)
        mc.name = "gitea"
        mc.get_authenticated_user.return_value = "Raven"
        mc.get_pr_reviews.return_value = [{"user": {"login": "Raven"}, "state": "APPROVED"}]
        mc.get_pr_requested_reviewers.return_value = []
        mc.get_pr_head_sha.return_value = "abc123"
        mc.fetch_pr_diff.return_value = "diff --git a/f.py b/f.py\n+line\n"
        mc.fetch_file.return_value = ""
        mc.submit_review.return_value = {"id": 1}
        mc.add_label_to_pr.return_value = None
        mc.merge_pr.return_value = True
        mc.get_commit_status.return_value = "success"
        return mc

    def _scale(self):
        from raven.severity import SeverityScale
        return SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                             blocks_at_or_above="bug")

    def test_finding_with_no_severity_key_uses_scale_least_severe(self):
        """No 'severity' key at all (not even an unrecognised one) — the
        exact shape a pre-validation legacy cache entry can carry."""
        mc = self._make_provider()
        review = {
            "severity": "nit", "summary": "s",
            "findings": [
                {"file": "f.py", "line": 3, "message": "no severity key"},
            ],
        }
        with (
            patch("raven.server.RAVEN_REVIEW_OUTPUT", "both"),
            patch("raven.server.review_diff", return_value=review),
            patch("raven.server._fetch_severity_scale", return_value=self._scale()),
            patch("raven.server.notify"),
            patch("raven.server.time.sleep"),
        ):
            _process_pr(mc, self._payload())
        inline = mc.submit_review.call_args.kwargs["inline_comments"]
        assert len(inline) == 1
        body = inline[0]["body"]
        assert "🟡" in body
        assert "[nit]" in body
        assert "🔴" not in body
        assert "[low]" not in body
