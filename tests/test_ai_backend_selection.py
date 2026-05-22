"""Tests for raven.ai backend registry and _select_backend() selection logic."""

import os
import pytest



class TestSelectBackendExplicit:
    def test_explicit_claude_cli(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "claude_cli")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")
        # Clear opposite-backend creds so they don't interfere
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"

    def test_explicit_openai_compatible(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "openai_compatible")
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000")
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "openai_compatible"

    def test_explicit_bogus_value_raises(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "not_a_backend")

        from raven.ai import _select_backend
        with pytest.raises(RuntimeError, match="Unknown RAVEN_AI_BACKEND"):
            _select_backend()

    def test_explicit_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "  Claude_CLI  ")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"

    def test_explicit_accepts_dashes(self, monkeypatch):
        """Operator-friendly: ``claude-cli`` (dash) resolves to
        ``claude_cli``. The underscore-vs-dash spelling shouldn't be a
        configuration footgun."""
        monkeypatch.setenv("RAVEN_AI_BACKEND", "claude-cli")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"

    def test_explicit_accepts_dashes_for_openai(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "OPENAI-COMPATIBLE")
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000")
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "openai_compatible"


class TestSelectBackendAutoDetect:
    def test_autodetect_openai_wins_when_both_creds_set(self, monkeypatch):
        """OpenAI creds present → openai_compatible wins even if
        CLAUDE_CODE_OAUTH_TOKEN is also set."""
        monkeypatch.delenv("RAVEN_AI_BACKEND", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000")
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "openai_compatible"

    def test_autodetect_claude_cli_when_only_token_set(self, monkeypatch):
        monkeypatch.delenv("RAVEN_AI_BACKEND", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"

    def test_autodetect_no_creds_raises(self, monkeypatch):
        monkeypatch.delenv("RAVEN_AI_BACKEND", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)

        from raven.ai import _select_backend
        with pytest.raises(RuntimeError, match="No AI backend credentials"):
            _select_backend()

    def test_autodetect_openai_needs_both_key_and_base(self, monkeypatch):
        """RAVEN_AI_API_KEY without RAVEN_AI_API_BASE does NOT select openai_compatible."""
        monkeypatch.delenv("RAVEN_AI_BACKEND", raising=False)
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"

    def test_autodetect_openai_needs_both_base_and_key(self, monkeypatch):
        """Inverse of the previous test: RAVEN_AI_API_BASE alone (without
        RAVEN_AI_API_KEY) must NOT pick openai_compatible. Both halves of
        the OpenAI credential pair are required for auto-selection."""
        monkeypatch.delenv("RAVEN_AI_BACKEND", raising=False)
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000")
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")

        from raven.ai import _select_backend
        backend = _select_backend()
        assert backend.name == "claude_cli"


class TestGetBackendCaches:
    def test_get_backend_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_BACKEND", "claude_cli")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "dummy")

        from raven.ai import get_backend, _reset_backend_cache
        _reset_backend_cache()  # test helper
        try:
            first = get_backend()
            second = get_backend()
            assert first is second
        finally:
            # Don't leak the populated cache to later tests in the suite —
            # monkeypatch restores env vars on teardown but the module-level
            # _cached_backend would otherwise persist.
            _reset_backend_cache()
