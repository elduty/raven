"""Global pytest setup.

Sets the small set of env vars that every test file's imports rely on,
so individual test modules don't each repeat the same `os.environ.setdefault`
prelude (which leaked into the broader test session and made test order
matter for any module that later asserted on these vars being unset).

Tests targeting a specific backend or auth path override these via
`monkeypatch.setenv` / `monkeypatch.delenv`, which is scoped per-test
and properly restored at teardown.
"""

import os


def pytest_configure(config):
    """Register custom markers so pytest doesn't emit unknown-marker
    warnings. ``slow`` is used for tests that hit real external APIs
    (Claude / Anthropic) and only run with explicit opt-in
    (RAVEN_LIVE_AI_TESTS=1)."""
    config.addinivalue_line(
        "markers",
        "slow: real-AI integration tests; opt-in via RAVEN_LIVE_AI_TESTS=1",
    )

# Provider config — required by raven.providers and raven.server imports
os.environ.setdefault("GITEA_URL", "https://gitea.example.com")
os.environ.setdefault("GITEA_TOKEN", "test-token")

# Pin the default AI backend to claude_cli for the test session. Tests
# that exercise auto-selection explicitly monkeypatch.delenv this so
# the registry's detection logic runs against the test-specific creds.
os.environ.setdefault("RAVEN_AI_BACKEND", "claude_cli")
os.environ.setdefault("CLAUDE_CODE_OAUTH_TOKEN", "dummy-token-for-tests")

# Webhook secret — server.py reads it at import time; failing to set it
# produces a startup error before any test can run.
os.environ.setdefault("GITEA_WEBHOOK_SECRET", "testsecret")

# OpenAI-compatible backend defaults — set so tests that instantiate
# OpenAICompatibleBackend() without a per-test monkeypatch get a valid
# config. Tests for backend auto-selection explicitly monkeypatch.delenv
# both vars when they need to verify the no-creds path.
os.environ.setdefault("RAVEN_AI_API_BASE", "http://proxy.example:4000")
os.environ.setdefault("RAVEN_AI_API_KEY", "sk-test")
