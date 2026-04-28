"""raven.ai — pluggable AI backends.

Each backend implements the AIBackend ABC (in base.py) and turns a prompt
into a text response. reviewer.py stays backend-agnostic: it builds the
prompt and parses the response, and calls get_backend().complete() for the
actual model invocation.
"""

import logging
import os
import threading

from raven.ai.base import AIBackend
from raven.ai.claude_cli import ClaudeCLIBackend

logger = logging.getLogger(__name__)

_KNOWN_BACKENDS = {"claude_cli", "openai_compatible"}

_backend_lock = threading.Lock()
_cached_backend: AIBackend | None = None


def _select_backend() -> AIBackend:
    """Pick the right backend per the spec's hybrid rule.

    1. Explicit RAVEN_AI_BACKEND override → that backend.
    2. RAVEN_AI_API_KEY + RAVEN_AI_API_BASE both set → openai_compatible.
    3. CLAUDE_CODE_OAUTH_TOKEN set → claude_cli.
    4. Otherwise: RuntimeError.

    Case- and whitespace-insensitive for the override.
    """
    override = (os.environ.get("RAVEN_AI_BACKEND") or "").strip().lower()
    if override:
        if override not in _KNOWN_BACKENDS:
            raise RuntimeError(f"Unknown RAVEN_AI_BACKEND={override!r}")
        return _instantiate(override)

    if os.environ.get("RAVEN_AI_API_KEY") and os.environ.get("RAVEN_AI_API_BASE"):
        return _instantiate("openai_compatible")
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return _instantiate("claude_cli")

    raise RuntimeError(
        "No AI backend credentials configured — set "
        "CLAUDE_CODE_OAUTH_TOKEN, or RAVEN_AI_API_BASE + RAVEN_AI_API_KEY, "
        "or set RAVEN_AI_BACKEND explicitly."
    )


def _instantiate(name: str) -> AIBackend:
    """Construct the named backend. Raises at first failure."""
    if name == "claude_cli":
        return ClaudeCLIBackend()
    if name == "openai_compatible":
        # Imported lazily so the openai dep is only required when the
        # backend is actually used. Task 6 adds OpenAICompatibleBackend.
        from raven.ai.openai_compatible import OpenAICompatibleBackend
        return OpenAICompatibleBackend()
    raise RuntimeError(f"Unknown backend: {name!r}")


def get_backend() -> AIBackend:
    """Return the process-wide backend instance (cached after first call)."""
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend
    with _backend_lock:
        if _cached_backend is None:
            _cached_backend = _select_backend()
            from raven.reviewer import CLAUDE_MODEL
            logger.info(
                "Using AI backend: %s (model=%s)",
                _cached_backend.name,
                CLAUDE_MODEL,
            )
    return _cached_backend


def _reset_backend_cache() -> None:
    """Test helper — drop the cached backend so the next get_backend()
    re-runs selection against the current environment."""
    global _cached_backend
    with _backend_lock:
        _cached_backend = None
