"""Tests for the AIBackend abstract base class."""

import pytest

from raven.ai.base import AIBackend, AIError


class TestAIBackendABC:
    def test_complete_is_abstract(self):
        """AIBackend cannot be instantiated without overriding complete."""
        with pytest.raises(TypeError):
            AIBackend()

    def test_shutdown_has_default_no_op(self):
        """shutdown() returns 0 by default (for HTTP-only backends)."""

        class Concrete(AIBackend):
            name = "concrete"

            def complete(self, prompt, *, model, effort, timeout, purpose):
                return "ok"

        backend = Concrete()
        assert backend.shutdown() == 0
        assert backend.shutdown(grace_period=5.0) == 0

    def test_subclass_must_set_name(self):
        """Subclasses should provide a name attribute — enforced only by convention."""

        class Concrete(AIBackend):
            name = "concrete"

            def complete(self, prompt, *, model, effort, timeout, purpose):
                return "ok"

        assert Concrete().name == "concrete"


class TestAIError:
    """AIError carries a classified ``.reason`` so the cause survives from
    the backend up to server.py (which builds the operator-facing comment
    + metric from it). It subclasses RuntimeError so every existing
    ``except RuntimeError`` / ``except Exception`` path keeps catching it."""

    def test_is_runtimeerror_subclass(self):
        # reviewer.py / server.py catch RuntimeError + Exception uniformly;
        # AIError must remain catchable by those clauses.
        assert issubclass(AIError, RuntimeError)
        assert isinstance(AIError("boom", reason="timeout"), RuntimeError)

    def test_reason_defaults_to_unknown(self):
        assert AIError("boom").reason == "unknown"

    def test_reason_is_stored(self):
        assert AIError("boom", reason="rate_limit").reason == "rate_limit"

    def test_message_preserved(self):
        assert str(AIError("the message", reason="auth")) == "the message"

    @pytest.mark.parametrize("reason", ["timeout", "rate_limit", "backend_5xx"])
    def test_retryable_reasons(self, reason):
        assert AIError("x", reason=reason).retryable is True

    @pytest.mark.parametrize("reason", ["usage_limit", "auth", "unknown"])
    def test_non_retryable_reasons(self, reason):
        assert AIError("x", reason=reason).retryable is False
