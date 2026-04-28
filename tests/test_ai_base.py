"""Tests for the AIBackend abstract base class."""

import pytest

from raven.ai.base import AIBackend


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
