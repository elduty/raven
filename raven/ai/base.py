"""AIBackend — abstract base class for AI provider backends."""

from abc import ABC, abstractmethod


class AIBackend(ABC):
    """Interface for AI backends used by raven.reviewer.

    Implementations must set ``name`` to a stable short identifier
    (``claude_cli`` | ``openai_compatible``) used for logs and the
    findings-cache config hash.
    """

    name: str = ""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        timeout: int,
        purpose: str,
    ) -> str:
        """Single-turn completion.

        Returns the model's raw text output. Raises ``RuntimeError`` on
        any transport, timeout, or protocol failure — reviewer.py catches
        ``RuntimeError`` uniformly regardless of backend.

        ``purpose`` is one of ``"review"`` | ``"respond"`` — used only
        for log messages; backends may route or tune based on it but
        need not.
        """
        raise NotImplementedError

    def shutdown(self, grace_period: float = 2.0) -> int:
        """Cancel in-flight requests; return count signalled.

        Default is a no-op (returns 0) for HTTP-based backends that
        don't need explicit teardown. The Claude-CLI backend overrides
        this to SIGTERM/SIGKILL running subprocesses on gunicorn
        shutdown.
        """
        return 0
