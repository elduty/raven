"""AIBackend — abstract base class for AI provider backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# Retryable failure classes: a short in-process retry can plausibly help.
# ``usage_limit`` (Claude plan cap) resets hours later, ``auth`` needs a
# config fix, and ``unknown`` is unclassified — none benefit from an
# immediate retry, so they're excluded.
_RETRYABLE_REASONS = frozenset({"timeout", "rate_limit", "backend_5xx"})


class AIError(RuntimeError):
    """A classified AI-backend failure.

    Carries a ``.reason`` so the cause survives from the backend (where the
    real exception type is known) up to server.py, which turns it into an
    actionable operator-facing comment + the ``raven_review_failures_total``
    metric, and reviewer.py, which decides whether to retry.

    Subclasses ``RuntimeError`` deliberately: reviewer.py and server.py
    already catch ``RuntimeError`` / ``Exception`` uniformly across both
    backends, so every existing handler keeps working — the ``.reason`` is
    additive.

    ``reason`` is one of:
      ``timeout``      — request exceeded RAVEN_AI_TIMEOUT (RETRYABLE)
      ``rate_limit``   — provider 429 (RETRYABLE, after a backoff)
      ``backend_5xx``  — provider 5xx / transient connection drop (RETRYABLE)
      ``usage_limit``  — Claude plan/session cap; resets later (NOT retryable)
      ``auth``         — bad/expired credentials (NOT retryable)
      ``unknown``      — unclassified fallback (NOT retryable)

    ``parse_error`` is intentionally NOT an AIError reason — a malformed
    model response flows through reviewer.py's existing ``_parse_error``
    dict path, not this exception type.
    """

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def retryable(self) -> bool:
        return self.reason in _RETRYABLE_REASONS


@dataclass
class CompletionResult:
    """Return value of ``AIBackend.complete``.

    ``text`` is the model's raw text output (what callers consume).
    ``input_tokens`` / ``output_tokens`` are the usage counts the backend
    could extract (0 when unavailable). ``cost_usd`` is the
    provider-reported cost for this call when the backend surfaces one
    (Claude CLI ``total_cost_usd``; LiteLLM ``x-litellm-response-cost``),
    or ``None`` when it doesn't — in which case the caller falls back to
    the configured price table. Usage/cost are best-effort telemetry and
    never affect ``text``.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


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
    ) -> CompletionResult:
        """Single-turn completion.

        Returns a :class:`CompletionResult` carrying the model's text plus
        best-effort token/cost telemetry. Raises ``RuntimeError`` on any
        transport, timeout, or protocol failure — reviewer.py catches
        ``RuntimeError`` uniformly regardless of backend.

        ``purpose`` is one of ``"review"`` | ``"respond"`` |
        ``"consolidate"`` — used for log messages and metric labels;
        backends may route or tune based on it but need not.
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
