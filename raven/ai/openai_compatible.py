"""OpenAICompatibleBackend — chat/completions via the openai SDK.

Works with any endpoint that speaks the OpenAI chat/completions protocol:
the LiteLLM proxy, vLLM, Ollama's OpenAI shim, actual OpenAI, etc. The
endpoint, auth, and routed model are configured by the operator; this
backend only knows how to send a prompt and parse the text back.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager

import openai

from raven.ai.base import AIBackend, CompletionResult

logger = logging.getLogger(__name__)

_EFFORT_TO_REASONING = {
    "max":    "high",
    "high":   "high",
    "medium": "medium",
    "low":    "low",
    "none":   None,  # omit reasoning_effort kwarg entirely
}

_MAX_CONCURRENT = max(int(os.environ.get("RAVEN_AI_MAX_CONCURRENT", "4")), 1)

# LiteLLM proxies report the per-call cost in this response header. Other
# OpenAI-compatible endpoints don't send it → cost stays None and the caller
# falls back to the configured price table.
_LITELLM_COST_HEADER = "x-litellm-response-cost"


def _parse_cost_header(headers) -> float | None:
    """Read the LiteLLM per-call cost header, or None if absent/unparseable.

    ``headers`` is the httpx.Headers off the raw response (case-insensitive).
    Cost is telemetry — a malformed value must never break the completion.
    """
    try:
        raw = headers.get(_LITELLM_COST_HEADER)
    except Exception:
        return None
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Unparseable %s header: %r", _LITELLM_COST_HEADER, raw)
        return None


class OpenAICompatibleBackend(AIBackend):
    name = "openai_compatible"

    def __init__(self) -> None:
        base_url = (os.environ.get("RAVEN_AI_API_BASE") or "").rstrip("/")
        api_key = os.environ.get("RAVEN_AI_API_KEY") or ""
        raw_max_tokens = os.environ.get("RAVEN_AI_MAX_TOKENS") or ""
        if not base_url:
            raise RuntimeError(
                "RAVEN_AI_API_BASE is required for the openai_compatible backend"
            )
        if not api_key:
            raise RuntimeError(
                "RAVEN_AI_API_KEY is required for the openai_compatible backend"
            )

        self._max_tokens: int | None = int(raw_max_tokens) if raw_max_tokens else None
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
        )
        self._semaphore = threading.Semaphore(_MAX_CONCURRENT)
        self._inflight: set[int] = set()
        self._inflight_lock = threading.Lock()
        self._next_ticket = 0

    @contextmanager
    def _track_request(self):
        with self._inflight_lock:
            self._next_ticket += 1
            ticket = self._next_ticket
            self._inflight.add(ticket)
        try:
            yield ticket
        finally:
            with self._inflight_lock:
                self._inflight.discard(ticket)

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        timeout: int,
        purpose: str,
    ) -> CompletionResult:
        reasoning = _EFFORT_TO_REASONING.get(effort)
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout,
        }
        if reasoning is not None:
            kwargs["reasoning_effort"] = reasoning
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        logger.info(
            "OpenAI-compatible %s (model=%s reasoning=%s prompt_chars=%d)",
            purpose, model, reasoning or "off", len(prompt),
        )

        # Use with_raw_response so we can read response headers — LiteLLM
        # returns the per-call cost in ``x-litellm-response-cost``. ``.parse()``
        # gives the usual typed ChatCompletion body. Non-LiteLLM endpoints
        # simply won't send the header (cost stays None → caller falls back
        # to the price table).
        with self._semaphore, self._track_request():
            try:
                raw = self._client.chat.completions.with_raw_response.create(**kwargs)
            except openai.APITimeoutError as e:
                raise RuntimeError(f"AI backend timed out after {timeout}s: {e}")
            except openai.APIError as e:
                raise RuntimeError(f"AI backend error: {e}")

        cost_usd = _parse_cost_header(raw.headers)
        response = raw.parse()

        if not response.choices:
            raise RuntimeError("AI backend returned no choices")
        choice = response.choices[0]
        # The OpenAI SDK guarantees ``content: str | None`` for standard
        # chat completions, but proxies (LiteLLM, vLLM, Ollama shim) and
        # newer multimodal/reasoning APIs can return a list of content
        # parts. Coerce known shapes; treat anything else as empty so
        # the empty-response branch fires with a clean error instead of
        # an AttributeError from ``str.upper()`` or similar downstream.
        content = choice.message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Multimodal/reasoning response: concatenate text-typed parts.
            text = "".join(
                getattr(p, "text", "") for p in content
                if getattr(p, "type", "") == "text"
            )
        else:
            text = ""
        if not text:
            raise RuntimeError(
                f"AI backend returned empty response "
                f"(finish_reason={choice.finish_reason})"
            )

        # Usage is best-effort — some proxies omit it. Never let a missing
        # usage object break the response.
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    def shutdown(self, grace_period: float = 2.0) -> int:
        """Close the HTTP client; return the count of in-flight requests
        still active at close time.

        Polls ``_inflight`` up to ``grace_period`` seconds for natural
        completion, then closes ``self._client``. The openai SDK exposes
        no per-request cancellation hook, so any requests still in
        flight at close fail when the underlying httpx connection is
        torn down. The returned count reflects what was still pending —
        it is not a cancellation count.
        """
        deadline = time.monotonic() + max(grace_period, 0.0)
        while time.monotonic() < deadline:
            with self._inflight_lock:
                if not self._inflight:
                    break
            time.sleep(0.05)
        with self._inflight_lock:
            count = len(self._inflight)
        if count:
            logger.info(
                "Closing OpenAI-compatible client with %d in-flight request(s)",
                count,
            )
        try:
            self._client.close()
        except Exception:  # pragma: no cover — best-effort teardown
            pass
        return count
