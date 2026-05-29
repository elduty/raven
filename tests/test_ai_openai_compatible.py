"""Tests for raven.ai.openai_compatible — OpenAI-compatible HTTP backend."""

import os
from unittest.mock import MagicMock, patch

import pytest


def _make_response(text: str, finish_reason: str = "stop",
                   prompt_tokens: int = 0, completion_tokens: int = 0) -> MagicMock:
    """Build a mock openai ChatCompletion response with one choice + usage."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def _raw(response, cost=None) -> MagicMock:
    """Wrap a parsed response as a with_raw_response object: ``.parse()``
    returns the body, ``.headers`` carries the optional LiteLLM cost."""
    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {} if cost is None else {"x-litellm-response-cost": str(cost)}
    return raw


def _patch_create(backend, *, return_value=None, side_effect=None):
    """Patch the with_raw_response.create call the backend now uses."""
    target = backend._client.chat.completions.with_raw_response
    if side_effect is not None:
        return patch.object(target, "create", side_effect=side_effect)
    return patch.object(target, "create", return_value=return_value)


class TestOpenAICompatibleBackendComplete:
    def test_name_attribute(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend
        assert OpenAICompatibleBackend().name == "openai_compatible"

    def test_complete_sends_prompt_as_user_message(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("model output"))) as mock_create:
            out = backend.complete(
                "the prompt",
                model="claude-opus-4-7",
                effort="max",
                timeout=600,
                purpose="review",
            )
        assert out.text == "model output"
        _, kwargs = mock_create.call_args
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
        assert kwargs["timeout"] == 600

    @pytest.mark.parametrize("effort,expected", [
        ("max", "high"),
        ("high", "high"),
        ("medium", "medium"),
        ("low", "low"),
    ])
    def test_effort_maps_to_reasoning_effort(self, effort, expected):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("x"))) as mock_create:
            backend.complete(
                "p", model="m", effort=effort, timeout=60, purpose="review",
            )
        _, kwargs = mock_create.call_args
        assert kwargs.get("reasoning_effort") == expected

    def test_effort_none_omits_reasoning_effort_kwarg(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("x"))) as mock_create:
            backend.complete(
                "p", model="m", effort="none", timeout=60, purpose="review",
            )
        _, kwargs = mock_create.call_args
        assert "reasoning_effort" not in kwargs


class TestOpenAICompatibleBackendUsageAndCost:
    def test_usage_tokens_extracted(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = _make_response("ok", prompt_tokens=120, completion_tokens=34)
        with _patch_create(backend, return_value=_raw(resp)):
            out = backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert out.input_tokens == 120
        assert out.output_tokens == 34

    def test_litellm_cost_header_read(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("ok"), cost="0.0234")):
            out = backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert out.cost_usd == 0.0234

    def test_cost_none_when_header_absent(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("ok"))):  # no cost header
            out = backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert out.cost_usd is None

    def test_unparseable_cost_header_is_none(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("ok"), cost="not-a-number")):
            out = backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert out.cost_usd is None

    def test_missing_usage_yields_zero_tokens(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = _make_response("ok")
        resp.usage = None  # some proxies omit usage entirely
        with _patch_create(backend, return_value=_raw(resp)):
            out = backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert out.input_tokens == 0 and out.output_tokens == 0
        assert out.text == "ok"


class TestOpenAICompatibleBackendErrorMapping:
    def test_timeout_becomes_runtimeerror(self):
        import openai
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, side_effect=openai.APITimeoutError(request=MagicMock())):
            with pytest.raises(RuntimeError, match="timed out after 60s"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_apierror_becomes_runtimeerror(self):
        import openai
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        err = openai.APIError("boom", request=MagicMock(), body=None)
        with _patch_create(backend, side_effect=err):
            with pytest.raises(RuntimeError, match="AI backend error"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_connection_error_becomes_runtimeerror(self):
        import openai
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, side_effect=openai.APIConnectionError(request=MagicMock())):
            with pytest.raises(RuntimeError, match="AI backend error"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )


class TestOpenAICompatibleBackendEmptyResponse:
    def test_empty_content_raises_runtimeerror_with_finish_reason(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("", finish_reason="length"))):
            with pytest.raises(RuntimeError, match="finish_reason=length"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_none_content_raises_runtimeerror(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = _make_response("x", finish_reason="stop")
        resp.choices[0].message.content = None
        with _patch_create(backend, return_value=_raw(resp)):
            with pytest.raises(RuntimeError, match="empty response"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_no_choices_raises_runtimeerror(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = MagicMock()
        resp.choices = []
        with _patch_create(backend, return_value=_raw(resp)):
            with pytest.raises(RuntimeError, match="no choices"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_multimodal_list_content_concatenates_text_parts(self):
        """Newer multimodal / reasoning-block responses can return
        ``content`` as a list of parts (each with ``type`` + payload).
        Concatenate the ``text``-typed parts; ignore other types."""
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = _make_response("placeholder", finish_reason="stop")
        part_text_a = MagicMock(type="text", text="first ")
        part_thinking = MagicMock(type="thinking", text="hidden")
        part_text_b = MagicMock(type="text", text="and second")
        resp.choices[0].message.content = [part_text_a, part_thinking, part_text_b]
        with _patch_create(backend, return_value=_raw(resp)):
            out = backend.complete(
                "p", model="m", effort="max", timeout=60, purpose="review",
            )
        assert out.text == "first and second"

    def test_unsupported_content_type_raises_empty_response(self):
        """If ``content`` is an unexpected type (e.g. dict, int — a proxy
        misbehaving), fall through to the empty-response error rather
        than crashing with AttributeError."""
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        resp = _make_response("placeholder", finish_reason="stop")
        resp.choices[0].message.content = {"unexpected": "dict"}
        with _patch_create(backend, return_value=_raw(resp)):
            with pytest.raises(RuntimeError, match="empty response"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )


class TestOpenAICompatibleBackendShutdown:
    def test_shutdown_closes_client(self):
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with patch.object(backend._client, "close") as mock_close:
            result = backend.shutdown()
        mock_close.assert_called_once()
        # shutdown() returns the count of in-flight requests signalled;
        # with no active requests, that's 0.
        assert result == 0

    def test_shutdown_with_inflight_request_returns_count(self):
        """When in-flight requests don't drain within grace_period,
        shutdown() returns the count and proceeds to close anyway."""
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        # Simulate two in-flight tickets
        backend._inflight.add(1)
        backend._inflight.add(2)
        with patch.object(backend._client, "close"):
            # Tiny grace_period: polling exits immediately, count survives.
            result = backend.shutdown(grace_period=0.0)
        assert result == 2

    def test_shutdown_polls_inflight_until_grace_period_elapses(self):
        """grace_period actually drives a poll loop on _inflight: if the
        set drains naturally during the grace window, shutdown() returns
        zero. This is the parameter's whole purpose."""
        import threading
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        backend._inflight.add(1)

        # Drain the in-flight set on a background thread shortly after
        # shutdown() begins polling.
        def drain():
            import time as _t
            _t.sleep(0.1)
            with backend._inflight_lock:
                backend._inflight.discard(1)

        t = threading.Thread(target=drain)
        with patch.object(backend._client, "close"):
            t.start()
            result = backend.shutdown(grace_period=2.0)
            t.join()
        assert result == 0


class TestOpenAICompatibleBackendInit:
    def test_init_requires_api_base(self, monkeypatch):
        monkeypatch.delenv("RAVEN_AI_API_BASE", raising=False)
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")
        from raven.ai.openai_compatible import OpenAICompatibleBackend
        with pytest.raises(RuntimeError, match="RAVEN_AI_API_BASE"):
            OpenAICompatibleBackend()

    def test_init_requires_api_key(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000")
        monkeypatch.delenv("RAVEN_AI_API_KEY", raising=False)
        from raven.ai.openai_compatible import OpenAICompatibleBackend
        with pytest.raises(RuntimeError, match="RAVEN_AI_API_KEY"):
            OpenAICompatibleBackend()

    def test_init_strips_trailing_slash_from_base(self, monkeypatch):
        """A trailing slash in RAVEN_AI_API_BASE is stripped before the
        openai.OpenAI client is constructed."""
        monkeypatch.setenv("RAVEN_AI_API_BASE", "http://proxy.example:4000/")
        monkeypatch.setenv("RAVEN_AI_API_KEY", "sk-test")
        with patch("raven.ai.openai_compatible.openai.OpenAI") as mock_cls:
            from raven.ai.openai_compatible import OpenAICompatibleBackend
            OpenAICompatibleBackend()
        _, kwargs = mock_cls.call_args
        assert kwargs["base_url"] == "http://proxy.example:4000"


class TestOpenAICompatibleBackendMaxTokens:
    def test_max_tokens_omitted_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("RAVEN_AI_MAX_TOKENS", raising=False)
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("x"))) as mock_create:
            backend.complete(
                "p", model="m", effort="max", timeout=60, purpose="review",
            )
        _, kwargs = mock_create.call_args
        assert "max_tokens" not in kwargs

    def test_max_tokens_passed_when_env_set(self, monkeypatch):
        monkeypatch.setenv("RAVEN_AI_MAX_TOKENS", "32768")
        from raven.ai.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()
        with _patch_create(backend, return_value=_raw(_make_response("x"))) as mock_create:
            backend.complete(
                "p", model="m", effort="max", timeout=60, purpose="review",
            )
        _, kwargs = mock_create.call_args
        assert kwargs["max_tokens"] == 32768
