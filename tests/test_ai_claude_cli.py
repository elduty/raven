"""Tests for raven.ai.claude_cli — subprocess lifecycle and shutdown."""

import os
import subprocess

import pytest
from unittest.mock import MagicMock, patch


from raven.ai.base import AIBackend, AIError
from raven.ai.claude_cli import (
    _active_procs,
    _active_procs_lock,
    _redact_secrets,
    _run_claude_cli,
    terminate_active_processes,
    ClaudeCLIBackend,
)


# ------------------------------------------------------------------ #
#  Claude subprocess tracking and termination                         #
# ------------------------------------------------------------------ #

class TestActiveProcessTracking:
    """The shutdown hook terminates in-flight Claude subprocesses so
    gunicorn's graceful timeout isn't consumed by LLM inference whose
    result will be discarded. That requires each Popen to be tracked in
    ``_active_procs`` for the lifetime of the call."""

    @staticmethod
    def _cm_mock(**attrs) -> MagicMock:
        """MagicMock configured as a context manager that yields itself —
        so code using ``with Popen(...) as proc:`` sees the same object
        it would otherwise get from ``Popen(...)``."""
        m = MagicMock(**attrs)
        m.__enter__.return_value = m
        m.__exit__.return_value = None
        return m

    def test_run_claude_cli_adds_and_removes_from_active_procs(self):
        from raven.ai.claude_cli import _run_claude_cli

        fake_proc = self._cm_mock(returncode=0)
        captured = {}

        def fake_communicate(input, timeout):
            # While communicate is running, the proc must be registered
            captured["in_set_during_call"] = fake_proc in _active_procs
            return ("stdout", "stderr")

        fake_proc.communicate.side_effect = fake_communicate

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            _run_claude_cli(["claude"], prompt="hi", timeout=10)

        assert captured["in_set_during_call"] is True
        assert fake_proc not in _active_procs

    def test_run_claude_cli_removes_on_timeout(self):
        from raven.ai.claude_cli import _run_claude_cli
        import subprocess as sp

        fake_proc = self._cm_mock()
        fake_proc.communicate.side_effect = sp.TimeoutExpired(cmd="claude", timeout=1)

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            with pytest.raises(sp.TimeoutExpired):
                _run_claude_cli(["claude"], prompt="hi", timeout=1)

        fake_proc.kill.assert_called_once()
        assert fake_proc not in _active_procs

    def test_run_claude_cli_kills_and_removes_on_non_timeout_exception(self):
        """Regression guard: if proc.communicate raises anything other than
        TimeoutExpired (BrokenPipeError, OSError, etc.) the subprocess must
        still be killed and removed from _active_procs — otherwise a live
        child plus three pipe FDs leak until interpreter exit, which is
        exactly what this module is meant to prevent."""
        from raven.ai.claude_cli import _run_claude_cli

        fake_proc = self._cm_mock()
        fake_proc.communicate.side_effect = BrokenPipeError("pipe gone")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            with pytest.raises(BrokenPipeError):
                _run_claude_cli(["claude"], prompt="hi", timeout=10)

        fake_proc.kill.assert_called_once()
        assert fake_proc not in _active_procs

    def test_run_claude_cli_uses_popen_as_context_manager(self):
        """Popen is entered as a context manager so __exit__ closes the
        stdin/stdout/stderr pipes and waits on the child even on the
        success path — matching subprocess.run's cleanup semantics."""
        from raven.ai.claude_cli import _run_claude_cli

        fake_proc = self._cm_mock(returncode=0)
        fake_proc.communicate.return_value = ("ok", "")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            _run_claude_cli(["claude"], prompt="hi", timeout=10)

        fake_proc.__enter__.assert_called_once()
        fake_proc.__exit__.assert_called_once()

    def test_terminate_active_processes_empty_set_returns_zero(self):
        import raven.ai.claude_cli as cli
        # Guard against leakage from other tests in the same module
        with patch.object(cli, "_active_procs", set()):
            assert terminate_active_processes() == 0

    def test_terminate_active_processes_sends_sigterm(self):
        import raven.ai.claude_cli as cli

        proc = MagicMock()
        # wait() returns cleanly — no escalation needed
        proc.wait.return_value = 0

        with patch.object(cli, "_active_procs", {proc}):
            count = terminate_active_processes(grace_period=0.1)

        assert count == 1
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    def test_terminate_active_processes_escalates_to_sigkill(self):
        """A process that ignores SIGTERM must be SIGKILLed after the
        grace period. Simulate by having wait() raise TimeoutExpired
        on the first call (the grace-period wait) and succeed on the
        second (the post-kill wait)."""
        import raven.ai.claude_cli as cli
        import subprocess as sp

        proc = MagicMock()
        proc.wait.side_effect = [
            sp.TimeoutExpired(cmd="claude", timeout=0.1),
            0,
        ]

        with patch.object(cli, "_active_procs", {proc}):
            count = terminate_active_processes(grace_period=0.01)

        assert count == 1
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_terminate_active_processes_survives_terminate_exception(self):
        """If one proc's terminate() raises (e.g. already reaped), we
        still process the remaining procs instead of bailing out."""
        import raven.ai.claude_cli as cli

        dead = MagicMock()
        dead.terminate.side_effect = OSError("No such process")
        dead.wait.return_value = 0

        live = MagicMock()
        live.wait.return_value = 0

        with patch.object(cli, "_active_procs", {dead, live}):
            count = terminate_active_processes(grace_period=0.1)

        assert count == 2
        live.terminate.assert_called_once()

    def test_terminate_active_processes_continues_on_non_timeout_wait_error(self):
        """Regression guard: if ``p.wait(timeout=remaining)`` raises
        anything other than TimeoutExpired (OSError on a racily-reaped
        child is the realistic case), the loop must continue to the
        next proc instead of aborting — otherwise later procs never
        get SIGKILLed, which is exactly what terminate is meant to
        guarantee."""
        import raven.ai.claude_cli as cli

        # First proc's wait blows up with OSError (already reaped).
        reaped = MagicMock()
        reaped.wait.side_effect = OSError("No child processes")

        # Second proc must still be escalated to SIGKILL after grace.
        import subprocess as sp
        stuck = MagicMock()
        stuck.wait.side_effect = [
            sp.TimeoutExpired(cmd="claude", timeout=0.01),  # grace wait
            0,  # post-kill wait
        ]

        # Use a list (ordered) so "reaped" is processed before "stuck"
        # and we can assert the loop didn't abort after the OSError.
        with patch.object(cli, "_active_procs", [reaped, stuck]):
            count = terminate_active_processes(grace_period=0.01)

        assert count == 2
        # The non-timeout error did NOT stop the loop — stuck still got SIGKILLed.
        stuck.kill.assert_called_once()


class TestClaudeCLIBackend:
    def test_is_aibackend_subclass(self):
        assert issubclass(ClaudeCLIBackend, AIBackend)

    def test_name_attribute(self):
        assert ClaudeCLIBackend().name == "claude_cli"

    def test_complete_invokes_claude_cli_with_model_and_effort(self):
        """complete() calls _run_claude_cli with --model and --effort flags
        derived from its kwargs."""
        backend = ClaudeCLIBackend()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "some model output"
        fake_result.stderr = ""

        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result) as mock_run:
            output = backend.complete(
                "the prompt",
                model="claude-opus-4-7",
                effort="max",
                timeout=600,
                purpose="review",
            )
        # Non-json stdout → graceful fallback: text is the raw stdout.
        assert output.text == "some model output"
        args, kwargs = mock_run.call_args
        cli_args = args[0]
        assert "--model" in cli_args and "claude-opus-4-7" in cli_args
        assert "--effort" in cli_args and "max" in cli_args
        assert kwargs["prompt"] == "the prompt"
        assert kwargs["timeout"] == 600


class TestClaudeCLIJsonEnvelope:
    """complete() parses the --output-format json envelope for text + usage
    + cost, and degrades gracefully when the envelope is unexpected."""

    def _run(self, stdout: str):
        import json
        backend = ClaudeCLIBackend()
        fake = MagicMock(returncode=0, stdout=stdout, stderr="")
        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake):
            return backend.complete("p", model="m", effort="max", timeout=60, purpose="review")

    def test_parses_text_tokens_and_cost(self):
        import json
        envelope = json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "result": "the review text",
            "total_cost_usd": 0.0893745,
            "usage": {"input_tokens": 5264, "output_tokens": 4,
                      "cache_read_input_tokens": 17484},
        })
        out = self._run(envelope)
        assert out.text == "the review text"
        assert out.input_tokens == 5264
        assert out.output_tokens == 4
        assert out.cost_usd == 0.0893745

    def test_malformed_json_falls_back_to_raw_text(self):
        out = self._run("not json at all")
        assert out.text == "not json at all"
        assert out.input_tokens == 0 and out.output_tokens == 0
        assert out.cost_usd is None

    def test_json_without_result_key_falls_back(self):
        import json
        out = self._run(json.dumps({"unexpected": "shape"}))
        assert out.text == json.dumps({"unexpected": "shape"})
        assert out.cost_usd is None

    def test_missing_usage_and_cost_yields_zero_none(self):
        import json
        out = self._run(json.dumps({"result": "text only"}))
        assert out.text == "text only"
        assert out.input_tokens == 0 and out.output_tokens == 0
        assert out.cost_usd is None

    def test_output_format_json_flag_present(self):
        backend = ClaudeCLIBackend()
        fake = MagicMock(returncode=0, stdout='{"result":"x"}', stderr="")
        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake) as mock_run:
            backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        cli_args = mock_run.call_args[0][0]
        idx = cli_args.index("--output-format")
        assert cli_args[idx + 1] == "json"
        # injection guard still present
        assert "--allowed-tools" in cli_args
        assert cli_args[cli_args.index("--allowed-tools") + 1] == ""

    def test_complete_raises_runtimeerror_on_nonzero_exit(self):
        backend = ClaudeCLIBackend()
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "claude: auth failed"
        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )

    def test_complete_raises_runtimeerror_on_timeout(self):
        backend = ClaudeCLIBackend()
        with patch(
            "raven.ai.claude_cli._run_claude_cli",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                backend.complete(
                    "p", model="m", effort="max", timeout=5, purpose="review",
                )

    def test_complete_raises_runtimeerror_on_missing_binary(self):
        backend = ClaudeCLIBackend()
        with patch("raven.ai.claude_cli._run_claude_cli", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="not found"):
                backend.complete(
                    "p", model="m", effort="max", timeout=60, purpose="review",
                )


class TestClaudeCLIErrorClassification:
    """The CLI collapses every failure mode to a non-zero exit + stderr
    text (or a TimeoutExpired). classify the cause into an ``AIError.reason``
    so server.py can post an actionable comment and reviewer.py can decide
    whether to retry. Reasons come from the stderr/stdout text the CLI
    emits — usage caps, rate limits, and auth failures each have a
    recognisable shape."""

    def _fail(self, *, returncode=1, stderr="", stdout=""):
        backend = ClaudeCLIBackend()
        fake = MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)
        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake):
            with pytest.raises(AIError) as ei:
                backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        return ei.value

    def test_timeout_classified(self):
        backend = ClaudeCLIBackend()
        with patch(
            "raven.ai.claude_cli._run_claude_cli",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5),
        ):
            with pytest.raises(AIError) as ei:
                backend.complete("p", model="m", effort="max", timeout=5, purpose="review")
        assert ei.value.reason == "timeout"
        assert ei.value.retryable is True

    def test_usage_limit_classified_from_stderr(self):
        # Claude CLI emits a usage/session cap message when the plan's
        # limit is hit — resets hours later, so an in-process retry can't help.
        err = self._fail(stderr="Claude AI usage limit reached. Your limit will reset at 5pm.")
        assert err.reason == "usage_limit"
        assert err.retryable is False

    def test_session_limit_classified_from_stderr(self):
        err = self._fail(stderr="5-hour session limit reached; try again later")
        assert err.reason == "usage_limit"
        assert err.retryable is False

    def test_rate_limit_classified_from_stderr(self):
        err = self._fail(stderr="API Error: 429 rate_limit_error: too many requests")
        assert err.reason == "rate_limit"
        assert err.retryable is True

    def test_auth_failure_classified_from_stderr(self):
        err = self._fail(stderr="API Error: 401 authentication_error: invalid x-api-key")
        assert err.reason == "auth"
        assert err.retryable is False

    def test_server_error_classified_from_stderr(self):
        err = self._fail(stderr="API Error: 529 overloaded_error: upstream overloaded")
        assert err.reason == "backend_5xx"
        assert err.retryable is True

    def test_unrecognised_nonzero_exit_is_unknown(self):
        err = self._fail(returncode=2, stderr="some unexpected failure")
        assert err.reason == "unknown"
        assert err.retryable is False

    def test_missing_binary_classified_as_unknown(self):
        backend = ClaudeCLIBackend()
        with patch("raven.ai.claude_cli._run_claude_cli", side_effect=FileNotFoundError):
            with pytest.raises(AIError) as ei:
                backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        # A missing binary is a deploy fault, not a transient one — don't retry.
        assert ei.value.reason == "unknown"
        assert ei.value.retryable is False

    def test_classification_does_not_leak_token_into_message(self, monkeypatch):
        # The exception message must stay redacted — it flows to operator
        # logs and (via server.py) the comment-building helper.
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-supersecrettoken1234567890")
        err = self._fail(stderr="401 auth failed for sk-ant-supersecrettoken1234567890")
        assert "supersecrettoken" not in str(err)
        assert err.reason == "auth"

    def test_shutdown_delegates_to_terminate_active_processes(self):
        backend = ClaudeCLIBackend()
        with patch(
            "raven.ai.claude_cli.terminate_active_processes", return_value=3,
        ) as mock_term:
            n = backend.shutdown(grace_period=1.5)
        assert n == 3
        mock_term.assert_called_once_with(1.5)

    def test_complete_passes_prompt_via_stdin_not_argv(self):
        """Security: the prompt must be piped via stdin (-p flag), not embedded
        in argv, to stay under ARG_MAX and out of the process table (`ps`)."""
        backend = ClaudeCLIBackend()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "output"
        fake_result.stderr = ""

        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result) as mock_run:
            backend.complete(
                "sensitive prompt content",
                model="claude-opus-4-7",
                effort="max",
                timeout=600,
                purpose="review",
            )
        cli_args = mock_run.call_args[0][0]
        assert "-p" in cli_args
        # The prompt itself must NOT appear in argv.
        assert "sensitive prompt content" not in cli_args
        # The prompt is passed via kwargs to _run_claude_cli (which pipes it
        # to subprocess stdin).
        assert mock_run.call_args.kwargs["prompt"] == "sensitive prompt content"

    def test_complete_disables_tools_for_prompt_injection_defense(self):
        """Security: --allowed-tools must be empty-string so the CLI refuses
        every tool. Without this gate, a prompt-injection attack in a diff
        could coerce the model into invoking tools with access to credentials
        and the network."""
        backend = ClaudeCLIBackend()
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "output"
        fake_result.stderr = ""

        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result) as mock_run:
            backend.complete(
                "p", model="m", effort="max", timeout=60, purpose="review",
            )
        cli_args = mock_run.call_args[0][0]
        # The --allowed-tools flag must be present, followed by exactly the
        # empty string. Any other value would permit at least one tool.
        assert "--allowed-tools" in cli_args
        tools_idx = cli_args.index("--allowed-tools")
        assert cli_args[tools_idx + 1] == ""


# ------------------------------------------------------------------ #
#  Working-directory isolation                                         #
# ------------------------------------------------------------------ #

class TestCwdIsolation:
    """The CLI subprocess must NOT inherit the service's working directory.

    The claude CLI ships file tools (grep/read) that operate relative to
    its cwd. Inheriting the app's cwd let reviews "verify" claims against
    the deployed container's own /app checkout — observed twice producing
    confident false "implementation absent" findings (PR #160 round-5,
    PR #163): greps of a checkout that is stale relative to the reviewed
    repo's main, and for any repo other than Raven itself an entirely
    unrelated codebase. Each spawn gets a fresh empty temp dir so the
    model's only evidence is what the prompt provides."""

    @staticmethod
    def _cm_mock(**attrs) -> MagicMock:
        m = MagicMock(**attrs)
        m.__enter__.return_value = m
        m.__exit__.return_value = None
        return m

    def test_popen_receives_isolated_empty_cwd(self):
        """Popen must be called with a cwd kwarg pointing at a directory
        that exists, is empty, and is not the service's own cwd."""
        fake_proc = self._cm_mock(returncode=0)
        captured = {}

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            def fake_communicate(input, timeout):
                cwd = mock_popen.call_args.kwargs.get("cwd")
                captured["cwd"] = cwd
                captured["exists_during_call"] = cwd is not None and os.path.isdir(cwd)
                captured["empty_during_call"] = (
                    cwd is not None and os.path.isdir(cwd) and os.listdir(cwd) == []
                )
                return ("stdout", "stderr")

            fake_proc.communicate.side_effect = fake_communicate
            _run_claude_cli(["claude"], prompt="hi", timeout=10)

        assert captured["cwd"] is not None, "Popen was not given an isolated cwd"
        assert captured["cwd"] != os.getcwd()
        assert captured["exists_during_call"] is True
        assert captured["empty_during_call"] is True

    def test_isolated_cwd_removed_after_success(self):
        fake_proc = self._cm_mock(returncode=0)
        fake_proc.communicate.return_value = ("stdout", "stderr")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            _run_claude_cli(["claude"], prompt="hi", timeout=10)
            cwd = mock_popen.call_args.kwargs.get("cwd")

        assert cwd is not None
        assert not os.path.exists(cwd), "isolated cwd leaked after a successful call"

    def test_isolated_cwd_removed_on_timeout(self):
        import subprocess as sp

        fake_proc = self._cm_mock()
        fake_proc.communicate.side_effect = sp.TimeoutExpired(cmd="claude", timeout=1)

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            with pytest.raises(sp.TimeoutExpired):
                _run_claude_cli(["claude"], prompt="hi", timeout=1)
            cwd = mock_popen.call_args.kwargs.get("cwd")

        assert cwd is not None
        assert not os.path.exists(cwd), "isolated cwd leaked after a timeout"

    def test_isolated_cwd_removed_on_non_timeout_exception(self):
        fake_proc = self._cm_mock()
        fake_proc.communicate.side_effect = BrokenPipeError("pipe gone")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            with pytest.raises(BrokenPipeError):
                _run_claude_cli(["claude"], prompt="hi", timeout=10)
            cwd = mock_popen.call_args.kwargs.get("cwd")

        assert cwd is not None
        assert not os.path.exists(cwd), "isolated cwd leaked after an exception"

    def test_isolated_cwd_removed_even_if_cli_wrote_files(self):
        """If the CLI drops session/debug files into its cwd, cleanup must
        still remove the directory (recursive delete, not bare rmdir)."""
        fake_proc = self._cm_mock(returncode=0)

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            def fake_communicate(input, timeout):
                cwd = mock_popen.call_args.kwargs["cwd"]
                with open(os.path.join(cwd, "session.json"), "w") as f:
                    f.write("{}")
                return ("stdout", "stderr")

            fake_proc.communicate.side_effect = fake_communicate
            _run_claude_cli(["claude"], prompt="hi", timeout=10)
            cwd = mock_popen.call_args.kwargs["cwd"]

        assert not os.path.exists(cwd)

    def test_popen_env_is_scoped_and_keeps_cli_auth(self, monkeypatch):
        """The child's env is now scoped to a fail-closed allowlist (see
        TestSubprocessEnvAllowlist) rather than a full passthrough — but the
        CLI's own auth must still reach it, so CLAUDE_CODE_OAUTH_TOKEN
        survives the scoping and the env override is present (not None)."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-token")
        fake_proc = self._cm_mock(returncode=0)
        fake_proc.communicate.return_value = ("stdout", "stderr")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            _run_claude_cli(["claude"], prompt="hi", timeout=10)

        env = mock_popen.call_args.kwargs.get("env")
        assert env is not None  # env IS scoped now (no longer full passthrough)
        assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "claude-token"

    def test_complete_disables_builtin_tools_entirely(self):
        """Defense-in-depth: ``--tools ""`` removes every built-in tool from
        the model's set. ``--allowed-tools`` alone only governs permission
        auto-approval — in -p mode read-only file tools (grep/read) still
        ran, which is how the false "implementation absent" findings were
        produced. The empty cwd is the primary isolation; this flag makes
        the tools not exist at all."""
        backend = ClaudeCLIBackend()
        fake_result = MagicMock(returncode=0, stdout="output", stderr="")

        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result) as mock_run:
            backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        cli_args = mock_run.call_args[0][0]
        assert "--tools" in cli_args
        assert cli_args[cli_args.index("--tools") + 1] == ""

    def test_each_call_gets_a_fresh_cwd(self):
        """Per-call isolation: two spawns must not share a directory, so
        nothing one call leaves behind is visible to the next."""
        fake_proc = self._cm_mock(returncode=0)
        fake_proc.communicate.return_value = ("stdout", "stderr")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            _run_claude_cli(["claude"], prompt="hi", timeout=10)
            first = mock_popen.call_args.kwargs.get("cwd")
            _run_claude_cli(["claude"], prompt="hi", timeout=10)
            second = mock_popen.call_args.kwargs.get("cwd")

        assert first is not None and second is not None
        assert first != second


# ------------------------------------------------------------------ #
#  Token redaction in error-path logs                                  #
# ------------------------------------------------------------------ #

class TestRedactSecrets:
    """Guards against credential leakage in CLI stderr/stdout logs.
    The CLI shouldn't echo CLAUDE_CODE_OAUTH_TOKEN, but a misconfigured
    auth or transport-error message can; redaction is belt-and-braces."""

    def test_redacts_configured_oauth_token_literally(self, monkeypatch):
        """The most targeted match: if the env-var value appears verbatim,
        replace it. Zero false positives."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-abc123superSecret-xyz")
        msg = "auth failed: token sk-ant-oat-abc123superSecret-xyz invalid"
        assert "abc123superSecret" not in _redact_secrets(msg)
        assert "***REDACTED***" in _redact_secrets(msg)

    def test_redacts_anthropic_sk_ant_pattern(self):
        """Unknown sk-ant-... key (not the configured env var) still gets
        redacted via pattern, keeping a 4-char prefix as a hint."""
        msg = "request failed with key sk-ant-1234567890abcdefghij"
        out = _redact_secrets(msg)
        assert "sk-ant-1234" in out  # prefix retained
        assert "567890abcdefghij" not in out
        assert "***REDACTED***" in out

    def test_redacts_generic_sk_pattern(self):
        msg = "Authorization: sk-proj-abcdefgh1234567890ijklmnop"
        out = _redact_secrets(msg)
        assert "abcdefgh1234567890ijklmnop" not in out

    def test_redacts_bearer_token(self):
        msg = "401 Unauthorized: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ"
        out = _redact_secrets(msg)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in out
        assert "Bearer " in out  # prefix retained
        assert "***REDACTED***" in out

    def test_short_token_env_var_not_substituted(self, monkeypatch):
        """Don't replace tiny env-var values to avoid false-positive
        substring matches (e.g. a 3-char token would match common words)."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "abc")
        msg = "abc def ghi — error about abc"
        # The 3-char token is under the 8-char threshold; left untouched.
        assert _redact_secrets(msg) == msg

    def test_empty_or_none_passes_through(self):
        assert _redact_secrets("") == ""
        assert _redact_secrets(None) is None  # type: ignore[arg-type]

    def test_normal_error_messages_unchanged(self):
        """No false positives on benign content. Real CLI error text
        shouldn't get garbled by the redactor."""
        msg = "claude CLI: model 'claude-opus-4-7' not found"
        assert _redact_secrets(msg) == msg

    def test_complete_redacts_stderr_in_runtime_error(self, monkeypatch):
        """End-to-end: a non-zero CLI exit with a token in stderr
        produces a RuntimeError whose detail is redacted, not raw."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-supersecret-token-xyz")
        backend = ClaudeCLIBackend()
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "auth: sk-ant-oat-supersecret-token-xyz expired"
        fake_result.stdout = ""

        with patch("raven.ai.claude_cli._run_claude_cli", return_value=fake_result):
            with pytest.raises(RuntimeError) as excinfo:
                backend.complete("p", model="m", effort="max", timeout=60, purpose="review")
        assert "supersecret-token-xyz" not in str(excinfo.value)
        assert "***REDACTED***" in str(excinfo.value)


# ------------------------------------------------------------------ #
#  Subprocess env allowlist (audit 2026-06-13 finding #3)             #
# ------------------------------------------------------------------ #

class TestSubprocessEnvAllowlist:
    """The Claude CLI child must inherit only an allowlisted env subset —
    never Raven's platform credentials. The CLI self-updates nightly from
    npm, so a compromised release inheriting GITEA_TOKEN / BITBUCKET_DC_TOKEN
    / webhook secrets / API keys could exfiltrate org push+merge creds. The
    allowlist is fail-closed: a new secret added to Raven's env later does
    NOT leak unless explicitly permitted."""

    @staticmethod
    def _cm_mock(**attrs):
        m = MagicMock(**attrs)
        m.__enter__.return_value = m
        m.__exit__.return_value = None
        m.communicate.return_value = ("ok", "")
        return m

    def _capture_env(self):
        fake_proc = self._cm_mock(returncode=0)
        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc) as mock_popen:
            _run_claude_cli(["claude"], prompt="hi", timeout=10)
        env = mock_popen.call_args.kwargs.get("env")
        assert env is not None, "Popen called without a scoped env allowlist"
        return env

    def test_strips_raven_platform_secrets(self, monkeypatch):
        for k, v in {
            "GITEA_TOKEN": "git-secret",
            "BITBUCKET_DC_TOKEN": "bb-secret",
            "GITEA_WEBHOOK_SECRET": "wh-secret",
            "BITBUCKET_DC_WEBHOOK_SECRET": "bbwh-secret",
            "RAVEN_AI_API_KEY": "ai-secret",
            "RAVEN_METRICS_TOKEN": "metrics-secret",
        }.items():
            monkeypatch.setenv(k, v)
        env = self._capture_env()
        for k in ("GITEA_TOKEN", "BITBUCKET_DC_TOKEN", "GITEA_WEBHOOK_SECRET",
                  "BITBUCKET_DC_WEBHOOK_SECRET", "RAVEN_AI_API_KEY",
                  "RAVEN_METRICS_TOKEN"):
            assert k not in env, f"{k} leaked into the CLI subprocess env"

    def test_keeps_cli_auth_and_system_essentials(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-token")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_REFRESH_TOKEN", "refresh-token")
        monkeypatch.setenv("HOME", "/home/raven")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = self._capture_env()
        assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "claude-token"
        assert env.get("CLAUDE_CODE_OAUTH_REFRESH_TOKEN") == "refresh-token"
        assert env.get("HOME") == "/home/raven"
        assert env.get("PATH") == "/usr/bin:/bin"

    def test_keeps_proxy_tls_and_locale_for_networked_envs(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/etc/ssl/corp.pem")
        monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert.pem")
        monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
        env = self._capture_env()
        assert env.get("HTTPS_PROXY") == "http://proxy:8080"
        assert env.get("NODE_EXTRA_CA_CERTS") == "/etc/ssl/corp.pem"
        assert env.get("SSL_CERT_FILE") == "/etc/ssl/cert.pem"
        assert env.get("LC_ALL") == "en_US.UTF-8"
