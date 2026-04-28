"""Tests for raven.ai.claude_cli — subprocess lifecycle and shutdown."""

import os
import subprocess

import pytest
from unittest.mock import MagicMock, patch


from raven.ai.base import AIBackend
from raven.ai.claude_cli import (
    _active_procs,
    _active_procs_lock,
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
        assert output == "some model output"
        args, kwargs = mock_run.call_args
        cli_args = args[0]
        assert "--model" in cli_args and "claude-opus-4-7" in cli_args
        assert "--effort" in cli_args and "max" in cli_args
        assert kwargs["prompt"] == "the prompt"
        assert kwargs["timeout"] == 600

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
