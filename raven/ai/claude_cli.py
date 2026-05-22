"""ClaudeCLIBackend internals — subprocess wrapper for the Claude Code CLI.

This module owns everything needed to talk to the /usr/bin/claude binary:
the subprocess-tracking globals, semaphore, and the Popen helper. The
Backend class itself is added in Task 3; Task 2 is a pure lift of the
existing code from reviewer.py so the re-exports let existing tests
keep passing during the refactor.
"""

import contextlib
import logging
import os
import re
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


# Token shapes we redact from CLI stdout/stderr before logging. The
# Claude CLI shouldn't echo ``CLAUDE_CODE_OAUTH_TOKEN`` itself, but a
# misconfigured CLI or an auth-failure message can include the token,
# an Anthropic-style ``sk-ant-...`` key, or a ``Bearer ...`` header from
# an HTTP error body. Each pattern matches a credential-looking prefix
# and replaces the secret portion with ``***REDACTED***`` so operators
# can still see the surrounding error context without leaking auth.
_TOKEN_PATTERNS = [
    re.compile(r"(sk-ant-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+"),
    re.compile(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]{16,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{16,}", re.IGNORECASE),
]


def _redact_secrets(text: str) -> str:
    """Strip credential-shaped substrings from log payloads.

    Belt-and-braces: also rewrites the literal configured
    ``CLAUDE_CODE_OAUTH_TOKEN`` value if it appears in the text, since
    that's the most targeted possible match (no false positives) and
    catches the case where the CLI echoes the env var verbatim. Pattern
    matches stay in place for unknown / future token shapes.
    """
    if not text:
        return text
    configured = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or ""
    if configured and len(configured) >= 8:
        text = text.replace(configured, "***REDACTED***")
    for pat in _TOKEN_PATTERNS:
        text = pat.sub(r"\1***REDACTED***", text)
    return text

CLAUDE_BIN = "/usr/bin/claude"
RAVEN_AI_MAX_CONCURRENT = max(int(os.environ.get("RAVEN_AI_MAX_CONCURRENT", "4")), 1)
_claude_semaphore = threading.Semaphore(RAVEN_AI_MAX_CONCURRENT)

# Track in-flight Claude CLI subprocesses so shutdown hooks can terminate
# them. Each active Popen is added to _active_procs for the lifetime of
# the call and removed in the finally block. terminate_active_processes()
# sends SIGTERM to each, escalating to SIGKILL after a grace period — this
# prevents gunicorn's graceful timeout from being consumed by LLM inference
# whose result will be discarded on exit.
_active_procs: set[subprocess.Popen] = set()
_active_procs_lock = threading.Lock()

# Restrict the Claude CLI to plain text generation. Raven feeds the CLI
# user-controlled content (diff text, PR comments, conversation history) —
# without an explicit tool allowlist, a prompt-injection could coerce the
# model into running tools with access to credentials and the network.
# Passing an empty --allowed-tools denies every tool; if a future CLI
# version changes the semantics, the output will surface the divergence
# in the no-tools integration test rather than silently regress.
CLAUDE_BASE_ARGS = [
    CLAUDE_BIN, "-p",
    "--output-format", "text",
    "--allowed-tools", "",
]


def _run_claude_cli(args: list[str], *, prompt: str, timeout: int) -> subprocess.CompletedProcess:
    """Spawn the Claude CLI with a tracked Popen handle.

    Equivalent to ``subprocess.run`` for our usage (stdin=prompt,
    captured stdout/stderr, text mode) but goes through Popen explicitly
    so the child is registered in ``_active_procs`` for the duration of
    the call. ``terminate_active_processes`` can then SIGTERM every
    in-flight subprocess on shutdown.

    The Popen is used as a context manager so ``__exit__`` closes the
    stdin/stdout/stderr pipes and waits on the child even when an
    exception unwinds the block — matching ``subprocess.run``'s
    cleanup semantics. The bare ``except BaseException`` clause
    mirrors ``subprocess.run``'s own handling: on any non-timeout
    error (``BrokenPipeError`` if the child exits mid-write, ``OSError``
    on I/O failure, ``KeyboardInterrupt``), the child is killed so the
    context-manager exit doesn't hang waiting for it.

    Raises ``subprocess.TimeoutExpired`` and ``FileNotFoundError`` with
    the same semantics as ``subprocess.run`` so callers can catch them.
    """
    with subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        with _active_procs_lock:
            _active_procs.add(proc)
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        except BaseException:
            # Any other exception (BrokenPipeError, OSError, KeyboardInterrupt):
            # kill the child so the with-exit wait() doesn't block.
            # subprocess.run uses the same bare-except pattern.
            proc.kill()
            raise
        finally:
            with _active_procs_lock:
                _active_procs.discard(proc)
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def terminate_active_processes(grace_period: float = 2.0) -> int:
    """SIGTERM every in-flight Claude subprocess, SIGKILL survivors.

    Called from the server's shutdown hook. Running reviews that survive
    ``cancel_futures`` are almost always blocked in a Claude CLI call
    (the diff fetch is short); killing those subprocesses unblocks the
    worker threads so gunicorn's graceful shutdown can finish instead of
    waiting the full ``RAVEN_AI_TIMEOUT``.

    Returns the number of processes that were signalled.
    """
    with _active_procs_lock:
        procs = list(_active_procs)
    if not procs:
        return 0
    logger.info("Terminating %d in-flight Claude CLI subprocess(es)", len(procs))
    for p in procs:
        with contextlib.suppress(Exception):
            p.terminate()
    deadline = time.monotonic() + grace_period
    for p in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
            continue  # exited cleanly within the grace period
        except subprocess.TimeoutExpired:
            pass  # still alive — escalate below
        except Exception:
            # Already reaped, OSError on racily-closed pipe, etc.
            # Nothing useful a kill can do; move on to the next proc
            # instead of aborting the whole loop.
            continue
        with contextlib.suppress(Exception):
            p.kill()
            p.wait(timeout=1)
    return len(procs)


from raven.ai.base import AIBackend


class ClaudeCLIBackend(AIBackend):
    """AIBackend that shells out to the Claude Code CLI (/usr/bin/claude).

    Reads auth from the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (the CLI itself
    picks it up from the environment; we don't pass it through explicitly).
    Raises RuntimeError for transport/timeout/auth failures so the reviewer
    can stay backend-agnostic in its exception handling.
    """

    name = "claude_cli"

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        effort: str,
        timeout: int,
        purpose: str,
    ) -> str:
        logger.info(
            "Claude CLI %s (model=%s effort=%s prompt_chars=%d)",
            purpose, model, effort, len(prompt),
        )
        args = CLAUDE_BASE_ARGS + ["--model", model, "--effort", effort]
        with _claude_semaphore:
            try:
                result = _run_claude_cli(args, prompt=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"claude CLI timed out after {timeout}s")
            except FileNotFoundError:
                raise RuntimeError(f"claude CLI not found at {CLAUDE_BIN}")

        if result.returncode != 0:
            # Redact token-shaped substrings before logging — the CLI
            # shouldn't echo CLAUDE_CODE_OAUTH_TOKEN but a misconfigured
            # auth-failure path can. ``detail`` is exception-message-bound
            # and likely flows into operator log aggregation; redact it
            # for the same reason.
            stderr_safe = _redact_secrets(result.stderr[:500])
            stdout_safe = _redact_secrets(result.stdout[:500])
            logger.error("claude CLI stderr: %s", stderr_safe)
            logger.error("claude CLI stdout: %s", stdout_safe)
            detail = _redact_secrets(result.stderr[:200] or result.stdout[:200])
            raise RuntimeError(f"claude CLI exited with code {result.returncode}: {detail}")
        return result.stdout

    def shutdown(self, grace_period: float = 2.0) -> int:
        return terminate_active_processes(grace_period)
