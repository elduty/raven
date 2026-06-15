"""ClaudeCLIBackend internals — subprocess wrapper for the Claude Code CLI.

This module owns everything needed to talk to the /usr/bin/claude binary:
the subprocess-tracking globals, semaphore, and the Popen helper. The
Backend class itself is added in Task 3; Task 2 is a pure lift of the
existing code from reviewer.py so the re-exports let existing tests
keep passing during the refactor.
"""

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
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
# Two layers, because --allowed-tools "" alone proved insufficient: it
# governs permission auto-approval, but in -p mode read-only file tools
# (grep/read) run without prompting anyway — which is how reviews
# "verified" claims against the deployed container's own /app checkout
# (false "implementation absent" findings on PR #160 / PR #163).
#   --tools ""         — removes every built-in tool from the model's set
#                        (per CLI help: 'Use "" to disable all tools').
#   --allowed-tools "" — kept as the original auto-approval deny layer.
# The third layer (primary for the stale-checkout incidents) is cwd
# isolation in _run_claude_cli: even a tool that slips through both
# flags sees only an empty temp directory.
#
# --output-format json wraps the response in an envelope carrying the
# response text under ``.result`` plus ``.usage`` (input_tokens /
# output_tokens) and ``.total_cost_usd`` — parsed in complete() for token
# and cost metrics. The envelope shape was confirmed against the installed
# CLI (v2.1.156) rather than assumed.
CLAUDE_BASE_ARGS = [
    CLAUDE_BIN, "-p",
    "--output-format", "json",
    "--tools", "",
    "--allowed-tools", "",
]


# ── Subprocess env allowlist ──────────────────────────────────────────────
# The Claude CLI child inherits ONLY the env vars matched below — never
# Raven's platform credentials (GITEA_TOKEN, BITBUCKET_DC_TOKEN,
# *_WEBHOOK_SECRET, RAVEN_AI_API_KEY, RAVEN_METRICS_TOKEN, …). The CLI is a
# binary that self-updates nightly from npm, and Raven feeds it
# attacker-influenced prompt content; without this scoping a single
# compromised release (or a prompt-injection reaching a tool despite the
# tool-disabling flags) could read every secret needed to push to and merge
# across the org. The CLI's OWN auth (CLAUDE_CODE_*/ANTHROPIC_*) and the
# system / locale / proxy / TLS vars it needs to run and reach the API are
# allowlisted explicitly.
#
# Fail-closed by design: anything not matched is dropped, so a NEW secret
# added to Raven's environment later never leaks by default. A var the CLI
# genuinely needs but that isn't listed fails the CLI VISIBLY (classified
# review failure) — the safe direction, never silent credential exposure.
_CLI_ENV_ALLOW_EXACT = frozenset({
    # System / shell
    "HOME", "PATH", "USER", "LOGNAME", "SHELL", "TERM", "TZ", "TMPDIR", "PWD",
    # Locale
    "LANG", "LANGUAGE",
    # Proxy (libraries read both upper- and lower-case spellings)
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    # TLS / CA bundles
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
})
_CLI_ENV_ALLOW_PREFIX = (
    "CLAUDE_CODE_",   # the CLI's own auth (OAuth token/refresh) + config
    "ANTHROPIC_",     # alternate Anthropic API-key auth / config
    "LC_",            # locale (LC_ALL, LC_CTYPE, …)
    "XDG_",           # config / cache / data dirs the CLI may use
    "SSL_",           # SSL_CERT_FILE / SSL_CERT_DIR
    "NODE_",          # NODE_EXTRA_CA_CERTS, NODE_OPTIONS — node runtime / TLS
)


def _build_subprocess_env() -> dict[str, str]:
    """Allowlisted environment for the Claude CLI child.

    A copy of the process environment filtered to ``_CLI_ENV_ALLOW_EXACT``
    ∪ (anything starting with one of ``_CLI_ENV_ALLOW_PREFIX``). Everything
    else — including Raven's platform credentials — is dropped. See the
    allowlist comment above for the fail-closed rationale.
    """
    return {
        k: v for k, v in os.environ.items()
        if k in _CLI_ENV_ALLOW_EXACT or k.startswith(_CLI_ENV_ALLOW_PREFIX)
    }


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

    The child runs in a fresh empty temp directory (``cwd=``), never the
    service's own working directory. The CLI's file tools (grep/read)
    operate relative to its cwd; inheriting the app's cwd let reviews
    "verify" claims against the deployed container's own /app checkout —
    a copy that is stale relative to the reviewed repo's main and, for
    any repo other than Raven itself, an entirely unrelated codebase
    (observed producing confident false "implementation absent" findings
    on PR #160 and PR #163). A per-call directory (rather than one shared
    at init) guarantees each spawn starts empty even if a previous CLI
    invocation dropped session/debug files. The child's environment is
    scoped to an allowlist (``_build_subprocess_env``) so it still finds its
    OAuth credentials via ``CLAUDE_CODE_OAUTH_TOKEN`` / HOME-based config but
    NEVER inherits Raven's platform secrets (git / Bitbucket tokens, webhook
    secrets, API keys).
    """
    isolated_cwd = tempfile.mkdtemp(prefix="raven-claude-cli-")
    try:
        with subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=isolated_cwd,
            env=_build_subprocess_env(),
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
    finally:
        shutil.rmtree(isolated_cwd, ignore_errors=True)
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


from raven.ai.base import AIBackend, AIError, CompletionResult


# Classification of a non-zero CLI exit into an AIError.reason. The CLI
# collapses every API failure to a stderr/stdout line, so we pattern-match
# the (already-redacted) text. Order matters: usage/session caps are
# checked before the generic rate-limit pattern because a plan cap is NOT
# retryable in-process (resets hours later) while a 429 is. Patterns are
# matched case-insensitively against the redacted detail string.
_USAGE_LIMIT_MARKERS = ("usage limit", "session limit", "quota", "plan limit")
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "too many requests")
_AUTH_MARKERS = (
    "401", "403", "authentication", "invalid x-api-key", "invalid api key",
    "unauthorized", "permission denied", "forbidden", "oauth",
)
# Anthropic surfaces overload as 529; 5xx generally is transient/retryable.
_SERVER_ERROR_MARKERS = (
    "500", "502", "503", "504", "529", "overloaded", "internal server error",
    "bad gateway", "service unavailable",
)


def _classify_cli_failure(detail: str) -> str:
    """Map a redacted CLI failure string to an :class:`AIError` reason.

    ``usage_limit`` (plan cap) is checked before ``rate_limit`` so a plan
    cap — which a short retry can't fix — never masquerades as a retryable
    429. Returns ``"unknown"`` when nothing matches.
    """
    low = (detail or "").lower()
    if any(m in low for m in _USAGE_LIMIT_MARKERS):
        return "usage_limit"
    if any(m in low for m in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(m in low for m in _AUTH_MARKERS):
        return "auth"
    if any(m in low for m in _SERVER_ERROR_MARKERS):
        return "backend_5xx"
    return "unknown"


def _parse_cli_json(stdout: str) -> CompletionResult:
    """Parse the ``--output-format json`` envelope into a CompletionResult.

    Expected shape (confirmed against CLI v2.1.156)::

        {"result": "<text>", "total_cost_usd": 0.0893,
         "usage": {"input_tokens": N, "output_tokens": M, ...}, ...}

    Degrades gracefully: if stdout isn't the expected JSON object (CLI
    version drift, an unexpected envelope), fall back to treating the raw
    stdout as the response text with no usage/cost rather than failing the
    review. Token/cost telemetry is best-effort; the review text is not.
    """
    try:
        data = json.loads(stdout)
        if not isinstance(data, dict) or "result" not in data:
            raise ValueError("unexpected envelope shape")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "Claude CLI json envelope unparseable (%s) — using raw stdout as text, "
            "no token/cost recorded", e,
        )
        return CompletionResult(text=stdout)

    usage = data.get("usage") or {}
    cost = data.get("total_cost_usd")
    return CompletionResult(
        text=data.get("result", ""),
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


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
    ) -> CompletionResult:
        logger.info(
            "Claude CLI %s (model=%s effort=%s prompt_chars=%d)",
            purpose, model, effort, len(prompt),
        )
        args = CLAUDE_BASE_ARGS + ["--model", model, "--effort", effort]
        with _claude_semaphore:
            try:
                result = _run_claude_cli(args, prompt=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise AIError(f"claude CLI timed out after {timeout}s", reason="timeout")
            except FileNotFoundError:
                # A missing binary is a deploy fault, not transient — don't
                # retry. ``unknown`` keeps it out of the retry path.
                raise AIError(f"claude CLI not found at {CLAUDE_BIN}", reason="unknown")

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
            reason = _classify_cli_failure(detail)
            raise AIError(
                f"claude CLI exited with code {result.returncode}: {detail}",
                reason=reason,
            )
        return _parse_cli_json(result.stdout)

    def shutdown(self, grace_period: float = 2.0) -> int:
        return terminate_active_processes(grace_period)
