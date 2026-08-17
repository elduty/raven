"""Guard against docker-compose ↔ code default drift (audit #7).

``docker-compose.yml`` mirrors several Python-side defaults as
``${VAR:-<default>}`` (documented convention: "defaults must match
raven/reviewer.py"). When the two drift — as ``RAVEN_AI_MODEL`` /
``RAVEN_AI_EFFORT`` did (Dockerfile ``claude-opus-4-7`` vs code
``claude-fable-5``) — the same image behaves differently depending on the
launch path. These tests pin the agreement at the SOURCE level (no env
influence) so a future default change must touch both sides, and assert the
image no longer carries a third, drifting default.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
_REVIEWER = (_ROOT / "raven" / "reviewer.py").read_text(encoding="utf-8")
_DOCKERFILE = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
_REVIEW_PROMPT = (_ROOT / "prompts" / "review.md").read_text(encoding="utf-8")
_SERVER = (_ROOT / "raven" / "server.py").read_text(encoding="utf-8")


def _compose_default(var: str) -> str | None:
    """The ``<default>`` in a ``- VAR=${VAR:-<default>}`` compose line."""
    m = re.search(rf"-\s*{re.escape(var)}=\$\{{{re.escape(var)}:-([^}}]*)\}}", _COMPOSE)
    return m.group(1) if m else None


def _compose_default_nocolon(var: str) -> str | None:
    """The ``<default>`` in a ``- VAR=${VAR-<default>}`` compose line.

    The colon-less form is required for the directory vars: with
    ``${VAR:-default}`` an empty string in ``.env`` substitutes the
    default, silently breaking the documented "empty disables it" switch.
    """
    m = re.search(rf"-\s*{re.escape(var)}=\$\{{{re.escape(var)}-([^}}]*)\}}", _COMPOSE)
    return m.group(1) if m else None


def _code_default(var: str) -> str | None:
    """The default literal in ``os.environ.get("VAR", "<default>")``."""
    m = re.search(rf'os\.environ\.get\(\s*"{re.escape(var)}"\s*,\s*"([^"]*)"\s*\)', _REVIEWER)
    return m.group(1) if m else None


def _server_default(var: str) -> str | None:
    """The default literal in server.py's ``os.environ.get("VAR", "...")``."""
    m = re.search(rf'os\.environ\.get\(\s*"{re.escape(var)}"\s*,\s*"([^"]*)"\s*\)', _SERVER)
    return m.group(1) if m else None


def test_ai_model_default_matches_between_compose_and_code():
    compose, code = _compose_default("RAVEN_AI_MODEL"), _code_default("RAVEN_AI_MODEL")
    assert compose is not None and code is not None
    assert compose == code, (
        f"docker-compose RAVEN_AI_MODEL default {compose!r} != reviewer.py "
        f"default {code!r} — keep the two in sync (audit #7 drift)")


def test_ai_effort_default_matches_between_compose_and_code():
    compose, code = _compose_default("RAVEN_AI_EFFORT"), _code_default("RAVEN_AI_EFFORT")
    assert compose is not None and code is not None
    assert compose == code, (
        f"docker-compose RAVEN_AI_EFFORT default {compose!r} != reviewer.py "
        f"default {code!r} — keep the two in sync (audit #7 drift)")


def test_ai_timeout_default_matches_between_compose_and_code():
    compose, code = _compose_default("RAVEN_AI_TIMEOUT"), _code_default("RAVEN_AI_TIMEOUT")
    assert compose is not None and code is not None
    assert compose == code, (
        f"docker-compose RAVEN_AI_TIMEOUT default {compose!r} != reviewer.py "
        f"default {code!r} — keep the two in sync (audit #7 drift)")


def test_config_dir_default_matches_between_compose_and_code():
    compose = _compose_default_nocolon("RAVEN_CONFIG_DIR")
    code = _server_default("RAVEN_CONFIG_DIR")
    assert compose is not None and code is not None
    assert compose == code, (
        f"docker-compose RAVEN_CONFIG_DIR default {compose!r} != server.py "
        f"default {code!r} — keep the two in sync (audit #7 drift)")


def test_config_and_rules_dirs_use_the_colonless_compose_form():
    """Both directory vars document "empty string disables this". That
    only holds with ``${VAR-default}``; ``${VAR:-default}`` substitutes
    the default on an empty value and silently re-enables the feature."""
    for var in ("RAVEN_RULES_DIR", "RAVEN_CONFIG_DIR"):
        assert _compose_default_nocolon(var) is not None, \
            f"{var} must use ${{{var}-default}} (no colon) in docker-compose.yml"
        assert _compose_default(var) is None, \
            f"{var} must NOT use the ${{{var}:-default}} form — it breaks the disable switch"


def test_raven_config_lives_outside_the_agent_rules_dir():
    """The whole point of the move: Raven's per-repo config must not
    default to a subdirectory of the rules dir, where every other agent
    in the repo sweeps it into context."""
    config_default = _server_default("RAVEN_CONFIG_DIR")
    rules_default = _server_default("RAVEN_RULES_DIR")
    assert config_default and rules_default
    assert not config_default.startswith(rules_default.rstrip("/") + "/"), (
        f"RAVEN_CONFIG_DIR default {config_default!r} is inside "
        f"RAVEN_RULES_DIR {rules_default!r} — that reintroduces the "
        "context leak the .raven/ move exists to fix")
    assert not config_default.startswith(".claude/")


def test_dockerfile_does_not_set_model_or_effort_env():
    # The image must NOT bake its own RAVEN_AI_MODEL/EFFORT default — that was
    # the third source that drifted. Defaults live in reviewer.py (+ compose).
    assert "RAVEN_AI_MODEL=" not in _DOCKERFILE, \
        "Dockerfile must not set a RAVEN_AI_MODEL ENV default (audit #7)"
    assert "RAVEN_AI_EFFORT=" not in _DOCKERFILE, \
        "Dockerfile must not set a RAVEN_AI_EFFORT ENV default (audit #7)"


def test_readme_documents_the_severity_scale_file():
    """README must document the per-repo severities.json file and its
    merge-blocking-threshold key — the two facts an operator needs before
    writing one (task-12 brief, spec 2026-08-04-configurable-severity-scale).
    """
    readme = open("README.md").read()
    assert "severities.json" in readme
    assert "blocks_at_or_above" in readme


def test_readme_documents_the_blocking_threshold_sentinel():
    """README must document the ``blocking`` notification-channel sentinel
    (``severity.BLOCKING``) — the only ``min_severity`` value that means the
    same thing in every repo's vocabulary.

    The word "blocking" alone is too weak to assert on: it already appears
    six times in README.md for unrelated reasons (``merge-blocking``,
    ``blocking the merge``, ...), so that assertion passes whether or not
    the sentinel itself is documented. Assert the literal config value
    instead, the way ``test_readme_documents_the_severity_scale_file``
    checks for ``severities.json`` and ``blocks_at_or_above``.
    """
    assert '"min_severity": "blocking"' in open("README.md").read()


def test_max_findings_matches_review_prompt_cap():
    """``reviewer.MAX_FINDINGS`` mirrors the cap stated in prompts/review.md.

    The prompt states the cap for the model; the constant enforces it
    deterministically on the policy-less chunked path (audit 07-02 #9).
    If the two drift, Raven either truncates below what it promised the
    model or posts above its own stated maximum.
    """
    prompt = re.search(r"Maximum\s+(\d+)\s+findings", _REVIEW_PROMPT)
    assert prompt is not None, \
        "prompts/review.md no longer states 'Maximum N findings' — update this guard"
    code = re.search(r"^MAX_FINDINGS\s*=\s*(\d+)", _REVIEWER, re.M)
    assert code is not None, "raven/reviewer.py no longer defines MAX_FINDINGS"
    assert int(code.group(1)) == int(prompt.group(1)), (
        f"prompts/review.md cap {prompt.group(1)} != reviewer.py MAX_FINDINGS "
        f"{code.group(1)} — keep the two in sync")
