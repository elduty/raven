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


def _compose_default(var: str) -> str | None:
    """The ``<default>`` in a ``- VAR=${VAR:-<default>}`` compose line."""
    m = re.search(rf"-\s*{re.escape(var)}=\$\{{{re.escape(var)}:-([^}}]*)\}}", _COMPOSE)
    return m.group(1) if m else None


def _code_default(var: str) -> str | None:
    """The default literal in ``os.environ.get("VAR", "<default>")``."""
    m = re.search(rf'os\.environ\.get\(\s*"{re.escape(var)}"\s*,\s*"([^"]*)"\s*\)', _REVIEWER)
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


def test_dockerfile_does_not_set_model_or_effort_env():
    # The image must NOT bake its own RAVEN_AI_MODEL/EFFORT default — that was
    # the third source that drifted. Defaults live in reviewer.py (+ compose).
    assert "RAVEN_AI_MODEL=" not in _DOCKERFILE, \
        "Dockerfile must not set a RAVEN_AI_MODEL ENV default (audit #7)"
    assert "RAVEN_AI_EFFORT=" not in _DOCKERFILE, \
        "Dockerfile must not set a RAVEN_AI_EFFORT ENV default (audit #7)"
