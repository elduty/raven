"""Raven's version, ``raven.__version__``, is what every review footer
shows. It is pinned to CHANGELOG.md so a release can't forget the bump:
the newest ``## vX.Y.Z`` heading, plus ``+dev`` while an ``## Unreleased``
section above it lists changes (main between releases)."""

import re
from pathlib import Path

import raven

_CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _expected_version(changelog: str) -> str:
    release = re.search(r"^## v(\d+\.\d+\.\d+)\b", changelog, re.M)
    assert release, "CHANGELOG.md has no '## vX.Y.Z' heading"
    above = changelog[:release.start()]
    unreleased = re.search(r"^## Unreleased\s*$(.*)", above, re.M | re.S)
    dev = bool(unreleased and re.search(r"^- ", unreleased.group(1), re.M))
    return release.group(1) + ("+dev" if dev else "")


def test_version_matches_the_changelog():
    expected = _expected_version(_CHANGELOG.read_text(encoding="utf-8"))
    assert raven.__version__ == expected, (
        f"raven.__version__ is {raven.__version__!r}; CHANGELOG.md says "
        f"{expected!r}. A release sets it to the new version; the first "
        f"change after one adds '+dev'.")


def test_the_rule():
    release = "# Changelog\n\n## v0.7.1 — 2026-10-01\n\n- a fix\n"
    assert _expected_version(release) == "0.7.1"
    assert _expected_version(release.replace(
        "## v0.7.1", "## Unreleased\n\n### Fixed\n\n- b fix\n\n## v0.7.1")) == "0.7.1+dev"
    assert _expected_version(release.replace(
        "## v0.7.1", "## Unreleased\n\n## v0.7.1")) == "0.7.1"
