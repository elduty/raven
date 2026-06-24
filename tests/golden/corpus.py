"""Loader for the golden-review scenario corpus.

On-disk layout (one directory per scenario under
``tests/golden/scenarios/``):

    scenarios/<name>/
        diff.patch          # required — the unified diff to review
        expectation.json    # required — coarse expected outcome (see below)
        files/<path>        # optional — full file contents, keyed by the
                            #            "file_contents" map in expectation.json

``expectation.json`` schema (only ``expect`` is required; everything
else has a sensible default)::

    {
      "description": "human note shown in the test id",
      "repo_name": "golden/clean-small",        // default: golden/<name>
      "is_incremental": false,                   // passed to review_diff
      "unchanged_files": ["billing/discount.py"],// optional; PR files NOT in
                                                 //   the delta — forwarded to
                                                 //   review_diff (pairs with
                                                 //   is_incremental)
      "max_diff_lines": 50,                      // optional; pins
                                                 //   reviewer.MAX_DIFF_LINES
                                                 //   for THIS scenario's call
                                                 //   (runner monkeypatches it)
                                                 //   so a coverage-gap scenario
                                                 //   triggers the oversized
                                                 //   path off a fixed cap
      "file_contents": {                         // optional
        "src/app.py": "files/app.py"             // value = path under the
      },                                         //   scenario dir to read
      "expect": { ... }                          // the scorer's expect block
    }

The loader is pure (filesystem only, no AI) so it is unit-tested offline.
``review_kwargs()`` returns exactly the keyword arguments to splat into
``raven.reviewer.review_diff(**kwargs)`` — keeping the runner a thin
shell over the corpus + scorer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


@dataclass
class Scenario:
    """One loaded golden scenario: its diff, the kwargs to feed
    ``review_diff``, and the coarse ``expect`` spec to score against."""

    name: str
    description: str
    diff: str
    repo_name: str
    file_contents: dict[str, str] | None
    is_incremental: bool
    unchanged_files: list[str] | None
    expect: dict
    # Optional per-scenario pin for ``reviewer.MAX_DIFF_LINES`` during this
    # scenario's review_diff call (the runner monkeypatches it). Lets a
    # coverage-gap scenario trigger the chunked/oversized path off a small,
    # FIXED cap instead of the ambient default — so an operator/CI
    # ``MAX_DIFF_LINES`` override (or a line-counting change) can't quietly
    # dispatch the oversized chunk to a real backend. ``None`` = leave the
    # ambient value alone.
    max_diff_lines: int | None = None

    def review_kwargs(self) -> dict:
        """Keyword args for ``raven.reviewer.review_diff(**kwargs)``.

        Only includes args the scenario actually sets, so the corpus
        stays forward-compatible with review_diff's many optional
        params (we add what we use, not the full signature)."""
        kwargs: dict = {"diff": self.diff, "repo_name": self.repo_name}
        if self.file_contents:
            kwargs["file_contents"] = self.file_contents
        if self.is_incremental:
            kwargs["is_incremental"] = True
        if self.unchanged_files:
            kwargs["unchanged_files"] = self.unchanged_files
        return kwargs


def _load_one(scenario_dir: Path) -> Scenario:
    name = scenario_dir.name
    diff_path = scenario_dir / "diff.patch"
    exp_path = scenario_dir / "expectation.json"
    if not diff_path.is_file():
        raise FileNotFoundError(f"scenario {name!r}: missing diff.patch")
    if not exp_path.is_file():
        raise FileNotFoundError(f"scenario {name!r}: missing expectation.json")

    diff = diff_path.read_text(encoding="utf-8")
    meta = json.loads(exp_path.read_text(encoding="utf-8"))
    if "expect" not in meta or not isinstance(meta["expect"], dict):
        raise ValueError(f"scenario {name!r}: expectation.json must contain an 'expect' object")

    # Resolve file_contents: each value is a path (relative to the
    # scenario dir) whose bytes become the file body. Reading from a
    # sibling file keeps large source bodies out of the JSON.
    file_contents: dict[str, str] | None = None
    raw_files = meta.get("file_contents")
    if raw_files:
        file_contents = {}
        for logical_path, source_ref in raw_files.items():
            ref_path = scenario_dir / source_ref
            if not ref_path.is_file():
                raise FileNotFoundError(
                    f"scenario {name!r}: file_contents[{logical_path!r}] points to "
                    f"{source_ref!r} which does not exist"
                )
            file_contents[logical_path] = ref_path.read_text(encoding="utf-8")

    mdl = meta.get("max_diff_lines")
    if mdl is not None and (not isinstance(mdl, int) or isinstance(mdl, bool) or mdl <= 0):
        raise ValueError(
            f"scenario {name!r}: max_diff_lines must be a positive int, got {mdl!r}")

    return Scenario(
        name=name,
        description=str(meta.get("description", "")),
        diff=diff,
        repo_name=str(meta.get("repo_name") or f"golden/{name}"),
        file_contents=file_contents,
        is_incremental=bool(meta.get("is_incremental", False)),
        unchanged_files=list(meta["unchanged_files"]) if meta.get("unchanged_files") else None,
        expect=meta["expect"],
        max_diff_lines=mdl,
    )


def load_scenarios(root: Path | None = None) -> list[Scenario]:
    """Load every scenario under ``root`` (default ``scenarios/``),
    sorted by name for stable test ordering. A directory without a
    ``diff.patch`` is skipped (lets the dir hold a README or fixtures)."""
    base = root or SCENARIOS_DIR
    scenarios: list[Scenario] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "diff.patch").is_file():
            continue
        scenarios.append(_load_one(child))
    return scenarios
