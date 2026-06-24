# Golden-review eval harness

A small, **opt-in** harness that replays recorded PR-review scenarios
against the **live AI backend** and scores the results with coarse,
robust checks (precision / recall / false-positive rate). It exists to:

- prove a change cut **false positives** without raising **false
  negatives** (the confident-false-positive class — e.g. a tests-only
  PR flagged as "implementation missing" — is the headline regression);
- A/B **reasoning effort** (`max` vs `high`) on review quality, not just
  latency/cost;
- gate future model / effort swaps before they touch the
  `RAVEN_AI_MODEL` / `RAVEN_AI_EFFORT` defaults.

It is **skipped in normal CI** — it needs a real backend and spends
tokens. The gating mirrors `tests/test_prompt_injection_live.py`:
`RAVEN_LIVE_AI_TESTS=1` plus the `slow` marker. The *scoring logic* is
unit-tested offline and **does** run in CI.

## What runs where

| File | Live AI? | Runs in CI? | Purpose |
|------|----------|-------------|---------|
| `scorer.py` | no | — (library) | Pure scoring: verdict bucket, must-/must-not-contain, coverage-gap, precision/recall/FP-rate. |
| `corpus.py` | no | — (library) | Loads scenarios from `scenarios/`, builds `review_diff` kwargs. |
| `test_scorer.py` | **no** | **yes** | Unit-tests the scorer against hand-written review dicts. |
| `test_corpus.py` | **no** | **yes** | Unit-tests the loader **and** the checked-in scenarios' regex patterns (catches a too-loose absence pattern or a real-bug pattern that never matches). |
| `test_golden_review_live.py` | **yes** | **no** (opt-in) | Replays each scenario through `raven.reviewer.review_diff` and scores it. |

So a normal `pytest tests/` exercises the measurement instrument
(scorer + corpus + every scenario's patterns) without ever calling a
backend; only `test_golden_review_live.py` needs the opt-in.

## Running the live harness

```bash
# Claude CLI backend
RAVEN_LIVE_AI_TESTS=1 CLAUDE_CODE_OAUTH_TOKEN=<token> \
    pytest -m slow tests/golden/test_golden_review_live.py -v -s

# OpenAI-compatible backend (backend-agnostic — same scenarios)
RAVEN_LIVE_AI_TESTS=1 RAVEN_AI_API_BASE=<url> RAVEN_AI_API_KEY=<key> \
    pytest -m slow tests/golden/test_golden_review_live.py -v -s
```

`-s` surfaces the `[golden:<effort>] …` summary line. Each scenario is
reviewed **once per process** (memoised), so the per-scenario tests and
the aggregate summary score the same runs.

## A/B-ing reasoning effort

`review_diff` reads `RAVEN_AI_EFFORT` at import (`reviewer.py:22`), so
A/B is just two runs with the env var flipped. The runner reports the
effort straight off `reviewer.RAVEN_AI_EFFORT`, so the printed number
can't drift from what was actually used:

```bash
RAVEN_AI_EFFORT=high RAVEN_LIVE_AI_TESTS=1 CLAUDE_CODE_OAUTH_TOKEN=<token> \
    pytest -m slow tests/golden/ -s -q
RAVEN_AI_EFFORT=max  RAVEN_LIVE_AI_TESTS=1 CLAUDE_CODE_OAUTH_TOKEN=<token> \
    pytest -m slow tests/golden/ -s -q
```

Compare the two lines, e.g.:

```
[golden:high] scenarios 4/4 passed | precision=1.00 recall=1.00 fp_rate=0.00 | TP=1 FP=0 FN=0 N-neg=3
[golden:max]  scenarios 4/4 passed | precision=1.00 recall=1.00 fp_rate=0.00 | TP=1 FP=0 FN=0 N-neg=3
```

A regression shows up as `fp_rate > 0` (a false positive crept in) or a
must_contain check flipping to FN (a real bug got missed).

## Scoring philosophy — coarse and robust

LLM output varies run to run, so the scorer never asserts exact finding
text or counts. It asserts on a few **robust properties** of the review
dict:

- **verdict bucket** — `approvable` (effective severity `low`, no
  medium/high anywhere) / `max_severity` cap / `coverage_gap`;
- **must-NOT-contain** — the FP guard: no finding message may match a
  forbidden regex. Each violated pattern is a **false positive**;
- **must-contain** — the real-finding guard: some finding must match a
  file pattern **and** name the defect class by keyword. Satisfied ⇒
  **true positive**; unsatisfied ⇒ **false negative**;

Aggregate `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`,
`fp_rate = FP/negatives`. A zero denominator reports `n/a` (a corpus of
only-positive or only-negative assertions still renders).

The live runner hard-fails on **any** false positive (`fp_rate > 0`) —
catching confident false positives is the whole point — and on any
scenario whose own checks didn't all pass.

## Scenario corpus

One directory per scenario under `scenarios/`:

```
scenarios/<name>/
    diff.patch          # required — the unified diff to review
    expectation.json    # required — coarse expected outcome
    files/<path>        # optional — full file bodies referenced by expectation
```

Seed scenarios:

| Scenario | Shape | Coarse expectation |
|----------|-------|--------------------|
| `clean_small` | tiny correct refactor (behaviour-preserving) | approvable, `max_severity: low`, must not invent an absence/crash defect |
| `tests_only_no_phantom_absence` | tests-only diff; implementation supplied as an unchanged file; declared incremental (the PR #157/#160 trap) | must **not** claim the implementation is missing/absent/not-implemented |
| `real_bug` | SQL built by string concat / `%`-format of an untrusted name | not approvable; ≥1 finding on `api/users.py` naming SQL-injection / unparameterised-query |
| `oversized_chunk` | single-file diff that, against the scenario's pinned `max_diff_lines` cap, exceeds `MAX_DIFF_LINES*3` | `coverage_gap: true`, not approvable. The scenario **pins** `max_diff_lines` (the runner monkeypatches `reviewer.MAX_DIFF_LINES` for this scenario only) so the oversized path triggers off a *fixed* cap regardless of ambient config — and the oversized chunk is filtered out *before* dispatch, so this one asserts `review_diff`'s gap bookkeeping and spends no tokens. |

### `expectation.json` format

```jsonc
{
  "description": "shown in the test id / failure report",
  "repo_name": "golden/<name>",          // optional; default golden/<dirname>
  "is_incremental": false,               // optional; passed to review_diff
  "unchanged_files": ["pkg/mod.py"],     // optional; PR files NOT in the delta,
                                         //   forwarded to review_diff (pairs
                                         //   with is_incremental)
  "max_diff_lines": 50,                  // optional; pins reviewer.MAX_DIFF_LINES
                                         //   for THIS scenario's call so a
                                         //   coverage-gap scenario triggers the
                                         //   oversized path off a fixed cap
  "file_contents": {                     // optional
    "pkg/mod.py": "files/mod.py"         // value = path under the scenario dir
  },
  "expect": {                            // required — the scorer's spec
    "approvable": true,                  // effective severity must be low
    // "approvable": false,              // …or must be medium+
    "max_severity": "low",               // effective severity <= this tier
    "coverage_gap": true,                // review["coverage_gap"] must equal this
    "negatives": 1,                      // FP-rate denominator (default: #must_not_contain_patterns)
    "must_not_contain_patterns": [       // FP guard — regex, case-insensitive
      "\\b(missing|absent|not implemented)\\b"
    ],
    "must_contain": [                    // real-finding guard
      {
        "file_pattern": "api/users\\.py",          // optional; regex over finding.file
        "message_patterns": ["sql\\s*injection"],  // any-of; regex over finding.message
        "label": "sql-injection"                   // optional; for the report
      }
    ]
  }
}
```

Every `expect` key is optional — a scenario only pays for the assertions
it declares.

## Adding a scenario

1. `mkdir tests/golden/scenarios/<name>` and drop a `diff.patch` in it.
2. Write `expectation.json` with a coarse `expect` block (see above).
   Prefer must-not-contain (FP guard) and must-contain-by-file+keyword
   (real-finding guard) over verdict-only checks — they're the robust
   signal. If the diff needs file context, add `files/<path>` bodies and
   reference them from `file_contents`.
3. Add an **offline** assertion in `test_corpus.py` proving your
   patterns match the phrasing you intend and *don't* fire on benign
   prose. This keeps a regex mistake out of the corpus without needing a
   live run.
4. Sanity-check it loads: `pytest tests/golden/test_corpus.py -q`.
5. When you have backend creds, run the live harness (above) to confirm
   the real model satisfies the expectation at your target effort.
