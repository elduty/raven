# Changelog

All notable changes to Raven are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/) loosely; dates are UTC.

## Unreleased

### New features

- **`RAVEN_REVIEW_MODE` flag** — `all` (default) / `gap` / `advisory`. The `advisory` mode posts a non-blocking **Raven Recommendation** comment with inline findings instead of a formal blocking review; Raven doesn't auto-add itself as a reviewer, doesn't dispatch auto-merge, and bypasses the reviewer-listed gate so it engages on every PR webhook. Useful for trial deployments or teams that prefer humans to drive merges.

### Breaking changes

- **`RAVEN_REVIEW_ALL_PRS` removed.** Use `RAVEN_REVIEW_MODE` instead: `true` (or unset) → `all`, `false` → `gap`. The old env var is silently ignored if set. Invalid `RAVEN_REVIEW_MODE` values fail closed (exit at startup).
- **Legacy env-var aliases removed.** Operators must rename their `.env` if they were using any of these:
  - `CLAUDE_MODEL` → `RAVEN_AI_MODEL`
  - `CLAUDE_EFFORT` → `RAVEN_AI_EFFORT`
  - `CLAUDE_EFFORT_COMMENT` → `RAVEN_AI_EFFORT_COMMENT`
  - `CLAUDE_TIMEOUT` → `RAVEN_AI_TIMEOUT`
  - `RAVEN_MAX_CONCURRENT_CLAUDE` → `RAVEN_AI_MAX_CONCURRENT`
  - `RAVEN_WEBHOOK_SECRET` → `GITEA_WEBHOOK_SECRET`
- **Python identifiers renamed** to match env vars (no API change for operators; affects out-of-tree code importing from `raven.reviewer`): `CLAUDE_MODEL` / `CLAUDE_EFFORT` / `CLAUDE_EFFORT_COMMENT` / `CLAUDE_TIMEOUT` / `MAX_CONCURRENT_CLAUDE` → `RAVEN_AI_MODEL` / `RAVEN_AI_EFFORT` / `RAVEN_AI_EFFORT_COMMENT` / `RAVEN_AI_TIMEOUT` / `RAVEN_AI_MAX_CONCURRENT`.
- **Legacy `/hook` endpoint removed.** Webhooks must use `/hook/gitea` or `/hook/bitbucket-dc`. The provider-order-dependent `/hook` route is gone.
- **Cache schema 3-tuple loader removed.** Pre-2026-05-13 cache files (which used a 3-tuple per entry) are no longer loaded — Raven now requires the `CacheEntry` dict shape. On upgrade, stale legacy entries are skipped at load time and the cache re-warms from the next push.

### Fixes (comment-thread retraction flow)

- **Retraction now works.** Previously, the AI was told to populate `retract_findings` with comment IDs from the active thread — but the rendered thread block stripped IDs. Result: the AI acknowledged in plain text but `retract_findings` was always empty, the inline comment never got resolved, and the verdict stayed at `needs_work`. The thread block now exposes `[id=N]` after each commenter's name so the AI can reference them.
- **Verdict revision now triggers on retraction.** The respond prompt's "DON'T revise: ... acknowledgements" rule conflicted with the DO-revise rules and led the AI to interpret its own acceptance as "just an acknowledgement, don't revise". Rewritten: removed the ambiguous "acknowledgements" line; added an explicit "if you retract findings that were the basis for `needs_work`, set `revise` to flip to `approve`" rule.
- **Auto-flip backstop in the server.** Defense in depth: when the AI retracts all cached findings but doesn't set `revise`, and the prior verdict was `needs_work`, the server now synthesizes a flip to `approve`. The basis for blocking is gone; the PR shouldn't stay blocked just because the AI was conservative about revising. New formal review goes out and auto-merge dispatches normally.
- **Retract action now authorship-filtered.** The server previously filtered AI-supplied `retract_findings` IDs against thread membership only — a hallucinating AI (or prompt-injection vector) could have caused Raven to mark a developer's comment as resolved on the platform. Now restricted to comments authored by Raven; everything else is dropped with a debug log.

## v0.1.0 — 2026-05-14

First public snapshot. Raven now reviews PRs on both Gitea and Bitbucket Data Center, talks to two pluggable AI backends, holds context-aware conversations with developers in PR threads, and revises its own verdict when the discussion warrants.

### New features

**Multi-provider support**
- Bitbucket Data Center provider with full parity: webhook parsing, diff fetch, inline comments, formal reviews (approve / needs-work), participant-aware reviewer detection, native comment resolve (`state=RESOLVED`).
- `GitProvider` ABC + provider registry — adding a new platform means implementing the abstract methods. Out-of-tree providers degrade gracefully on the conversational-context methods (concrete defaults on the base class).
- Single Raven instance handles webhooks from multiple platforms simultaneously via `/hook/<provider>` routing.

**Pluggable AI backends**
- `claude_cli` — wraps the Claude Code CLI as a subprocess.
- `openai_compatible` — HTTP via the `openai` SDK, talks to any OpenAI-compatible endpoint (LiteLLM proxy, vLLM, OpenRouter, OpenAI itself).
- Auto-selection from configured credentials; explicit override via `RAVEN_AI_BACKEND`.
- Per-backend concurrency cap via `RAVEN_AI_MAX_CONCURRENT` (semaphore-gated).
- `RAVEN_AI_EFFORT` / `RAVEN_AI_EFFORT_COMMENT` map to backend reasoning controls — supports `none` to omit `reasoning_effort` entirely for non-reasoning models behind a proxy that doesn't strip the param.

**Comment thread context + verdict re-evaluation**
- When a developer replies to a Raven finding, Raven sees the **whole conversation it's replying inside** (root + replies), not just the last 20 PR-wide comments. Thread root is always preserved through truncation.
- AI returns structured JSON: response text + optional verdict revision + optional finding retraction list.
- When the discussion provides substantive new information ("this isn't a bug because the path is unreachable"), Raven can submit a **new formal review revising its prior verdict**. Auto-merge gates fire on `needs_work → approve` flips and also on retraction-only paths when the prior verdict was already approve.
- When discussion invalidates a specific inline finding, Raven retracts it via the platform's **native resolve action**: Bitbucket DC marks the comment `state=RESOLVED`; Gitea ≥1.24 uses `POST /pulls/comments/{id}/resolve`.
- Atomic race guards prevent two concurrent comments from submitting opposing reviews.
- Configurable budgets: `RAVEN_RESPOND_THREAD_TOTAL_CHARS`, `RAVEN_RESPOND_VERDICT_BODY_CHARS`.

**Conversational follow-up polish**
- Immediate 👀 reaction on Gitea so developers know Raven picked up the comment within a second.
- Threaded replies on platforms that support them (Bitbucket DC).
- No re-@mention needed inside threads where Raven already participated.
- Line-aware context — inline diff comments get a line-numbered snippet of the file injected into the prompt.

**Reviewer assignment + auto-merge**
- `RAVEN_REVIEW_ALL_PRS=true` (default): Raven auto-adds itself to every PR; auto-merges only when it's the sole reviewer; leaves PRs with humans for humans to merge.
- `RAVEN_REVIEW_ALL_PRS=false`: fill-gap mode — Raven only auto-adds on PRs with no other reviewer.
- Adding Raven as a reviewer manually always triggers a fresh review.
- `RAVEN_GITEA_AUTO_MERGE` opt-in: use Gitea's native auto-merge queue instead of polling CI ourselves.

**Repo-supplied review context**
- `CLAUDE.md` at the repo root injected as "Repository Context" (read from **PR head**).
- `.claude/rules/*.md` injected as "Repository Rules" (read from **PR base branch** — a PR can't add or modify its own review criteria).
- `.claude/rules/raven/prompts/{review,respond}.md` — per-repo prompt overrides that completely replace Raven's built-in prompts (same merge-first security model).
- PR title, description, and recent comments included in the review prompt as "PR Conversation" context with configurable per-item and global character budgets.
- Rules take priority over the built-in prompt template.

**Incremental + consolidated reviews**
- Diff hashes stored per file — only changed files are re-reviewed on push.
- Findings from unchanged files **carried forward** and merged with new findings for a consolidated verdict.
- Cache invalidates automatically on backend / model / effort / prompt change.
- LRU eviction at `RAVEN_MAX_CACHED_PRS` (default 200).
- Cache schema is a `CacheEntry` dataclass with verdict + summary persistence (auto-migrates legacy 3-tuple entries on load).

**Operations**
- `/metrics` Prometheus endpoint with **mandatory bearer-token auth** via `RAVEN_METRICS_TOKEN` (returns 404 when unset — doesn't advertise itself).
- New counters: `raven_verdict_revisions_total`, `raven_retractions_total`, `raven_response_parse_errors_total`, `raven_revision_submit_errors_total`, plus the existing review / merge / error counters.
- `/healthz` open for load balancers (no auth).
- Log rotation via Docker's json-file driver (10 MB × 5 files default ceiling).
- Graceful shutdown: on SIGTERM, queued reviews and CI-wait tasks are cancelled and in-flight Claude CLI subprocesses are SIGTERMed so gunicorn's shutdown window isn't consumed by discarded work.
- Dedicated thread pool for the post-review CI-wait phase (`RAVEN_CI_WAIT_WORKERS`) so sleeping waits don't starve incoming webhooks.

### Security

- **Stale-approval bypass closed** on auto-merge — head SHA re-verified at merge time so a force-push between approval and merge can't slip past review.
- **Prompt-injection defense** — all user-controlled content (diff, comments, file contents, rules, CLAUDE.md) wrapped in randomised `<untrusted_input_<tag_id>>` tags introduced by a trust preamble. Backed by an opt-in real-AI integration test (`RAVEN_LIVE_AI_TESTS=1`) that exercises an adversarial diff with planted injection attempts.
- **Rules read from PR base branch** — a PR can't add or modify its own review criteria; new rules only take effect after they've been merged through a review of their own. Same security model applies to prompt overrides.
- **OAuth token JSON escape** — the entrypoint now writes the bare-token JSON via Python's `json.dumps`, so tokens containing `"` or `\` no longer break the credential file.
- **`/metrics` requires bearer token** (fail-closed: 404 when unset).
- **PR dedup is SHA-aware** — webhook redelivery storms are absorbed, but a new head SHA is treated as a fresh event so pushes during an in-flight review aren't dropped.

### Fixes

- Bitbucket DC `get_pr_reviews` no longer maps every non-APPROVED participant (including the PR author) to a synthetic `state=COMMENT` review — only APPROVED / NEEDS_WORK participants count as reviewers, so auto-merge actually works on BB DC.
- Gitea `requested_reviewers` read from the PR object (the `/pulls/{id}/requested_reviewers` endpoint 404s in some configurations).
- Raven excluded from the `requested_reviewers` self-check in the auto-merge gate (it should never block its own merge).
- `_is_bot_author` tightened to affix match so usernames containing `bot` as a substring aren't accidentally skipped.
- File paths URL-encoded in `fetch_file` for both providers — paths with spaces / special chars no longer 404.
- `NOTIFY_CHANNELS` re-read on every `notify()` call so config edits take effect without a restart.
- Auto-add reviewer gate distinguishes auth failures (401 / 403 on the participants endpoint) from a real reviewer list — log messages now say "could not verify reviewer state — service account may lack repo access" instead of "PR already has other reviewers".
- `CI_WAIT_TIMEOUT=0` short-circuits the CI wait as documented (previously fell into the polling loop with a zero timeout).
- `_fetch_rules` deferred until after the no-changes skip on incremental re-reviews (saves a base-branch round-trip when nothing changed).
- Comment replies fetch `CLAUDE.md` from the PR head SHA, consistent with `_process_pr`.
- Rules / `CLAUDE.md` fetch errors logged at `warning` level (the "file legitimately absent" 404 path stays silent — that's the common case).

### Migration guide

Most new env vars are optional with defaults. The notable ones to know about:

| Variable | Default | Purpose |
|---|---|---|
| `RAVEN_AI_BACKEND` | auto | Force `claude_cli` or `openai_compatible`; omit to auto-detect from creds |
| `RAVEN_AI_API_BASE`, `RAVEN_AI_API_KEY` | — | Enables the `openai_compatible` backend |
| `RAVEN_AI_MAX_TOKENS` | — | Lift the model's default output cap (e.g., GPT-4o defaults to 16k) |
| `RAVEN_AI_MODEL`, `RAVEN_AI_EFFORT`, `RAVEN_AI_EFFORT_COMMENT`, `RAVEN_AI_TIMEOUT` | sensible defaults | Backend-agnostic model / reasoning / timeout controls |
| `RAVEN_AI_MAX_CONCURRENT` | 4 | Caps in-flight AI calls per backend |
| `BITBUCKET_DC_URL`, `BITBUCKET_DC_TOKEN`, `BITBUCKET_DC_WEBHOOK_SECRET`, `BITBUCKET_DC_USERNAME` | — | Enables the Bitbucket Data Center provider |
| `RAVEN_REVIEW_ALL_PRS` | true | false → fill-gap mode (Raven only auto-adds on PRs with no other reviewer) |
| `RAVEN_GITEA_AUTO_MERGE` | false | Use Gitea's native auto-merge queue instead of polling CI |
| `RAVEN_RESPOND_THREAD_TOTAL_CHARS`, `RAVEN_RESPOND_VERDICT_BODY_CHARS` | 8000 / 4000 | Comment-reply context budgets |
| `RAVEN_REVIEW_COMMENT_CONTEXT`, `RAVEN_REVIEW_PR_CONTEXT_ITEM_CHARS`, `RAVEN_REVIEW_PR_CONTEXT_TOTAL_CHARS` | 20 / 4000 / 16000 | PR-conversation context in review prompt |
| `RAVEN_RULES_DIR`, `RAVEN_REVIEW_RULES_TOTAL_CHARS` | `.claude/rules` / 16000 | Repo-supplied rules + prompt overrides; empty string disables both |
| `RAVEN_METRICS_TOKEN` | — | **Required to access `/metrics`** — endpoint returns 404 when unset |
| `RAVEN_MAX_CACHED_PRS` | 200 | LRU cap on the findings cache |
| `RAVEN_CI_WAIT_WORKERS` | 32 | Dedicated pool for post-review CI-wait |

**Legacy aliases still honored** (no need to rename if you're upgrading in place): `CLAUDE_MODEL`, `CLAUDE_TIMEOUT`, `CLAUDE_EFFORT`, `CLAUDE_EFFORT_COMMENT`, `RAVEN_MAX_CONCURRENT_CLAUDE`, `RAVEN_WEBHOOK_SECRET`.

**Provider compatibility**
- Native finding retraction (`POST /pulls/comments/{id}/resolve`) requires **Gitea ≥1.24**. On older Gitea instances, retraction logs a warning and no-ops — verdict revision and thread context still work normally.
- Bitbucket Data Center: no minimum version requirement for this feature.

**Cache**
- The findings cache schema migrated from a 3-tuple to a `CacheEntry` dataclass with verdict + summary. **Migration is automatic on load** — no manual step. The cache auto-wipes on startup if the backend / model / effort / prompt hash changed.

**Webhooks**
- The legacy `/hook` route still works (alias for `/hook/gitea`). New deployments should use `/hook/gitea` and `/hook/bitbucket-dc` explicitly.

### Stats

- 585 tests across 13 test files
- 18 files changed in the public-snapshot diff (+2954, -286) — represents the full delta since the prior published snapshot
