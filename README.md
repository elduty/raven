# Raven

[![Tests](https://github.com/elduty/raven/actions/workflows/test.yml/badge.svg)](https://github.com/elduty/raven/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Automated AI code review for Gitea and Bitbucket Data Center. Every pull request is reviewed before merge. Multi-provider — a single Raven instance handles webhooks from multiple git platforms simultaneously.

## How it works

```
push to branch
  -> open PR
  -> webhook fires (Gitea, Bitbucket DC, or future provider)
  -> Raven receives webhook, returns 200 immediately
  -> background thread:
      -> deduplicate (skip if same PR reviewed within 30s)
      -> fetch PR diff via provider API
      -> strip lockfiles and binaries
      -> incremental check: only review files changed since last review
      -> fetch full file contents for context
      -> send to the configured AI backend (claude_cli subprocess, or openai_compatible HTTP)
      -> parse JSON response with file/line locations
      -> guard: if parse failed, post warning, notify, do NOT merge
      -> submit formal review (APPROVED or REQUEST_CHANGES)
      -> post inline comments on specific diff lines
      -> add raven-reviewed label
      -> decision:
          -> Raven is sole reviewer + approved + CI passes -> auto-merge
          -> Other reviewers present -> leave open
          -> severity high/medium -> Slack/webhook alert, PR stays open
```

Reviews typically complete in 10-30 seconds. Diffs up to 3000 lines are reviewed as a single full-context pass; larger diffs are split by file and reviewed chunk-by-chunk. Repos can tune reviews via a few optional inputs:

- **`CLAUDE.md`** at the repo root — free-form project guidance. Injected as "Repository Context" inside a trusted `<repo_policy>` block (the model treats it as authoritative). **Fetched from the PR's base branch** (already-merged state).
- **`.claude/rules/*.md`** — one file per rule category (e.g. `security.md`, `style.md`, `testing.md`). Injected as "Repository Rules" inside the same trusted `<repo_policy>` block — the model applies them as authoritative review criteria that override conflicting guidance in the built-in prompt. Flat directory, `.md` only. **Fetched from the PR's base branch** (already-merged state) so neither rules nor CLAUDE.md can be added or modified to bias their own PR's review — changes only take effect after the introducing PR has merged through its own review cycle.

- **`.claude/rules/raven/prompts/review.md`** and **`.claude/rules/raven/prompts/respond.md`** — optional overrides that **completely replace** Raven's built-in review / respond prompt for this repo. Fetched from the **PR base branch** (same security model as `.claude/rules/*.md`) — new prompts only take effect after being merged through a review of their own. To start, copy `prompts/review.md` or `prompts/respond.md` from this repo into `.claude/rules/raven/prompts/` and edit from there. Gated on `RAVEN_RULES_DIR`: setting it to empty string disables prompt overrides as well as rule injection.

All of the above are optional and independent; missing files are skipped silently.

Pushing new commits to a PR branch triggers an incremental re-review (only changed files). Findings from unchanged files are re-validated rather than blindly carried: the model sees them as a "prior findings" block and explicitly drops the ones the push made obsolete (anything it doesn't explicitly drop is kept, so a malformed answer can never erase findings), and the survivors are included in the consolidated verdict. Clicking "re-request review" or adding Raven as a reviewer also triggers a fresh review.

**Review mode**: `RAVEN_REVIEW_MODE` controls engagement and how blocking Raven's verdict is.

- `all` (default) — Raven auto-adds itself to every PR, submits a formal review, and auto-merges PRs where it's the only reviewer.
- `gap` — Raven only auto-adds on PRs with no other reviewer (fill-gap mode); still submits a formal review when engaged.
- `advisory` — Raven posts a non-blocking **Raven Recommendation** comment with inline findings. It does not auto-add itself, does not submit a formal review, and does not auto-merge. Useful for trial deployments or teams that prefer humans to drive merges.

Adding Raven as a reviewer manually always triggers a fresh review (in `advisory` mode that fresh review is still advisory).

**Review output**: `RAVEN_REVIEW_OUTPUT` controls which channels carry the findings — orthogonal to the mode above.

- `both` (default) — the summary comment (verdict + findings list) **and** per-line inline comments.
- `summary` — only the summary comment; no inline comments.
- `inline` — only per-line inline comments; **no summary/recommendation comment**. Findings that can't be anchored to a line (PR-wide notes, ⚠️ coverage-gap markers) get a minimal body listing just those, so nothing is dropped — a clean PR posts only the verdict.

### Conversational follow-up

Developers can @mention Raven in PR comments to ask questions or dispute findings. Raven responds with context-aware answers using the PR diff and conversation history.

- **Immediate acknowledgment** — on Gitea, Raven reacts 👀 to the triggering comment within a second so you know it was picked up, even though the Claude response takes 10–30s.
- **Threaded replies** — on platforms that support comment threads (Bitbucket Data Center), Raven replies inside the thread rather than posting a top-level comment.
- **No re-@mention inside Raven's threads** — any reply in a thread where Raven has already participated triggers a follow-up, so the conversation flows naturally. To require an explicit tag on *every* reply (and never auto-engage on an untagged in-thread reply), set `RAVEN_REPLY_REQUIRE_MENTION`; tags match the bot's account username plus any `RAVEN_MENTION_NAMES`.
- **Active-thread context** — Raven sees the whole conversation it's replying inside (root + replies), not just the last 20 PR-wide comments. The thread root (usually Raven's own original finding) is always preserved through truncation. Cap via `RAVEN_RESPOND_THREAD_TOTAL_CHARS` (default 8000).
- **Line-aware context** — inline diff comments get a line-numbered snippet of the file injected into the prompt, so Raven answers about the exact code without having to parse hunk headers.
- **Verdict re-evaluation** — when the conversation provides substantive new information (e.g. "this isn't a bug because the path is unreachable", "this pattern is intentional convention"), Raven can submit a NEW formal review revising its prior verdict. Auto-merge gates fire on `needs_work → approve` flips and also on retraction-only paths when the prior verdict was already approve (Bitbucket DC "all comments resolved" unblock). Cap via `RAVEN_RESPOND_VERDICT_BODY_CHARS` (default 4000).
- **Finding retraction** — when the discussion explicitly invalidates a specific inline finding, Raven retracts it using the provider's native resolve action: Bitbucket Data Center marks the thread root `threadResolved=true` (the same flag the UI's "Resolve thread" button sets); **Gitea ≥1.24** uses `POST /pulls/comments/{id}/resolve`. On older Gitea instances retraction logs a warning and no-ops; verdict revision and thread context still work normally.
- **Respect user-resolved findings** — when a developer marks one of Raven's inline findings resolved via the UI (Gitea ≥1.24 `/resolve`, BB DC "Resolve thread"), the next incremental re-review drops that finding from carry-forward and from the cache so the consolidated verdict doesn't re-litigate dismissed feedback. Symmetric to AI-driven retraction: that flow is Raven resolving on the AI's behalf; this is Raven respecting the user's direct dismissal. Best-effort: provider API failure falls back to no filtering (review proceeds as before).
- **Faster effort** — comment replies default to `RAVEN_AI_EFFORT_COMMENT=medium`; PR reviews still use max.

## Quick start

```bash
git clone https://github.com/elduty/raven.git
cd raven
cp config.example.env .env
# Edit .env with your values (see Configuration below)
docker compose up -d
```

### Gitea setup

1. **System webhook** (Site Administration -> System Webhooks):
   - URL: `https://raven.yourdomain.com/hook/gitea`
   - Content Type: `application/json`
   - Secret: must match `GITEA_WEBHOOK_SECRET`
   - Events: Pull Request, Push, Issue Comment, Pull Request Comment, Pull Request Review Request

2. **Allowed hosts** in Gitea `app.ini`:
   ```ini
   [webhook]
   ALLOWED_HOST_LIST = raven.yourdomain.com
   ```

3. **Per-repo label** (one-time setup per repo):
   - Name: `raven-reviewed`
   - Color: `#7B68EE` (purple)

4. **Optional**: Add a `CLAUDE.md` at the repo root for project-wide guidance, and/or `.claude/rules/*.md` files for focused review criteria.
   - **Both are read from the PR base branch** (already-merged state). New or updated `CLAUDE.md` / `.claude/rules/*.md` content only takes effect after the PR introducing it has been merged — a PR can't add policy that biases its own review.
   - Both are rendered inside a `<repo_policy>` block in the prompt and marked as **authoritative review policy** that the model must apply. Same trust tier as the built-in prompt template; distinct from the diff / PR-author content which is wrapped as `<untrusted_input>` data.
   - `.claude/rules/raven/prompts/{review,respond}.md` are likewise read from the **PR base branch** and completely replace Raven's built-in prompts for this repo. Same merge-first trust property.

### Bitbucket Data Center setup

See [docs/bitbucket-dc-setup.md](docs/bitbucket-dc-setup.md) for the full guide. Quick summary:

1. Create a service account with project write access
2. Generate a personal access token (Repository Write + Pull Request Write)
3. Set `BITBUCKET_DC_URL`, `BITBUCKET_DC_TOKEN`, `BITBUCKET_DC_WEBHOOK_SECRET`, `BITBUCKET_DC_USERNAME`
4. Create a webhook at repo or project level pointing to `/hook/bitbucket-dc`
5. Enable events: `repo:refs_changed`, `pr:opened`, `pr:from_ref_updated`, `pr:reopened`, `pr:comment:added`, `pr:reviewer:updated`

## Authentication

### Claude Code OAuth

Only required when using the `claude_cli` backend (see [AI backend](#ai-backend) below).

Generate a portable token (~1 year lifetime) on a machine with Claude Code installed:

```bash
claude setup-token
```

Set `CLAUDE_CODE_OAUTH_TOKEN` to the output. The token is a JSON blob containing the access token, refresh token, and scopes. The container's entrypoint writes it to the CLI's credential store on each start.

**Important:** Don't use Claude Code on the source machine after extracting tokens — refresh token rotation will invalidate the copy in Docker.

### Gitea token

Create a personal access token in Gitea with these permissions:
- `repo`: read (to fetch diffs and file contents)
- `issue`: write (to post comments and manage labels)
- `repo`: write (to merge PRs and delete branches)

Consider creating a dedicated service account for Raven rather than using a personal token.

## Configuration

All configuration is via environment variables. See `config.example.env` for the full list with descriptions.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GITEA_WEBHOOK_SECRET` | Gitea | — | HMAC-SHA256 shared secret for Gitea webhooks. Required if using Gitea. |
| `GITEA_URL` | Gitea | — | Base URL of the Gitea instance. |
| `GITEA_TOKEN` | Gitea | — | Gitea personal access token. |
| `CLAUDE_CODE_OAUTH_TOKEN` | claude_cli | — | Claude Code OAuth token (JSON blob from `claude setup-token`, or a bare access token). Required only for the `claude_cli` backend. |
| `CLAUDE_CODE_OAUTH_REFRESH_TOKEN` | No | — | Only needed if `CLAUDE_CODE_OAUTH_TOKEN` is a bare token string (not a JSON blob). Enables automatic token refresh. |
| `RAVEN_AI_BACKEND` | No | auto | Force a backend (`claude_cli` or `openai_compatible`). Omit to auto-detect from credentials. |
| `RAVEN_AI_API_BASE` | openai_compatible | — | Base URL of the OpenAI-compatible endpoint (LiteLLM proxy, vLLM, OpenRouter, OpenAI, etc.). |
| `RAVEN_AI_API_KEY` | openai_compatible | — | API key for the OpenAI-compatible endpoint. |
| `RAVEN_AI_MAX_TOKENS` | No | — | When set, passed as `max_tokens` to the `openai_compatible` backend. Use this to lift the model's default output cap (e.g., GPT-4o defaults to 16k) so long reviews don't truncate mid-JSON. Ignored by `claude_cli`. |
| `RAVEN_AI_PRICES` | No | — | Fallback price table for the cost metric: JSON `{"model":{"input":N,"output":N}}` in USD per 1M tokens. Only used when the backend doesn't self-report cost (`claude_cli` and LiteLLM do, so this is usually unnecessary). Unset → tokens still tracked, cost stays 0. |
| `NOTIFY_CHANNELS` | No | — | JSON array of notification channels (see below). |
| `REVIEW_APPROVE_MAX_SEVERITY` | No | `low` | Approve PRs at or below this severity (`low`, `medium`, `high`). |
| `MERGE_STRATEGY` | No | `squash` | Merge method (`squash`, `merge`, `rebase`). |
| `MAX_DIFF_LINES` | No | `3000` | Max diff lines before splitting review by file. |
| `RAVEN_MAX_FILE_LINES` | No | `500` | Max lines a changed file may have for its full contents to be attached to the review prompt. Larger files are skipped, and the prompt discloses the omission so the model knows its evidence is incomplete. |
| `RAVEN_MAX_FILES` | No | `10` | Max changed files whose full contents are attached to the review prompt. Files beyond the cap are skipped and disclosed in the prompt like line-cap omissions. |
| `RAVEN_CARRIED_REVALIDATION_MAX` | No | `20` | Max carried findings offered to the incremental review for re-validation (top severity first). Overflow findings are carried verbatim — never silently dropped — keeping the prompt bounded on PRs with large finding sets. |
| `CI_WAIT_TIMEOUT` | No | `300` | Seconds to wait for CI before giving up. `0` to skip CI check. |
| `RAVEN_REQUIRE_CI` | No | — | Require CI to register before auto-merging. Off by default: a "no status yet" result counts as "no CI configured" and the PR can merge immediately. Set truthy (`1`/`true`/`yes`) on repos that always run CI, so an unregistered status is treated as pending and Raven waits (up to `CI_WAIT_TIMEOUT`) instead of merging before CI starts. |
| `SKIP_REPOS` | No | — | Comma-separated `owner/repo` list to skip. |
| `SKIP_AUTHORS` | No | — | Comma-separated author names to skip. |
| `RAVEN_AI_MODEL` | No | `claude-opus-4-8` | Model to use for reviews. Backend-agnostic — the string is whatever your active backend (and any proxy) routes. |
| `RAVEN_AI_EFFORT` | No | `max` | Thinking effort on PR reviews (`none`, `low`, `medium`, `high`, `max`). `none` omits `reasoning_effort` from the OpenAI-compatible request entirely — use it when routing to non-reasoning models behind a proxy that doesn't strip the param. |
| `RAVEN_AI_EFFORT_COMMENT` | No | `medium` | Thinking effort on comment replies (`none`, `low`, `medium`, `high`, `max`). Q&A rarely needs max. |
| `RAVEN_AI_TIMEOUT` | No | `1800` | Per-request timeout in seconds (CLI subprocess for `claude_cli`, HTTP call for `openai_compatible`). Sized for the default `max` effort on medium-large diffs; lower it (e.g. `600`) if you route to a faster model or run a lower effort. |
| `RAVEN_AI_RETRY` | No | `1` | Retries for **transient** AI failures (timeout, rate-limit 429, backend 5xx) before giving up and posting a classified failure comment. Usage-cap / auth / unknown errors never retry. `0` disables retry. **Wall-time note:** each retry can add up to `RAVEN_AI_TIMEOUT + RAVEN_AI_RETRY_BACKOFF` seconds — with defaults a single review tops out near ~3603s (2×1800 + 3) instead of 1800s. For a chunked (large-diff) review, retry is per-chunk. |
| `RAVEN_AI_RETRY_BACKOFF` | No | `3` | Fixed backoff in seconds slept between an AI call and its retry. |
| `RAVEN_COMMENT_HISTORY` | No | `20` | Recent comments passed to Claude as conversation context (for @mention replies). |
| `RAVEN_RESPOND_THREAD_TOTAL_CHARS` | No | `8000` | Char cap on the "Active Thread" block when replying to a PR comment. Preserves the thread root (Raven's original finding) + newest replies; drops the middle with a truncation marker. |
| `RAVEN_RESPOND_VERDICT_BODY_CHARS` | No | `4000` | Char cap on the prior-verdict body shown when Raven may revise its verdict. Middle-truncates (keeps first + last paragraphs). |
| `RAVEN_MAX_PR_REPLIES_PER_HOUR` | No | `20` | Circuit breaker against comment-reply loops: max AI replies Raven posts on one PR per rolling hour (bot-authored comments are skipped outright). Guards against another auto-responder driving unbounded paid replies. `0` disables the limit. |
| `RAVEN_MENTION_NAMES` | No | `Raven` | Display name(s) that count as an `@mention` of Raven, in addition to the bot's account username — comma-separated, recognized in **both** modes. Set empty to match only the account username. (`@`-strings inside `` `code` `` spans and email / `user@host` local parts never count.) |
| `RAVEN_REPLY_REQUIRE_MENTION` | No | — | Mention-only mode. When truthy (`1`/`true`/`yes`), Raven replies to a comment **only** if it tags Raven (account username or a `RAVEN_MENTION_NAMES` display name), and never to an untagged in-thread reply. Unset (default) keeps current behaviour: a tag **or** any reply inside a thread Raven is already in triggers a response. |
| `RAVEN_REVIEW_COMMENT_CONTEXT` | No | `20` | Max non-bot PR comments included in the review prompt's "PR Conversation" section. `0` disables the subsection. |
| `RAVEN_REVIEW_PR_CONTEXT_ITEM_CHARS` | No | `4000` | Per-item character cap applied to PR title, description, and each comment body before they're concatenated into the review prompt. Approximate — a truncation marker of ~30-40 chars is appended after the prefix. `0` disables truncation (keep full text). |
| `RAVEN_REVIEW_PR_CONTEXT_TOTAL_CHARS` | No | `16000` | Global budget across the whole PR Context block (title + description + all included comments). Prevents discussion from dwarfing a small diff in the prompt. Comments are added newest-first until the budget is hit. `0` disables the global cap (per-item caps still apply). |
| `RAVEN_RULES_DIR` | No | `.claude/rules` | Repository directory whose top-level `*.md` files are injected into every review prompt as "Repository Rules" context (flat listing — subdirectories are ignored for rules). Also hosts optional prompt overrides at `<RAVEN_RULES_DIR>/raven/prompts/{review,respond}.md`. Set to empty string to disable both rule injection and prompt overrides. |
| `RAVEN_REVIEW_RULES_TOTAL_CHARS` | No | `16000` | Global budget across all rule files concatenated into the prompt. Per-file truncation uses `RAVEN_REVIEW_PR_CONTEXT_ITEM_CHARS`. `0` disables the global cap. |
| `RAVEN_GITEA_AUTO_MERGE` | No | `false` | Gitea-only. Queue the merge and let Gitea wait for CI. BB DC has no equivalent REST flag; its CI enforcement lives in repo-level merge checks, so this setting does nothing on BB DC. Default behaviour (poll CI, then merge) works on every provider. |
| `RAVEN_REVIEW_MODE` | No | `all` | Review engagement: `all` (auto-add + formal review + sole-reviewer auto-merge), `gap` (auto-add only when no other reviewer; formal review), or `advisory` (no auto-add; non-blocking "Raven Recommendation" comment; no auto-merge). Invalid values exit at startup. |
| `RAVEN_REVIEW_OUTPUT` | No | `both` | Review output channels: `both` (summary comment + inline comments), `summary` (summary comment only), or `inline` (inline comments only — no summary/recommendation comment; only findings that can't be anchored to a line get a minimal body so they aren't dropped). Invalid values exit at startup. |
| `RAVEN_LABEL_NAME` | No | `raven-reviewed` | Label name added to reviewed PRs. |
| `RAVEN_CACHE_DIR` | No | `/tmp/raven` | Directory for persistent findings cache. Use a Docker volume in production. |
| `RAVEN_MAX_CACHED_PRS` | No | `200` | Max PR entries in the findings cache (LRU eviction). |
| `RAVEN_METRICS_TOKEN` | No | — | Bearer token gating the `/metrics` endpoint. Unset → `/metrics` returns 404 (disabled). When set, Prometheus must send `Authorization: Bearer <token>`. |
| `RAVEN_MAX_WORKERS` | No | `16` | Main review pool size (webhook dispatch, diff fetch, Claude CLI, review submission). |
| `RAVEN_CI_WAIT_WORKERS` | No | `32` | Dedicated pool for the post-review CI-wait-and-merge phase. Sized larger than the main pool because these tasks spend nearly all their time sleeping. |
| `RAVEN_AI_MAX_CONCURRENT` | No | `4` | Max concurrent in-flight AI calls. For `claude_cli` this caps subprocesses (memory); for `openai_compatible` it caps simultaneous HTTP requests to the proxy. |
| `BITBUCKET_DC_URL` | BB DC | — | Bitbucket Data Center base URL. Enables BB DC provider when set with token + secret + username. |
| `BITBUCKET_DC_TOKEN` | BB DC | — | BB DC personal access token (Repository Write + Pull Request Write). |
| `BITBUCKET_DC_WEBHOOK_SECRET` | BB DC | — | HMAC-SHA256 secret for BB DC webhooks. |
| `BITBUCKET_DC_USERNAME` | BB DC | — | BB DC service account username (slug). Required for identity checks. |

### AI backend

Raven talks to an LLM through one of two pluggable backends. It picks one at startup based on which credentials are present:

| Backend | Selected when | Required env vars |
|---|---|---|
| `claude_cli` | `CLAUDE_CODE_OAUTH_TOKEN` is set (and no OpenAI creds) | `CLAUDE_CODE_OAUTH_TOKEN` |
| `openai_compatible` | `RAVEN_AI_API_BASE` + `RAVEN_AI_API_KEY` both set | `RAVEN_AI_API_BASE`, `RAVEN_AI_API_KEY` |

If both credential sets are present, `openai_compatible` wins. To pin the selection explicitly, set `RAVEN_AI_BACKEND=claude_cli` or `RAVEN_AI_BACKEND=openai_compatible`.

The `openai_compatible` backend talks to any endpoint that speaks OpenAI chat/completions — a LiteLLM proxy, vLLM, OpenRouter, actual OpenAI, etc. The upstream model is whatever the endpoint routes; set `RAVEN_AI_MODEL` to a string that endpoint accepts (e.g. `claude-opus-4-8`, `anthropic/claude-opus-4-8`, or a proxy-defined alias).

`RAVEN_AI_EFFORT` / `RAVEN_AI_EFFORT_COMMENT` map to the backend's reasoning controls: the Claude CLI passes them as `--effort`; the OpenAI-compatible backend translates `max→high`, `high→high`, `medium→medium`, `low→low`, `none→` (parameter omitted). The target model / proxy is expected to translate `reasoning_effort` further if it hosts an Anthropic model (LiteLLM does this natively).

**Non-reasoning models:** if your route lands on a model that doesn't support `reasoning_effort` (most non-reasoning OpenAI models, Llama, Mistral, base Gemini) and the proxy doesn't strip the parameter for you, set `RAVEN_AI_EFFORT=none` and `RAVEN_AI_EFFORT_COMMENT=none` to omit the kwarg entirely.

Example LiteLLM proxy setup:

```bash
export RAVEN_AI_API_BASE=http://proxy.internal:4000
export RAVEN_AI_API_KEY=sk-proxy-key
export RAVEN_AI_MODEL=claude-opus-4-8
# do NOT set CLAUDE_CODE_OAUTH_TOKEN
```

### Notification channels

Notifications are opt-in. Configure via `NOTIFY_CHANNELS` JSON array:

```json
[
  {"type": "webhook", "url": "https://...", "token": "...", "min_severity": "medium"},
  {"type": "slack", "url": "https://hooks.slack.com/services/..."},
  {"type": "webhook", "url": "https://example.com/hook", "token": "..."}
]
```

- `type`: `slack` or `webhook`
- `min_severity`: only notify at or above this level (omit for all)
- `repos`: limit to specific repos (omit for all)

### Auto-merge behaviour

Raven auto-merges a PR when all of these are true:

1. Raven is a listed reviewer of the PR (auto-added or manually added).
2. No other reviewer or requested reviewer is listed — Raven is the only reviewer.
3. Raven submitted an `APPROVED` review with severity ≤ `REVIEW_APPROVE_MAX_SEVERITY`.
4. The consolidated verdict (including re-validated carried findings from unchanged files) meets the threshold.
5. No coverage gap — every chunk of the diff was actually reviewed (oversized/failed chunks force `needs_work` and block both merge paths until re-reviewed).
6. CI passes, or CI_WAIT_TIMEOUT is configured to skip CI.
7. Head SHA hasn't changed since CI was checked (force-push protection).

No other merge path exists. Raven never merges a PR that has any other reviewer or requested reviewer listed — that case is left for humans.

## Review format

Raven submits formal reviews (APPROVED or REQUEST_CHANGES on Gitea, approve/needs-work on BB DC) with inline comments on specific diff lines:

```
🦅 **Raven Review**

**🔴 HIGH** — Unescaped user input passed to raw SQL query in auth handler

**Findings:**
- 🔴 [high] `db.py:execute()` — user input concatenated directly into SQL string
- 🟠 [medium] `auth.py:login()` — no rate limiting on failed attempts

*Reviewed by Raven · claude-opus-4-8 · effort max · 2026-03-22 00:41 UTC*
```

Each finding with a file and line number is also posted as an inline comment on the exact diff line.

On **Bitbucket DC** — which has no single-call review API — the summary comment is posted **after** the inline anchors so it sorts on top of them in the PR activity. If the summary post fails, the inline anchors that already posted are rolled back (deleted) so a retry can't duplicate them. Gitea posts the whole review in one call, so ordering there is whatever the platform renders.

## Customising the review prompt

Edit `prompts/review.md` to tune review quality. The prompt is loaded at container startup — restart the container to pick up changes, no rebuild needed.

## Architecture

```
raven/
├── server.py              Flask app, webhook routing, async dispatch, merge/notify logic
├── providers/
│   ├── __init__.py        GitProvider ABC + provider registry
│   ├── gitea.py           GiteaProvider — Gitea REST API + webhook parsing
│   └── bitbucket_dc.py    BitbucketDCProvider — BB Data Center REST API + webhook parsing
├── ai/
│   ├── __init__.py        Backend registry + auto-selection (get_backend, _select_backend)
│   ├── base.py            AIBackend ABC (complete, shutdown)
│   ├── claude_cli.py      ClaudeCLIBackend — wraps the Claude Code CLI subprocess
│   ├── openai_compatible.py  OpenAICompatibleBackend — chat/completions via openai SDK
│   └── pricing.py         Fallback model price table (RAVEN_AI_PRICES) for cost metric
├── reviewer.py            Diff chunking, JSON response parsing, routes through AIBackend
├── notifier.py            Channel-based notifications (Slack, webhook)
├── metrics.py             In-memory counters exposed via /metrics (Prometheus format)
prompts/
├── review.md              Review prompt template
├── respond.md             Conversational response prompt template
├── audit.md               Standalone prompt for operator-driven manual audits (not code-loaded)
observability/             Opt-in Prometheus + Grafana bundle (--profile observability)
├── prometheus/            Scrape config (bearer via boot-written token file)
└── grafana/               Provisioned datasource + dashboards (raven.json)
entrypoint.sh              Writes OAuth credentials, updates Claude CLI on start + daily
```

### Request flow

1. The git platform sends a webhook to `/hook/<provider>` (e.g. `/hook/gitea`, `/hook/bitbucket-dc`)
2. The provider validates the HMAC-SHA256 signature, returns `200 Accepted` immediately
3. For push events: looks up the open PR for the branch; if found, triggers a re-review
4. For `pull_request_review_request` events: re-reviews if Raven is the requested reviewer
5. For `issue_comment` / `pull_request_comment` events: responds if @mentioned
6. A background thread processes the PR:
   - Checks the in-memory dedup cache; skips if the same PR was dispatched within 30s
   - Fetches the PR diff via the provider's API
   - Strips lockfiles and binary files
   - Compares against cached diff hashes for incremental review (only changed files)
   - Fetches full file contents for all PR files (cross-file context)
   - Sends the prompt, diff, and file contents to the configured AI backend (`claude_cli` pipes through stdin; `openai_compatible` posts via HTTP)
   - Parses the JSON response with file/line locations per finding
   - If parse failed: posts warning comment, sends notification, stops (no merge)
   - Carries forward findings from unchanged files, recomputes consolidated verdict
   - Dismisses previous Raven reviews (after submitting new one)
   - Submits formal review (APPROVED/REQUEST_CHANGES) with inline comments
   - Adds the `raven-reviewed` label
   - Caches findings to disk (persists across restarts, invalidated on model/prompt change)
   - Auto-merge decision: when Raven is sole reviewer and consolidated verdict approves
   - On a transient AI failure (timeout / rate-limit / backend 5xx): retries once (configurable) before giving up
   - On unhandled error: clears dedup entry, posts a **classified** failure comment naming the cause + next step

### Safety gates

- **Startup assertion**: refuses to start without at least one complete provider config (Gitea or Bitbucket DC)
- **Empty diff guard**: diffs empty after stripping lockfiles/binaries are not sent to Claude
- **Parse error guard**: unparseable review output blocks merge and sends notification
- **Review-before-merge**: if submitting the review fails, the PR is not merged
- **Sole reviewer gate**: only merges when Raven is the only reviewer (no human reviews or requests)
- **CI gate**: waits for CI to pass; failure/timeout blocks merge with notification
- **Force-push protection**: verifies head SHA hasn't changed before merging
- **PR dedup**: 30 s in-memory cache keyed on `(repo, pr_number, head_sha)` absorbs webhook redelivery storms without dropping legitimate new pushes (a new SHA is treated as a fresh event, not a duplicate)
- **Consolidated incremental reviews**: findings from unchanged files are re-validated by the model (explicit-drop contract — a malformed answer keeps everything) and included in the verdict; drops apply to the cache only after the review posts, and resolve their inline threads
- **Coverage-gap guard**: oversized or failed review chunks set a sticky per-file gap — the verdict is forced to `needs_work` and both merge paths refuse auto-merge until the gap files are re-reviewed cleanly (or leave the PR)
- **Stale review dismissal**: previous Raven reviews dismissed after new review is posted
- **Cache invalidation**: findings cache wiped automatically on model or prompt change
- **Classified failure handling**: AI failures are classified (`timeout` / `rate_limit` / `backend_5xx` / `usage_limit` / `auth` / `unknown`); transient classes retry once (`RAVEN_AI_RETRY`) and the posted comment names the cause and the actionable next step (e.g. "raise `RAVEN_AI_TIMEOUT`", "resets on the next push") instead of an opaque "internal error". Counted in `raven_review_failures_total{reason,repo}`. Comment text is static per-reason — no exception detail is interpolated, so credential-bearing error strings can't leak. Dedup entry is cleared so a webhook retry can re-attempt.
- **Always 200**: all webhook responses return HTTP 200 to prevent retry loops
- **Graceful shutdown**: on SIGTERM, queued reviews and CI-wait tasks are cancelled and in-flight Claude CLI subprocesses are SIGTERMed so gunicorn's graceful-shutdown window isn't consumed by discarded work

### Metrics

`/metrics` endpoint exposes Prometheus-format counters:
- `raven_reviews_total{severity,repo}` — review count
- `raven_review_duration_seconds{repo}` — Claude CLI call duration
- `raven_merges_total{repo}` — successful auto-merges
- `raven_errors_total{type,repo}` — errors by type
- `raven_review_failures_total{reason,repo}` — classified review failures (`reason` = `timeout`/`rate_limit`/`backend_5xx`/`usage_limit`/`auth`/`unknown`)
- `raven_ci_failures_total{repo}` — CI failures
- `raven_responses_total{repo}` — comment responses
- `raven_reviews_skipped_total{reason,repo}` — skipped reviews
- `raven_verdict_revisions_total{repo,from,to}` — comment-driven verdict revisions
- `raven_retractions_total{repo,result}` — comment-driven finding retractions
- `raven_response_parse_errors_total{repo}` — AI JSON parse failures
- `raven_revision_submit_errors_total{repo}` — verdict-revision `submit_review` failures
- `raven_ai_calls_total{backend,model,repo}` — AI completion calls
- `raven_ai_tokens_total{backend,model,repo,kind}` — tokens consumed (`kind` = `input`/`output`)
- `raven_ai_cost_usd_total{backend,model,repo}` — AI cost in USD

**Token & cost metrics**: `raven_ai_tokens_total` and `raven_ai_cost_usd_total` track AI spend, broken down by backend, model, and repo. Cost is taken from the provider when it reports one — `claude_cli` exposes `total_cost_usd`, LiteLLM proxies return an `x-litellm-response-cost` header — so for those setups cost is exact and needs no configuration. For OpenAI-compatible endpoints that report no cost, set `RAVEN_AI_PRICES` (a JSON `model → {input, output}` $/1M-token table) to estimate it; without it, tokens are still tracked and cost stays 0.

**Authentication**: the `/metrics` endpoint requires a bearer token. Set `RAVEN_METRICS_TOKEN` to enable; leave unset and the endpoint returns 404 (it doesn't advertise itself). Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: raven
    static_configs:
      - targets: ['raven:8080']
    authorization:
      credentials_file: /etc/prom/raven_metrics_token
```

`/healthz` stays open for load balancers — no auth required.

**Dashboards & long-term storage**: an opt-in observability bundle ships Prometheus (5-year retention) and a provisioned Grafana dashboard as a Compose profile:

```bash
# Set RAVEN_METRICS_TOKEN and GRAFANA_ADMIN_PASSWORD in .env, then:
docker compose --profile observability up -d
```

Grafana comes up at `http://localhost:${GRAFANA_PORT:-3000}` — **bound to loopback by default** (`GRAFANA_BIND=127.0.0.1`) so the `admin` fallback password isn't exposed off-host (login `admin` / `$GRAFANA_ADMIN_PASSWORD`). Any non-loopback bind (`GRAFANA_BIND=0.0.0.0`, `::`, or a routable IP) **requires** a non-default `GRAFANA_ADMIN_PASSWORD` — the stack fails fast (the Grafana container refuses to start) otherwise. Note that Grafana serves **plain HTTP**, so a direct non-loopback bind sends the admin password and session cookies in cleartext; it's appropriate only on a trusted network or via a localhost tunnel. For real off-host access, front Grafana with a **TLS-terminating reverse proxy** (and still set a strong password). Dashboard **Raven — Review, Reliability & Cost** loads automatically: throughput by severity, merges/CI failures, average review duration, errors, comment activity, and token/cost breakdowns by model and repo — all filterable by a `repo` template variable. Plain `docker compose up` (no profile) runs Raven alone. See [`observability/README.md`](observability/README.md) for token-file handling, scraping a second instance, retention tuning, and licensing.

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

~926 tests across 15 test files covering webhook handling (BB DC `pr:comment:added`/`:edited` version-aware dedup, `pr:reviewer:approved`/`:changes_requested` parity with Gitea, activities-endpoint pagination cap with WARNING), review parsing, inline comments, notification dispatch, metrics with bearer-token auth, SHA-aware PR dedup, incremental reviews, findings cache persistence (`CacheEntry` dataclass with verdict + summary), conversational follow-up (mention, thread, reply-in-Raven-thread, active-thread context with `[id=N]` + `[YOU]` markers, BB DC activities-based thread discovery, line-windowed truncation, code-snippet injection), comment-driven verdict revision and finding retraction (with atomic race guards + Raven-authorship filter + auto-flip backstop + in-memory thread-root walk-up + same-thread dedupe), user-resolved findings dropped from carry-forward (BB DC threadResolved + state=RESOLVED, Gitea ≥1.24 resolver-field), two-tier prompt trust model (`<repo_policy>` for CLAUDE.md + rules at base ref vs `<untrusted_input>` for diff + comments), chunked-review consolidation pass that re-applies repo policy to aggregated findings, three-mode review engagement (`all` / `gap` / `advisory`), three-way review output channels (`both` / `summary` / `inline`), Claude subprocess tracking and graceful-shutdown termination, PR conversation context in reviews, repo-supplied rules injection, per-repo prompt overrides, both git providers, the AI backend interface (claude_cli + openai_compatible), backend auto-selection, and the full PR flow including CI gating.

## CI

CI runs `pytest` on every push and pull request via GitHub Actions (also works with Gitea Actions). See `.github/workflows/test.yml`.

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/elduty/raven.git
cd raven
pip install -r requirements-dev.txt
pytest tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).
