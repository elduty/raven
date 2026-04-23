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
      -> send to Claude Opus via stdin (extended thinking, --effort max)
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

- **`CLAUDE.md`** at the repo root — free-form project guidance. Injected as "Repository Context".
- **`.claude/rules/*.md`** — one file per rule category (e.g. `security.md`, `style.md`, `testing.md`). Injected as "Repository Rules" and the model is instructed to apply them as review criteria. Flat directory, `.md` only. **Fetched from the PR's base branch** (already-merged state) so a PR can't add or modify rules to bias its own review — new rules only take effect after they've been merged through a review of their own.

- **`.claude/rules/raven/prompts/review.md`** and **`.claude/rules/raven/prompts/respond.md`** — optional overrides that **completely replace** Raven's built-in review / respond prompt for this repo. Fetched from the **PR base branch** (same security model as `.claude/rules/*.md`) — new prompts only take effect after being merged through a review of their own. To start, copy `prompts/review.md` or `prompts/respond.md` from this repo into `.claude/rules/raven/prompts/` and edit from there. Gated on `RAVEN_RULES_DIR`: setting it to empty string disables prompt overrides as well as rule injection.

All of the above are optional and independent; missing files are skipped silently.

Pushing new commits to a PR branch triggers an incremental re-review (only changed files). Findings from unchanged files are carried forward and included in the consolidated verdict. Clicking "re-request review" or adding Raven as a reviewer also triggers a fresh review.

**Reviewer assignment**: by default Raven auto-adds itself to every PR it receives a webhook for and reviews it. It will auto-merge only PRs where it's the only reviewer; PRs with humans reviewing are reviewed but left for humans to merge. Set `RAVEN_REVIEW_ALL_PRS=false` for fill-gap mode — Raven only auto-adds on PRs with no other reviewer. Either way, adding Raven as a reviewer manually always triggers a fresh review.

### Conversational follow-up

Developers can @mention Raven in PR comments to ask questions or dispute findings. Raven responds with context-aware answers using the PR diff and conversation history.

- **Immediate acknowledgment** — on Gitea, Raven reacts 👀 to the triggering comment within a second so you know it was picked up, even though the Claude response takes 10–30s.
- **Threaded replies** — on platforms that support comment threads (Bitbucket Data Center), Raven replies inside the thread rather than posting a top-level comment.
- **No re-@mention inside Raven's threads** — any reply in a thread where Raven has already participated triggers a follow-up, so the conversation flows naturally.
- **Line-aware context** — inline diff comments get a line-numbered snippet of the file injected into the prompt, so Raven answers about the exact code without having to parse hunk headers.
- **Faster effort** — comment replies default to `CLAUDE_EFFORT_COMMENT=medium`; PR reviews still use max.

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
   - URL: `https://raven.yourdomain.com/hook/gitea` (or `/hook` for backward compat)
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
   - `CLAUDE.md` is read from the **PR head** — documentation changes land atomically with the code they describe.
   - `.claude/rules/*.md` are read from the **PR base branch** (already-merged state) so a PR can't add or modify its own review criteria. New or updated rules only take effect after they've been merged.
   - `.claude/rules/raven/prompts/{review,respond}.md` are read from the **PR base branch** and completely replace Raven's built-in prompts for this repo. Same merge-first security property as rules.

### Bitbucket Data Center setup

See [docs/bitbucket-dc-setup.md](docs/bitbucket-dc-setup.md) for the full guide. Quick summary:

1. Create a service account with project write access
2. Generate a personal access token (Repository Write + Pull Request Write)
3. Set `BITBUCKET_DC_URL`, `BITBUCKET_DC_TOKEN`, `BITBUCKET_DC_WEBHOOK_SECRET`, `BITBUCKET_DC_USERNAME`
4. Create a webhook at repo or project level pointing to `/hook/bitbucket-dc`
5. Enable events: `repo:refs_changed`, `pr:opened`, `pr:from_ref_updated`, `pr:reopened`, `pr:comment:added`, `pr:reviewer:updated`

## Authentication

### Claude Code OAuth

Raven uses the Claude Code CLI with OAuth authentication.

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
| `RAVEN_WEBHOOK_SECRET` | — | — | Deprecated alias for `GITEA_WEBHOOK_SECRET`. Still works for backward compat. |
| `GITEA_URL` | Gitea | — | Base URL of the Gitea instance. |
| `GITEA_TOKEN` | Gitea | — | Gitea personal access token. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Yes | — | Claude Code OAuth token (JSON blob from `claude setup-token`, or a bare access token). |
| `CLAUDE_CODE_OAUTH_REFRESH_TOKEN` | No | — | Only needed if `CLAUDE_CODE_OAUTH_TOKEN` is a bare token string (not a JSON blob). Enables automatic token refresh. |
| `NOTIFY_CHANNELS` | No | — | JSON array of notification channels (see below). |
| `REVIEW_APPROVE_MAX_SEVERITY` | No | `low` | Approve PRs at or below this severity (`low`, `medium`, `high`). |
| `MERGE_STRATEGY` | No | `squash` | Merge method (`squash`, `merge`, `rebase`). |
| `MAX_DIFF_LINES` | No | `3000` | Max diff lines before splitting review by file. |
| `CI_WAIT_TIMEOUT` | No | `300` | Seconds to wait for CI before giving up. `0` to skip CI check. |
| `SKIP_REPOS` | No | — | Comma-separated `owner/repo` list to skip. |
| `SKIP_AUTHORS` | No | — | Comma-separated author names to skip. |
| `CLAUDE_MODEL` | No | `claude-opus-4-6` | Model to use for reviews. |
| `CLAUDE_EFFORT` | No | `max` | Thinking effort on PR reviews (`low`, `medium`, `high`, `max`). |
| `CLAUDE_EFFORT_COMMENT` | No | `medium` | Thinking effort on comment replies. Q&A rarely needs max. |
| `CLAUDE_TIMEOUT` | No | `600` | Timeout in seconds for each Claude CLI invocation. |
| `RAVEN_COMMENT_HISTORY` | No | `20` | Recent comments passed to Claude as conversation context (for @mention replies). |
| `RAVEN_REVIEW_COMMENT_CONTEXT` | No | `20` | Max non-bot PR comments included in the review prompt's "PR Conversation" section. `0` disables the subsection. |
| `RAVEN_REVIEW_PR_CONTEXT_ITEM_CHARS` | No | `4000` | Per-item character cap applied to PR title, description, and each comment body before they're concatenated into the review prompt. Approximate — a truncation marker of ~30-40 chars is appended after the prefix. `0` disables truncation (keep full text). |
| `RAVEN_REVIEW_PR_CONTEXT_TOTAL_CHARS` | No | `16000` | Global budget across the whole PR Context block (title + description + all included comments). Prevents discussion from dwarfing a small diff in the prompt. Comments are added newest-first until the budget is hit. `0` disables the global cap (per-item caps still apply). |
| `RAVEN_RULES_DIR` | No | `.claude/rules` | Repository directory whose top-level `*.md` files are injected into every review prompt as "Repository Rules" context (flat listing — subdirectories are ignored for rules). Also hosts optional prompt overrides at `<RAVEN_RULES_DIR>/raven/prompts/{review,respond}.md`. Set to empty string to disable both rule injection and prompt overrides. |
| `RAVEN_REVIEW_RULES_TOTAL_CHARS` | No | `16000` | Global budget across all rule files concatenated into the prompt. Per-file truncation uses `RAVEN_REVIEW_PR_CONTEXT_ITEM_CHARS`. `0` disables the global cap. |
| `RAVEN_GITEA_AUTO_MERGE` | No | `false` | Gitea-only. Queue the merge and let Gitea wait for CI. BB DC has no equivalent REST flag; its CI enforcement lives in repo-level merge checks, so this setting does nothing on BB DC. Default behaviour (poll CI, then merge) works on every provider. |
| `RAVEN_REVIEW_ALL_PRS` | No | `true` | Whether Raven auto-adds itself to every PR (default) or only to PRs with no other reviewer. Set to `false` (or `0` / `no`) for fill-gap behaviour — Raven stays out of human-reviewed PRs. |
| `RAVEN_LABEL_NAME` | No | `raven-reviewed` | Label name added to reviewed PRs. |
| `RAVEN_CACHE_DIR` | No | `/tmp/raven` | Directory for persistent findings cache. Use a Docker volume in production. |
| `RAVEN_MAX_CACHED_PRS` | No | `200` | Max PR entries in the findings cache (LRU eviction). |
| `RAVEN_MAX_WORKERS` | No | `16` | Main review pool size (webhook dispatch, diff fetch, Claude CLI, review submission). |
| `RAVEN_CI_WAIT_WORKERS` | No | `32` | Dedicated pool for the post-review CI-wait-and-merge phase. Sized larger than the main pool because these tasks spend nearly all their time sleeping. |
| `RAVEN_MAX_CONCURRENT_CLAUDE` | No | `4` | Max concurrent Claude CLI subprocesses. Limits memory usage. |
| `BITBUCKET_DC_URL` | BB DC | — | Bitbucket Data Center base URL. Enables BB DC provider when set with token + secret + username. |
| `BITBUCKET_DC_TOKEN` | BB DC | — | BB DC personal access token (Repository Write + Pull Request Write). |
| `BITBUCKET_DC_WEBHOOK_SECRET` | BB DC | — | HMAC-SHA256 secret for BB DC webhooks. |
| `BITBUCKET_DC_USERNAME` | BB DC | — | BB DC service account username (slug). Required for identity checks. |

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
4. The consolidated verdict (including carried findings from unchanged files) meets the threshold.
5. CI passes, or CI_WAIT_TIMEOUT is configured to skip CI.
6. Head SHA hasn't changed since CI was checked (force-push protection).

No other merge path exists. Raven never merges a PR that has any other reviewer or requested reviewer listed — that case is left for humans.

## Review format

Raven submits formal reviews (APPROVED or REQUEST_CHANGES on Gitea, approve/needs-work on BB DC) with inline comments on specific diff lines:

```
🦅 **Raven Review**

**🔴 HIGH** — Unescaped user input passed to raw SQL query in auth handler

**Findings:**
- 🔴 [high] `db.py:execute()` — user input concatenated directly into SQL string
- 🟡 [medium] `auth.py:login()` — no rate limiting on failed attempts

*Reviewed by Raven · 2026-03-22 00:41 UTC*
```

Each finding with a file and line number is also posted as an inline comment on the exact diff line.

## Customising the review prompt

Edit `prompts/review.md` to tune review quality. The prompt is loaded at container startup — restart the container to pick up changes, no rebuild needed.

## Architecture

```
raven/
├── server.py           Flask app, webhook routing, async dispatch, merge/notify logic
├── providers/
│   ├── __init__.py     GitProvider ABC + provider registry
│   ├── gitea.py        GiteaProvider — Gitea REST API + webhook parsing
│   └── bitbucket_dc.py BitbucketDCProvider — BB Data Center REST API + webhook parsing
├── reviewer.py         Claude CLI invocation, diff chunking, JSON response parsing
├── notifier.py         Channel-based notifications (Slack, webhook)
├── metrics.py          In-memory counters exposed via /metrics (Prometheus format)
prompts/
├── review.md           Review prompt template
├── respond.md          Conversational response prompt template
├── audit.md            Full technical audit prompt
entrypoint.sh           Writes OAuth credentials, updates Claude CLI on start + daily
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
   - Pipes the prompt, diff, and file contents to Claude CLI via stdin
   - Parses the JSON response with file/line locations per finding
   - If parse failed: posts warning comment, sends notification, stops (no merge)
   - Carries forward findings from unchanged files, recomputes consolidated verdict
   - Dismisses previous Raven reviews (after submitting new one)
   - Submits formal review (APPROVED/REQUEST_CHANGES) with inline comments
   - Adds the `raven-reviewed` label
   - Caches findings to disk (persists across restarts, invalidated on model/prompt change)
   - Auto-merge decision: when Raven is sole reviewer and consolidated verdict approves
   - On unhandled error: clears dedup entry, posts "internal error" comment

### Safety gates

- **Startup assertion**: refuses to start without at least one complete provider config (Gitea or Bitbucket DC)
- **Empty diff guard**: diffs empty after stripping lockfiles/binaries are not sent to Claude
- **Parse error guard**: unparseable review output blocks merge and sends notification
- **Review-before-merge**: if submitting the review fails, the PR is not merged
- **Sole reviewer gate**: only merges when Raven is the only reviewer (no human reviews or requests)
- **CI gate**: waits for CI to pass; failure/timeout blocks merge with notification
- **Force-push protection**: verifies head SHA hasn't changed before merging
- **PR dedup**: 30s in-memory cache prevents concurrent reviews
- **Consolidated incremental reviews**: carried findings from unchanged files included in verdict
- **Stale review dismissal**: previous Raven reviews dismissed after new review is posted
- **Cache invalidation**: findings cache wiped automatically on model or prompt change
- **Error visibility**: unhandled exceptions post "internal error" comment and clear dedup entry
- **Always 200**: all webhook responses return HTTP 200 to prevent retry loops
- **Graceful shutdown**: on SIGTERM, queued reviews and CI-wait tasks are cancelled and in-flight Claude CLI subprocesses are SIGTERMed so gunicorn's graceful-shutdown window isn't consumed by discarded work

### Metrics

`/metrics` endpoint exposes Prometheus-format counters:
- `raven_reviews_total{severity,repo}` — review count
- `raven_review_duration_seconds{repo}` — Claude CLI call duration
- `raven_merges_total{repo}` — successful auto-merges
- `raven_errors_total{type,repo}` — errors by type
- `raven_ci_failures_total{repo}` — CI failures
- `raven_responses_total{repo}` — comment responses
- `raven_reviews_skipped_total{reason,repo}` — skipped reviews

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

450 tests across 6 test files covering webhook handling, review parsing, inline comments, notification dispatch, metrics, PR dedup, incremental reviews, findings cache persistence, conversational follow-up (mention, thread, reply-in-Raven-thread, line-windowed truncation, code-snippet injection), Claude subprocess tracking and graceful-shutdown termination, PR conversation context in reviews, repo-supplied rules injection, both git providers, and the full PR flow including CI gating.

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
