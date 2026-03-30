# Bitbucket Data Center Setup

Connect Raven to Bitbucket Data Center so every pull request gets an automated code review.

## Prerequisites

- **Bitbucket Data Center 7.x or later** (any version with REST API v1 and webhook support)
- **Network access**: Raven must reach the BB DC base URL over HTTPS (or HTTP if internal). BB DC must reach Raven's `/hook/bitbucket-dc` endpoint to deliver webhooks.
- **A dedicated service account** with access to the projects/repos you want reviewed
- **Claude Code OAuth token** already configured (see main README)

## 1. Create a service account

A dedicated account keeps Raven's actions separate from human users and makes it easy to identify bot comments.

1. Go to **Administration > Users > Create User**
2. Create a user (e.g. `raven-bot`)
3. Grant the account **project-level access** to each project you want reviewed:
   - Go to **Project Settings > Permissions**
   - Add `raven-bot` with **Write** access (needed for posting comments, approving PRs, and merging)
4. If you want Raven to review repos across multiple projects, repeat for each project

The account needs these effective permissions:
- **PROJECT_READ** — list repos, fetch PR metadata
- **REPO_READ** — fetch diffs and file contents
- **REPO_WRITE** — merge PRs, delete source branches
- **PR_WRITE** — post comments, approve/set needs-work, post inline comments

Write access at the project level covers all of these.

## 2. Generate a personal access token

1. Log in as `raven-bot`
2. Go to **Account > HTTP access tokens** (or **Manage account > Personal access tokens** on older versions)
3. Click **Create token**
4. Name: `raven` (or whatever you prefer)
5. Permissions: **Repository Write** and **Pull Request Write**
6. Click **Create**
7. Copy the token immediately — you won't see it again

This token becomes the `BITBUCKET_DC_TOKEN` env var.

## 3. Find your username

Raven needs the exact BB DC username (slug) because the Data Center API has no "whoami" endpoint.

The username is the **slug** shown in the account URL:
```
https://bitbucket.yourcompany.com/users/raven-bot
                                       ^^^^^^^^^
```

This value goes into `BITBUCKET_DC_USERNAME`. It must match exactly — it's used to identify Raven's own comments (to avoid responding to itself) and to set participant status on PRs.

## 4. Configure Raven

Add the Bitbucket DC env vars alongside your existing config. Raven supports multiple providers simultaneously — you can run Gitea and BB DC from the same instance.

### Required env vars

| Variable | Description |
|---|---|
| `BITBUCKET_DC_URL` | Base URL of your BB DC instance (e.g. `https://bitbucket.yourcompany.com`) |
| `BITBUCKET_DC_TOKEN` | Personal access token from step 2 |
| `BITBUCKET_DC_WEBHOOK_SECRET` | HMAC secret you'll configure in the webhook (step 5) |
| `BITBUCKET_DC_USERNAME` | Username (slug) of the service account |

### docker-compose.yml

Add the four `BITBUCKET_DC_*` variables to your existing `.env` file. The project's `docker-compose.yml` already passes them through — no changes to the compose file are needed.

If you're starting fresh, see the `docker-compose.yml` and `config.example.env` in the repo root.

Or add the four `BITBUCKET_DC_*` vars to your existing `.env` file if you already run Raven for Gitea.

### Generate a webhook secret

Pick a random string (32+ characters). Use the same value in `BITBUCKET_DC_WEBHOOK_SECRET` and in the webhook configuration in step 5.

```bash
openssl rand -hex 32
```

## 5. Create a webhook in Bitbucket DC

Webhooks can be created at the **repository level** or **project level**. Project-level webhooks fire for all repos in the project.

### Repository-level webhook

1. Go to **Repository Settings > Webhooks**
2. Click **Create webhook**
3. Fill in:
   - **Name**: `Raven`
   - **URL**: `https://raven.yourdomain.com/hook/bitbucket-dc`
   - **Secret**: the same value as `BITBUCKET_DC_WEBHOOK_SECRET`
4. Enable these events:
   - **Repository: Push** (`repo:refs_changed`)
   - **Pull Request: Opened** (`pr:opened`)
   - **Pull Request: Source branch updated** (`pr:from_ref_updated`)
   - **Pull Request: Reopened** (`pr:reopened`)
   - **Pull Request: Comment added** (`pr:comment:added`)
   - **Pull Request: Reviewer updated** (`pr:reviewer:updated`)
5. Click **Save**

### Project-level webhook

Same steps, but navigate to **Project Settings > Webhooks** instead. This covers all repos in the project without creating individual webhooks.

### Events summary

| BB DC event | What Raven does |
|---|---|
| `repo:refs_changed` | Push to PR branch triggers re-review |
| `pr:opened` | New PR triggers review |
| `pr:from_ref_updated` | New commits pushed triggers re-review |
| `pr:reopened` | Reopened PR triggers review |
| `pr:comment:added` | @mention triggers conversational response |
| `pr:reviewer:updated` | Adding Raven as reviewer triggers review |

## 6. Verify

### Check Raven is running

```bash
curl https://raven.yourdomain.com/healthz
# {"status": "ok"}
```

### Check provider registered

Look at the Raven startup logs. You should see the provider register without errors:

```
INFO raven.server — Registered provider: bitbucket-dc
```

If the `BITBUCKET_DC_USERNAME` is missing, startup will succeed but reviews will fail at runtime with:

```
RuntimeError: BitbucketDCProvider requires 'username' at init
```

### Test with a real PR

1. Create a branch, make a small change, push it
2. Open a pull request in BB DC
3. Check Raven logs — you should see the webhook arrive and a review get dispatched
4. The PR should receive a comment with the review findings

### Test @mention responses

Comment on the PR with `@raven-bot what does this function do?` (use the actual username). Raven should reply within a few seconds.

## 7. Optional: per-repo CLAUDE.md

Add a `CLAUDE.md` file to the root of any repository to give Raven codebase context. This file is fetched on every review and passed to Claude alongside the diff.

Use it for:
- Project-specific coding standards
- Architecture notes that help the reviewer understand the codebase
- Known patterns or conventions to check for
- Files or directories to pay special attention to

Example:

```markdown
# Project context

Python 3.12, FastAPI backend, PostgreSQL. All database access through SQLAlchemy ORM.

## Conventions
- All API endpoints return JSON with `{"data": ..., "error": ...}` envelope
- Errors raise HTTPException, never return error dicts directly
- Tests use pytest with factory_boy fixtures

## Security
- All user input validated with Pydantic models before reaching handlers
- No raw SQL — always use ORM queries
```

## 8. Troubleshooting

### Webhook signature mismatch (403)

```
WARNING raven.providers.bitbucket_dc — Webhook signature mismatch
```

The `BITBUCKET_DC_WEBHOOK_SECRET` in Raven doesn't match the secret configured in the BB DC webhook. They must be identical — copy-paste to avoid whitespace issues.

BB DC sends the signature in the `X-Hub-Signature` header with a `sha256=` prefix. Raven handles this automatically.

### 403 on approve or merge

The service account lacks write permissions on the repository. Check:
- Project-level permissions: `raven-bot` needs **Write** access
- Repository-level overrides: make sure the repo hasn't restricted permissions below the project default
- The token has **Repository Write** and **Pull Request Write** scopes

### Merge fails with version conflict

```
Failed to merge PR #42: 409
```

BB DC requires a `version` parameter on merge to prevent race conditions. Raven fetches the current version before merging, but if someone else merges or updates the PR in between, the version becomes stale. This is expected — Raven will log the error and move on.

### "Empty diff after stripping lockfiles/binaries"

The PR only contains lockfiles (e.g. `package-lock.json`, `yarn.lock`) or binary files. Raven strips these before review. This is normal — no action needed.

### Comments not triggering responses

Raven only responds to comments that @mention the bot username. Check:
- The mention matches `BITBUCKET_DC_USERNAME` exactly (case-insensitive)
- The comment event (`pr:comment:added`) is enabled in the webhook
- The 30-second per-PR cooldown hasn't been hit

### Push events not triggering re-reviews

Push events only trigger re-reviews when there's an open PR for the branch. Raven looks up the PR via the BB DC API. If the push is to a branch without a PR, it's silently skipped. Also check:
- The `repo:refs_changed` event is enabled in the webhook
- The push is to a branch, not a tag (tag pushes are ignored)

### Raven reviews its own PRs

Add the service account username to `SKIP_AUTHORS`:

```
SKIP_AUTHORS=raven-bot
```

### No labels on PRs

This is expected. Bitbucket Data Center has no label system. The `add_label_to_pr` call is a no-op.

### Needs-work status not showing

When Raven requests changes, it sets the participant status to `NEEDS_WORK` via the participants API. This shows as a "needs work" indicator on the PR in BB DC. If it's not appearing, verify the username in `BITBUCKET_DC_USERNAME` matches the authenticated user exactly.
