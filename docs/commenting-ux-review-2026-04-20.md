# Commenting UX Review — 2026-04-20

End-to-end review of Raven's PR-comment flow: webhook handler, `_process_comment`, `respond_to_comment`, provider implementations, and the `respond.md` prompt. Findings ranked by user-visible impact.

---

## High-impact

### 1. Silent cooldown drops
`_check_comment_cooldown` enforces 30s **per PR**, not per user or per comment. If two people comment within 30s, the second is silently dropped with no signal. The user sees Raven answer someone else, then gets nothing. Feels broken.

The `_claude_semaphore` already limits concurrent Claude calls, and per-comment dedup handles duplicate webhooks — the cooldown is redundant for its original purpose (anti-spam) and costs real UX.

**Fix:** drop the cooldown entirely (or shorten to ~5s as an anti-loop floor).

### 2. Deep-thread replies fail the "reply in Raven's thread" check
BB DC thread replies carry `commentParentId = thread_root`, not the comment you clicked "Reply" on. So if Raven **replied to** someone's top-level comment and the user replies to Raven's reply, `parent_comment_id` points at the user's own original comment, not Raven's — and the auto-respond-without-mention logic fails.

Currently it only works when Raven *started* the thread (review body, inline diff comment, top-level PR comment).

**Fix:** fetch the parent comment and scan its `comments[]` (child replies) for Raven authorship. One extra API call. Treat any Raven presence anywhere in the thread as "directed at Raven."

### 3. Silent failures in `_process_comment`
If `respond_to_comment` raises (Claude CLI timeout, diff fetch failure, empty response), the exception is caught and logged — nothing posts. User waits for a reply that never arrives.

**Fix:** on failure, post a short `⚠️ couldn't respond` reply threaded to the triggering comment so the user knows Raven tried. Mirrors what `_process_pr` already does for review failures.

### 4. Redundant location header on threaded replies
Every response prepends `🦅 **Re: `file.py` line 42**`. On BB DC threaded replies, the thread is already anchored at that file/line — the header duplicates what the UI shows. Gitea (no threading) still needs it.

**Fix:** only include the location header when `parent_comment_id` is not set (i.e., top-level post) or when the provider doesn't thread.

### 5. Position-based diff truncation loses relevant context
For a diff-comment on line 42 of `app/auth.py`, we truncate the diff at 3000 lines from the start. If `auth.py` sits after the cutoff, Raven answers without seeing the code being discussed.

**Fix:** when `file_path` is set, split the diff by file and reorder — put the relevant file's hunk first, then fill the remainder up to the truncation limit.

---

## Medium-impact

### 6. No "thinking" signal during 10–30s waits
Claude CLI with max thinking is slow. Users wait blindly. Gitea supports emoji reactions — Raven could 👀-react to the triggering comment as an immediate "I saw you." BB DC has no reaction API, so it stays no-op there.

### 7. `--effort max` for conversational replies is overkill
Reviews benefit from max thinking. Answering "why is this a bug?" rarely does. Introduce `CLAUDE_EFFORT_COMMENT` (default `medium`) separate from `CLAUDE_EFFORT`. Faster, cheaper replies.

### 8. Conversation window hard-capped at 20
`get_pr_comments(...)[-20:]` — on long discussions, early context is lost.

**Fix:** make tunable via `RAVEN_COMMENT_HISTORY` (default 20).

---

## Low-impact polish (deferred)

- **No per-PR opt-out.** A `no-raven` label would let devs mute Raven on specific PRs without disabling the webhook.
- **No help command.** `@raven help` produces a generic answer; a short static response listing what Raven can do would be friendlier.
- **Inline-comment context could include the code snippet** directly in the prompt instead of relying on Claude to locate it in the diff.

---

## Implementation plan

Four focused PRs:

1. **Small UX bundle** — #1 (remove cooldown), #3 (error feedback), #4 (skip redundant header when threaded), #7 (separate effort env var), #8 (tunable history window)
2. **Deep-thread lookup** — #2 (scan full thread for Raven authorship)
3. **Relevance-biased truncation** — #5 (put comment's file first in truncated diff)
4. **Comment reactions** — #6 (Gitea 👀 ack; BB DC no-op)
