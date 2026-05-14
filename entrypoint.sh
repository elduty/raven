#!/bin/sh
set -e

# ── Install / update Claude Code CLI ─────────────────────────────────────────
# Try to update on startup (non-fatal — Dockerfile has a baseline version).
echo "entrypoint: updating Claude Code CLI..."
npm install -g @anthropic-ai/claude-code --loglevel=warn >/dev/null 2>&1 || echo "entrypoint: CLI update failed (using baseline from image)"
echo "entrypoint: claude CLI version $(claude --version 2>/dev/null || echo 'unknown')"

# Background daily update (runs every 24h while container is alive)
# - Failure is non-fatal (won't crash the container)
# - npm install -g does an atomic symlink swap, so in-progress reviews are safe
# - Requires --init in docker-compose for clean SIGTERM delivery
(trap 'exit 0' TERM INT; while true; do
    sleep 86400
    echo "entrypoint: daily Claude CLI update..."
    if npm install -g @anthropic-ai/claude-code --loglevel=warn >/dev/null 2>&1; then
        echo "entrypoint: updated to claude CLI $(claude --version 2>/dev/null || echo 'unknown')"
    else
        echo "entrypoint: CLI update failed (non-fatal), will retry in 24h"
    fi
done) &

# ── Initialise Claude Code credentials from env vars ────────────────────────
# CLAUDE_CODE_OAUTH_TOKEN can be either:
#   a) The full JSON blob from keychain/setup-token (contains accessToken, refreshToken, etc.)
#   b) Just the access token string (legacy format — requires CLAUDE_CODE_OAUTH_REFRESH_TOKEN)
#
# The CLI reads ~/.claude/.credentials.json on Linux and also needs
# ~/.claude.json with onboarding flag to skip interactive setup.

CRED_DIR="/root/.claude"
CRED_FILE="${CRED_DIR}/.credentials.json"
CLAUDE_JSON="/root/.claude.json"

if [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    mkdir -p "$CRED_DIR"

    # Detect if the token is a JSON blob or a bare token string
    case "$CLAUDE_CODE_OAUTH_TOKEN" in
        \{*)
            # Full JSON blob — write as-is
            printf '%s' "$CLAUDE_CODE_OAUTH_TOKEN" > "$CRED_FILE"
            ;;
        *)
            # Bare access token — wrap in expected format. Build the JSON
            # via python3 so tokens containing " or \ get properly escaped.
            # Heredoc shell-interpolation produced invalid JSON for such
            # tokens (a stray " closes the field early; the CLI then fails
            # to load credentials and presents as an opaque "auth failed"
            # at startup). Forward-compatible: alphanumeric tokens produce
            # identical output to the heredoc; the difference only matters
            # for tokens that previously broke.
            CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
            CLAUDE_CODE_OAUTH_REFRESH_TOKEN="${CLAUDE_CODE_OAUTH_REFRESH_TOKEN:-}" \
            python3 -c 'import json, os, sys
sys.stdout.write(json.dumps({
    "claudeAiOauth": {
        "accessToken": os.environ["CLAUDE_CODE_OAUTH_TOKEN"],
        "refreshToken": os.environ.get("CLAUDE_CODE_OAUTH_REFRESH_TOKEN", ""),
        "scopes": ["user:inference", "user:profile"],
    },
}))' > "$CRED_FILE"
            ;;
    esac

    chmod 600 "$CRED_FILE"
    echo "entrypoint: wrote Claude credentials to ${CRED_FILE}"

    # Bypass interactive onboarding
    if [ ! -f "$CLAUDE_JSON" ]; then
        echo '{"hasCompletedOnboarding":true}' > "$CLAUDE_JSON"
        echo "entrypoint: created ${CLAUDE_JSON} (skip onboarding)"
    fi
else
    echo "entrypoint: CLAUDE_CODE_OAUTH_TOKEN not set — skipping credential init"
fi

exec "$@"
