FROM python:3.12-slim

WORKDIR /app

# Install Node.js + claude CLI (baseline version; entrypoint updates to latest on start)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git gnupg && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    npm install -g @anthropic-ai/claude-code && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY raven/ ./raven/
COPY prompts/ ./prompts/
COPY entrypoint.sh ./entrypoint.sh

# Logs directory
RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1 \
    CLAUDE_MODEL=claude-opus-4-6 \
    CLAUDE_EFFORT=max

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "raven.server:create_app()"]
