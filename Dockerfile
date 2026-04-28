FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js from official binary
ARG NODE_VERSION=22.22.2
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then ARCH="x64"; fi && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${ARCH}.tar.gz" \
      -o /tmp/node.tar.gz && \
    tar -xzf /tmp/node.tar.gz -C /usr --strip-components=1 && \
    rm /tmp/node.tar.gz

# Install Claude Code CLI (baseline; entrypoint updates to latest on start)
RUN npm install -g @anthropic-ai/claude-code

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
    RAVEN_AI_MODEL=claude-opus-4-7 \
    RAVEN_AI_EFFORT=max

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "raven.server:create_app()"]
