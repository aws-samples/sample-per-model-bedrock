# AgentCore Runtime requires linux/arm64.
FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_MAJOR=22 \
    HOME=/home/agent

# git and openssh are how the agent reaches GitHub; curl/ca-certificates for the
# feeds it polls; nodejs because the Claude Agent SDK drives the Claude Code CLI,
# which is a Node program even when the SDK bindings are Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client gnupg jq unzip \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove gnupg \
    && rm -rf /var/lib/apt/lists/*

# The CLI the SDK spawns. Pinned: an agent that rewrites a published repository
# should not have its reasoning engine change under it without a deliberate bump.
RUN npm install -g --no-fund --no-audit @anthropic-ai/claude-code@2.1.263

# Runtime dependencies, from a fully resolved and hash-pinned lock. --require-hashes
# means a wheel that has been re-uploaded or tampered with fails the build rather than
# quietly shipping: this container commits to a public AWS repository, so a substituted
# dependency is a supply-chain problem, not an inconvenience.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock.txt

# The scanners, each in its own virtual environment with only its command on PATH.
# `uv tool install` is what keeps them isolated -- see requirements-scanners.txt for
# why sharing one environment with the SDK is not possible.
ENV UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin
COPY requirements-scanners.txt /tmp/requirements-scanners.txt
RUN pip install --no-cache-dir uv==0.9.11 \
    && grep -v '^\s*\(#\|$\)' /tmp/requirements-scanners.txt \
       | while read -r spec; do echo "installing $spec"; uv tool install "$spec"; done \
    && chmod -R a+rX /opt/uv-tools \
    && rm -rf /root/.cache/uv

# Non-root. AgentCore does not require it, but the agent runs shell commands it
# composes itself, so it should not compose them as root.
RUN useradd --create-home --shell /bin/bash --uid 1000 agent
WORKDIR /app
COPY --chown=agent:agent . /app
RUN mkdir -p /home/agent/.ssh /app/work && chown -R agent:agent /home/agent /app
USER agent

# Prove the image can do its job, as the user that will actually do it, and that the
# guards that keep it safe still hold. verify_image.py checks the toolchain; the tests
# plant the specific mistake each guard exists to catch. Both run here so a broken
# guard fails the build rather than being discovered by its consequences.
RUN python verify_image.py && python tests/test_guards.py

# Where the agent clones the repository it maintains.
ENV AGENT_WORKDIR=/app/work \
    CLAUDE_CODE_USE_BEDROCK=1 \
    PATH="/home/agent/.local/bin:${PATH}"

EXPOSE 8080

# Launched under `opentelemetry-instrument`, not bare python. The aws-opentelemetry-distro
# only activates when it wraps the process: with `python app.py` the OTEL environment
# variables are set and read by nobody, so spans and metrics are silently never emitted
# and the CloudWatch GenAI pages stay empty. That looks exactly like a service problem
# and is not one.
CMD ["opentelemetry-instrument", "python", "-u", "app.py"]
