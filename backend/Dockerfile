FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

USER root

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --no-dev

RUN mkdir -p /app/.ssh && chmod 700 /app/.ssh

COPY docker/entrypoint.sh /usr/local/bin/backend-entrypoint.sh
RUN chmod +x /usr/local/bin/backend-entrypoint.sh

EXPOSE 2024

ENTRYPOINT ["/usr/local/bin/backend-entrypoint.sh"]