FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# base toolchain; apt runs at build time only (build stage has egress, runtime is offline)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-server \
        python3 \
        python3-pip \
        nodejs \
        npm \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# user-approved apt packages; injected at rebuild by manager (task-05)
ARG APT_PKGS=""
RUN if [ -n "$APT_PKGS" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends $APT_PKGS \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# dedicated unprivileged exec user; workspace dir is a mounted volume (workspace-vol)
RUN useradd --create-home --shell /bin/bash exec \
    && mkdir -p /home/exec/workspace \
    && chown -R exec:exec /home/exec

# key-only SSH; graph_to_code public key baked from build context (stable across rebuilds)
COPY authorized_keys /tmp/authorized_keys
RUN mkdir -p /home/exec/.ssh \
    && install -o exec -g exec -m 600 /tmp/authorized_keys /home/exec/.ssh/authorized_keys \
    && chmod 700 /home/exec/.ssh \
    && rm /tmp/authorized_keys

COPY sshd_config /etc/ssh/sshd_config
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

VOLUME /home/exec/workspace
EXPOSE 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
