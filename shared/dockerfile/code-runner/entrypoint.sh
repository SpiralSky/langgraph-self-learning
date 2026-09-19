#!/bin/sh
set -e

# generate host keys on first start (fresh identity per image; client uses host key checking off / known_hosts handling)
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -A >/dev/null 2>&1
fi

mkdir -p /run/sshd

# password-only mode when a password file is mounted (SSH_PASSWORD_FILE): feed the
# pw to sshd via chpasswd and flip auth config; image default (no pw file) stays key-only
if [ -n "$SSH_PASSWORD_FILE" ] && [ -s "$SSH_PASSWORD_FILE" ]; then
    pw="$(awk 'NF{print;exit}' "$SSH_PASSWORD_FILE")"
    if [ -n "$pw" ]; then
        echo "exec:$pw" | chpasswd
        sed -i 's/^\(PasswordAuthentication\) no/\1 yes/' /etc/ssh/sshd_config
        sed -i 's/^\(PubkeyAuthentication\) yes/\1 no/' /etc/ssh/sshd_config
    fi
fi

exec /usr/sbin/sshd -D -e
