#!/bin/sh
set -e

# generate host keys on first start (fresh identity per image; client uses host key checking off / known_hosts handling)
if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
    ssh-keygen -A >/dev/null 2>&1
fi

mkdir -p /run/sshd
exec /usr/sbin/sshd -D -e
