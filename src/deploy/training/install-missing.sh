#!/bin/sh
# Install only packages missing from the current image stage.
set -eu
missing=""
for package in "$@"; do
    status=$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)
    if [ "$status" = "install ok installed" ]; then
        printf 'Already installed: %s\n' "$package"
    else
        missing="$missing $package"
    fi
done
if [ -n "$missing" ]; then
    apt-get update
    # Package names originate from the explicit Dockerfile list.
    apt-get install -y --no-install-recommends --no-upgrade $missing
    rm -rf /var/lib/apt/lists/*
fi
