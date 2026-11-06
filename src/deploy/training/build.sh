#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec sudo docker build \
  -f Dockerfile \
  -t "${IMAGE:-flowmeter_train:v1.1}" \
  .
