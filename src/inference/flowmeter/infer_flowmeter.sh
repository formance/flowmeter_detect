#!/usr/bin/env bash
set -euo pipefail

project_dir="$(realpath -- "${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}")"
image="${IMAGE:-flowmeter_infer:v1.1}"
docker_cmd=(docker)
if (( EUID != 0 )) && ! docker image inspect "$image" >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p -- "$project_dir/outputs"

exec "${docker_cmd[@]}" run --rm -i --pull=never --network=none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=-1 \
  --mount "type=bind,src=${project_dir},dst=/app,readonly" \
  --mount "type=bind,src=${project_dir}/outputs,dst=/app/outputs" \
  --workdir /app --entrypoint python "$image" -m src.inference.flowmeter.infer_flowmeter "$@"
