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

mount_options=(
  --mount "type=bind,src=${project_dir},dst=/app,readonly"
  --mount "type=bind,src=${project_dir}/outputs,dst=/app/outputs"
)
if [[ "$project_dir" != /app ]]; then
  mount_options+=(
    --mount "type=bind,src=${project_dir},dst=${project_dir},readonly"
    --mount "type=bind,src=${project_dir}/outputs,dst=${project_dir}/outputs"
  )
fi
if [[ -L "$project_dir/data" ]]; then
  data_dir="$(realpath -e -- "$project_dir/data")"
  [[ -d "$data_dir" ]] || { printf 'Data link target is not a directory: %s\n' "$data_dir" >&2; exit 1; }
  if [[ "$data_dir" != "$project_dir" && "$data_dir" != "$project_dir/"* ]]; then
    mount_options+=(--mount "type=bind,src=${data_dir},dst=${data_dir},readonly")
  fi
fi

exec "${docker_cmd[@]}" run --rm -i --pull=never --network=none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=-1 \
  "${mount_options[@]}" \
  --workdir "$project_dir" --entrypoint python "$image" -m src.inference.flowmeter.infer_flowmeter "$@"
