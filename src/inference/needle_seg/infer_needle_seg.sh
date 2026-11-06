#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
image="${IMAGE:-flowmeter_train:v1.1}"
weights="${WEIGHTS:-${project_dir}/weights/needle_seg_best.pt}"
dataset="${DATASET:-${project_dir}/data/scene01_seg_min190}"
split_images="${SPLIT_IMAGES:-${project_dir}/data/scene01_yolo/images}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"

[[ -f "$weights" ]] || { printf 'Checkpoint not found: %s\n' "$weights" >&2; exit 1; }
for dir in "$dataset" "$split_images/train" "$split_images/val" "$split_images/test"; do
  [[ -d "$dir" ]] || { printf 'Required directory not found: %s\n' "$dir" >&2; exit 1; }
done

docker_cmd=(docker)
if (( EUID != 0 )) && ! docker image inspect "$image" >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p -- "$output_root"
printf 'Overlay output root on host: %s\n' "$(realpath -- "$output_root")"

exec "${docker_cmd[@]}" run --rm --pull=never --network=none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONPATH=/app \
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=-1 \
  --mount "type=bind,src=$(realpath -- "$project_dir/src"),dst=/app/src,readonly" \
  --mount "type=bind,src=$(realpath -- "$dataset"),dst=/app/data/scene01_seg_min190,readonly" \
  --mount "type=bind,src=$(realpath -- "$split_images"),dst=/app/data/scene01_yolo/images,readonly" \
  --mount "type=bind,src=$(realpath -- "$weights"),dst=/weights/needle_seg_best.pt,readonly" \
  --mount "type=bind,src=$(realpath -- "$output_root"),dst=/outputs" \
  --workdir /outputs --entrypoint python3.12 "$image" \
  -m src.inference.needle_seg.infer_needle_seg --weights /weights/needle_seg_best.pt --output-root /outputs "$@"
