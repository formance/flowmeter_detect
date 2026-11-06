#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
image="${IMAGE:-flowmeter_train:v1.1}"
det_weights="${DET_WEIGHTS:-${project_dir}/weights/gauge_det_best.pt}"
needle_weights="${NEEDLE_WEIGHTS:-${project_dir}/weights/needle_seg_best.pt}"
output_dir="${OUTPUT_DIR:-${project_dir}/weights/onnx}"

docker_cmd=(docker)
if (( EUID != 0 )) && ! docker image inspect "$image" >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
container_options=(
  --rm --pull=never --user "$(id -u):$(id -g)" --env HOME=/tmp --env PYTHONPATH=/app
  --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=-1 --entrypoint python3.12
)

for file in "$det_weights" "$needle_weights" "$project_dir/src/training/export/export_models_onnx.py" "$project_dir/src/common/needle/needle_seg_model.py"; do
  [[ -f "$file" ]] || { printf 'Required file not found: %s\n' "$file" >&2; exit 1; }
done

mkdir -p -- "$output_dir"
printf 'ONNX output directory on host: %s\n' "$(realpath -- "$output_dir")"
exec "${docker_cmd[@]}" run "${container_options[@]}" --network=none \
  --mount "type=bind,src=$(realpath -- "$project_dir/src"),dst=/app/src,readonly" \
  --mount "type=bind,src=$(realpath -- "$det_weights"),dst=/checkpoints/gauge_det_best.pt,readonly" \
  --mount "type=bind,src=$(realpath -- "$needle_weights"),dst=/checkpoints/needle_seg_best.pt,readonly" \
  --mount "type=bind,src=$(realpath -- "$output_dir"),dst=/onnx" \
  --workdir /onnx "$image" -m src.training.export.export_models_onnx \
  --det-weights /checkpoints/gauge_det_best.pt --needle-weights /checkpoints/needle_seg_best.pt --output-dir /onnx "$@"
