#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
image="${IMAGE:-flowmeter_infer:v1.1}"
python="${PYTHON:-python}"
dataset="${DATASET:-${project_dir}/data/scene01_seg}"
onnx="${ONNX:-${project_dir}/weights/meter_seg_model.onnx}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"
runtime_dir="${RUNTIME_DIR:-${project_dir}/.stage2_runtime}"

docker_cmd=(docker)
if (( EUID != 0 )) && ! docker image inspect "$image" >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
container_options=(
  --rm --pull=never --user "$(id -u):$(id -g)"
  --env HOME=/tmp --env NVIDIA_VISIBLE_DEVICES=void --env TZ=Asia/Shanghai
  --entrypoint "$python"
)

if [[ "${1:-}" == --install-runtime ]]; then
  shift
  mkdir -p -- "$runtime_dir"
  exec "${docker_cmd[@]}" run "${container_options[@]}" \
    --mount "type=bind,src=$(realpath -- "$runtime_dir"),dst=/runtime" \
    "$image" -m pip install --no-cache-dir --prefix /runtime 'onnxruntime==1.22.1' "$@"
fi

[[ -d "$dataset" ]] || { printf 'Crop directory not found: %s\n' "$dataset" >&2; exit 1; }
for file in "$onnx" "${project_dir}/src/inference/scale_seg/infer_stage2_onnx.py"; do
  [[ -f "$file" ]] || { printf 'Required file not found: %s\n' "$file" >&2; exit 1; }
done
if [[ -d "$runtime_dir" ]]; then
  container_options+=(
    --mount "type=bind,src=$(realpath -- "$runtime_dir"),dst=/runtime,readonly"
    --env PYTHONPATH=/runtime/lib/python3.12/site-packages
  )
fi

mkdir -p -- "$output_root"
printf 'Validation outputs on host: %s\n' "$(realpath -- "$output_root")"
exec "${docker_cmd[@]}" run "${container_options[@]}" --network=none \
  --mount "type=bind,src=$(realpath -- "$project_dir/src"),dst=/app/src,readonly" \
  --mount "type=bind,src=$(realpath -- "$dataset"),dst=/crops,readonly" \
  --mount "type=bind,src=$(realpath -- "$onnx"),dst=/app/weights/meter_seg_model.onnx,readonly" \
  --mount "type=bind,src=$(realpath -- "$output_root"),dst=/app/outputs" \
  --workdir /app "$image" -m src.inference.scale_seg.infer_stage2_onnx --source-dir /crops "$@"
