#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${IMAGE:-flowmeter_train:v1.0}"
dataset="${DATASET:-${project_dir}/data/scene01_yolo}"
weights="${WEIGHTS:-${project_dir}/weights/gauge_detection_model.pt}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"
trainer="${project_dir}/src/train_meter_det.py"
gpu="${GPU:-0}"

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a single host GPU index.' >&2; exit 1; }
for file in "$trainer" "$weights" "$dataset/data.yaml"; do
  [[ -f "$file" ]] || { printf 'Required file not found: %s\n' "$file" >&2; exit 1; }
done

docker_cmd=(docker)
if (( EUID != 0 )); then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p -- "$output_root"
printf 'Training outputs on host: %s\n' "$(realpath -- "$output_root")"

exec "${docker_cmd[@]}" run --rm --pull=never \
  --gpus "device=${gpu}" --shm-size=2g --network=none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,src=$(realpath -- "$trainer"),dst=/app/src/train_meter_det.py,readonly" \
  --mount "type=bind,src=$(realpath -- "$dataset"),dst=/app/data/scene01_yolo,readonly" \
  --mount "type=bind,src=$(realpath -- "$weights"),dst=/app/weights/gauge_detection_model.pt,readonly" \
  --mount "type=bind,src=$(realpath -- "$output_root"),dst=/outputs" \
  --workdir /outputs --entrypoint python3.12 "$image" \
  /app/src/train_meter_det.py --output-root /outputs "$@"
