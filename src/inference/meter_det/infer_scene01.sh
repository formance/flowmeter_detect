#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
dataset="${DATASET:-${project_dir}/data/scene01}"
weights="${WEIGHTS:-${project_dir}/weights/gauge_detection_model.pt}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"
device="${DEVICE:-cpu}"
gpu="${GPU:-0}"
conf="${CONF:-0.5}"
iou="${IOU:-0.5}"
imgsz="${IMGSZ:-640}"
batch="${BATCH:-8}"

case "$device" in
  gpu)
    default_image=flowmeter_train:v1.1
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a single host GPU index.' >&2; exit 1; }
    device_options=(--gpus "device=${gpu}")
    model_device=0
    ;;
  cpu)
    default_image=flowmeter_infer:v1.1
    device_options=(--env NVIDIA_VISIBLE_DEVICES=void)
    model_device=cpu
    ;;
  *) echo 'DEVICE must be gpu or cpu.' >&2; exit 1 ;;
esac
image="${IMAGE:-$default_image}"

[[ -d "$dataset" ]] || { printf 'Image directory not found: %s\n' "$dataset" >&2; exit 1; }
[[ -f "$weights" ]] || { printf 'Weights not found: %s\n' "$weights" >&2; exit 1; }
dataset="$(realpath -- "$dataset")"
weights="$(realpath -- "$weights")"

docker_cmd=(docker)
if (( EUID != 0 )); then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null

mkdir -p -- "$output_root"
output_root="$(realpath -- "$output_root")"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
run_base="${output_root}/scene01_${device}_${timestamp}"
run_dir="$run_base"
run_number=0
until mkdir -- "$run_dir" 2>/dev/null; do
  if [[ ! -e "$run_dir" && ! -L "$run_dir" ]]; then
    printf 'Could not create output directory: %s\n' "$run_dir" >&2
    exit 1
  fi
  run_number=$((run_number + 1))
  run_dir="${run_base}_${run_number}"
done
printf 'Prediction images: %s/predictions\n' "$run_dir"

exec "${docker_cmd[@]}" run --rm --pull=never \
  "${device_options[@]}" --shm-size=2g --network=none \
  --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,src=${dataset},dst=/images,readonly" \
  --mount "type=bind,src=${weights},dst=/weights/gauge_detection_model.pt,readonly" \
  --mount "type=bind,src=${run_dir},dst=/outputs" \
  --entrypoint yolo "$image" detect predict \
  model=/weights/gauge_detection_model.pt source=/images \
  device="$model_device" conf="$conf" iou="$iou" imgsz="$imgsz" batch="$batch" \
  half=False save=True save_txt=False save_crop=False verbose=False \
  project=/outputs name=predictions exist_ok=True
