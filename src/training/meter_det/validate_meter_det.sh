#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${PROJECT_DIR:-$(cd -- "${script_dir}/../../.." && pwd)}"
image="${IMAGE:-flowmeter_train:v1.1}"
device="${DEVICE:-gpu}"
gpu="${GPU:-0}"
conf="${CONF:-0.5}"
iou="${IOU:-0.5}"
dataset="${DATASET:-/data/sevnce/hh/flowmeter/meter_det}"
weights="${WEIGHTS:-${project_dir}/weights/gauge_detection_model.pt}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"
imgsz="${IMGSZ:-640}"
batch="${BATCH:-8}"
threads="${THREADS:-4}"

# Support both the source tree and the existing standalone remote validator.
validator="${VALIDATOR:-${project_dir}/src/training/meter_det/validate_meter_det.py}"
if [[ -z "${VALIDATOR:-}" && ! -f "$validator" ]]; then
  validator="${project_dir}/validate_meter_det.py"
fi

case "$device" in
  gpu)
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      printf 'GPU must be a single host GPU index, for example GPU=0.\n' >&2
      exit 1
    fi
    device_options=(--gpus "device=${gpu}")
    model_device=0
    device_label="GPU ${gpu} (container device 0)"
    ;;
  cpu)
    device_options=(--env NVIDIA_VISIBLE_DEVICES=void)
    model_device=cpu
    device_label=CPU
    ;;
  *)
    printf 'DEVICE must be gpu or cpu.\n' >&2
    exit 1
    ;;
esac
for value in "$imgsz" "$batch" "$threads"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'IMGSZ, BATCH and THREADS must be positive integers.\n' >&2
    exit 1
  fi
done
for file in "$validator" "$weights" "${dataset}/annotations/instance_test.json"; do
  if [[ ! -f "$file" ]]; then
    printf 'Required file not found: %s\n' "$file" >&2
    exit 1
  fi
done
if [[ ! -d "${dataset}/test" ]]; then
  printf 'Test image directory not found: %s/test\n' "$dataset" >&2
  exit 1
fi

validator="$(realpath -- "$validator")"
weights="$(realpath -- "$weights")"
dataset="$(realpath -- "$dataset")"
docker_cmd=(docker)
if (( EUID != 0 )); then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null

mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
timestamp="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
run_base="${output_root}/meter_det_${device}_${timestamp}"
run_dir="$run_base"
run_number=0
# Reserve each directory atomically so concurrent runs cannot share it.
until mkdir -m 700 -- "$run_dir" 2>/dev/null; do
  if [[ ! -e "$run_dir" && ! -L "$run_dir" ]]; then
    printf 'Could not create output directory: %s\n' "$run_dir" >&2
    exit 1
  fi
  run_number=$((run_number + 1))
  run_dir="${run_base}_${run_number}"
done
printf 'Device: %s\nPrediction images: %s/predictions\n' "$device_label" "$run_dir" >&2

exec "${docker_cmd[@]}" run --rm --pull=never \
  "${device_options[@]}" \
  --shm-size=2g \
  --network=none \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --mount "type=bind,src=${validator},dst=/validation/validate_meter_det.py,readonly" \
  --mount "type=bind,src=${dataset},dst=/datasets/meter_det,readonly" \
  --mount "type=bind,src=${weights},dst=/weights/gauge_detection_model.pt,readonly" \
  --mount "type=bind,src=${run_dir},dst=/outputs" \
  --workdir /outputs \
  --entrypoint python3.12 \
  "$image" /validation/validate_meter_det.py \
  --dataset /datasets/meter_det \
  --weights /weights/gauge_detection_model.pt \
  --output /outputs \
  --device "$model_device" --conf "$conf" --iou "$iou" \
  --imgsz "$imgsz" --batch "$batch" --threads "$threads"
