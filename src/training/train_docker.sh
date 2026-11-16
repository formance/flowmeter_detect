#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${IMAGE:-flowmeter_train:v1.1}"
gpus="${GPUS:-0}"
dataset="${DATASET:-$project_dir/data/pointer_scale_seg}"
weights="${WEIGHTS:-$project_dir/weights/resnet18-f37072fd.pth}"
output="${OUTPUT_ROOT:-$project_dir/outputs}"
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPUS.' >&2; exit 1; }
IFS=, read -r -a gpu_ids <<< "$gpus"
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
  [[ -z "${seen[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 1; }
  seen[$gpu]=1
done
[[ -f "$dataset/manifest.json" ]] || { echo "Dataset not found: $dataset" >&2; exit 1; }
[[ -f "$weights" ]] || { echo "Encoder weights not found: $weights (see docs/TRAINING.md)" >&2; exit 1; }
mkdir -p -- "$output"
visible=""
for index in "${!gpu_ids[@]}"; do visible+="${visible:+,}$index"; done
exec docker run --rm --pull=never --network=none --gpus "\"device=${gpus}\"" \
  --shm-size=2g --ulimit memlock=-1 --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env TZ=Asia/Shanghai --env PYTHONPATH=/app --env PYTHONDONTWRITEBYTECODE=1 \
  --env "GPUS=$visible" --env 'NCCL_SOCKET_IFNAME==lo' --env NCCL_CUMEM_HOST_ENABLE=0 \
  --mount "type=bind,src=$project_dir/src,dst=/app/src,readonly" \
  --mount "type=bind,src=$(realpath -- "$dataset"),dst=/app/data/pointer_scale_seg,readonly" \
  --mount "type=bind,src=$(realpath -- "$weights"),dst=/encoder.pth,readonly" \
  --mount "type=bind,src=$(realpath -- "$output"),dst=/outputs" \
  --workdir /app --entrypoint /bin/bash "$image" src/training/train.sh \
  --encoder-weights /encoder.pth --output-root /outputs "$@"
