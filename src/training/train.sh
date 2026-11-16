#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
gpus="${GPUS:-0}"
[[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'GPUS must be comma-separated GPU indices.' >&2; exit 1; }
IFS=, read -r -a gpu_ids <<< "$gpus"
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
  [[ -z "${seen[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 1; }
  seen[$gpu]=1
done
export CUDA_VISIBLE_DEVICES="$gpus"
if (( ${#gpu_ids[@]} > 1 )); then
  exec "${PYTHON:-python3.12}" -B -m torch.distributed.run --nnodes=1 \
    --master-addr=127.0.0.1 "--master-port=${MASTER_PORT:-29500}" \
    "--nproc-per-node=${#gpu_ids[@]}" --module src.training.train --device "${DEVICE:-cuda:0}" "$@"
fi
exec "${PYTHON:-python3.12}" -B -m src.training.train --device "${DEVICE:-cuda:0}" "$@"
