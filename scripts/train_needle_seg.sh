#!/usr/bin/env bash
set -euo pipefail

project_dir="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${IMAGE:-flowmeter_train:v1.0}"
dataset="${DATASET:-${project_dir}/data/scene01_seg_min190}"
split_images="${SPLIT_IMAGES:-${project_dir}/data/scene01_yolo/images}"
weights="${WEIGHTS:-${project_dir}/weights/resnet18-f37072fd.pth}"
output_root="${OUTPUT_ROOT:-${project_dir}/outputs}"
gpus="${GPUS:-${GPU:-0}}"

[[ "$gpus" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || {
  echo 'GPUS must be a list of host GPU indices, for example GPUS=0,1.' >&2
  exit 1
}
IFS=, read -r -a gpu_ids <<< "$gpus"
declare -A seen_gpus=()
for gpu_id in "${gpu_ids[@]}"; do
  if [[ -n "${seen_gpus[$gpu_id]:-}" ]]; then
    printf 'GPU index specified more than once: %s\n' "$gpu_id" >&2
    exit 1
  fi
  seen_gpus[$gpu_id]=1
done
gpu_options=(--gpus "device=${gpus}")
launcher=()
if (( ${#gpu_ids[@]} > 1 )); then
  # The offline container uses loopback and the existing /dev/shm for NCCL 2.26.2.
  gpu_options=(--gpus "\"device=${gpus}\"" --env 'NCCL_SOCKET_IFNAME==lo'
               --env NCCL_CUMEM_HOST_ENABLE=0 --ulimit memlock=-1)
  launcher=(-m torch.distributed.run --nnodes=1 "--nproc-per-node=${#gpu_ids[@]}"
            --master-addr=127.0.0.1 --master-port=29500)
fi
for file in "$weights" "$project_dir/src/train_needle_seg.py" "$project_dir/src/needle_seg_model.py"; do
  [[ -f "$file" ]] || { printf 'Required file not found: %s\n' "$file" >&2; exit 1; }
done
for dir in "$dataset/xanylabel_json" "$split_images/train" "$split_images/val" "$split_images/test"; do
  [[ -d "$dir" ]] || { printf 'Required directory not found: %s\n' "$dir" >&2; exit 1; }
done

docker_cmd=(docker)
if (( EUID != 0 )) && ! docker image inspect "$image" >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
"${docker_cmd[@]}" image inspect "$image" >/dev/null
mkdir -p -- "$output_root"
printf 'Training outputs on host: %s\n' "$(realpath -- "$output_root")"

exec "${docker_cmd[@]}" run --rm --pull=never --network=none \
  "${gpu_options[@]}" --shm-size=2g --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,src=$(realpath -- "$project_dir/src"),dst=/app/src,readonly" \
  --mount "type=bind,src=$(realpath -- "$dataset"),dst=/app/data/scene01_seg_min190,readonly" \
  --mount "type=bind,src=$(realpath -- "$split_images"),dst=/app/data/scene01_yolo/images,readonly" \
  --mount "type=bind,src=$(realpath -- "$weights"),dst=/app/weights/resnet18-f37072fd.pth,readonly" \
  --mount "type=bind,src=$(realpath -- "$output_root"),dst=/outputs" \
  --workdir /outputs --entrypoint python3.12 "$image" \
  "${launcher[@]}" /app/src/train_needle_seg.py --output-root /outputs "$@"
