#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${IMAGE:-flowmeter_train:v1.1}"
gpu="${GPU:-0}"

if [[ ! -f "${script_dir}/check_training_gpu.py" ]]; then
  printf 'Missing validation script: %s/check_training_gpu.py\n' "$script_dir" >&2
  exit 1
fi

docker_cmd=(docker)
if (( EUID != 0 )); then
  docker_cmd=(sudo docker)
fi

printf 'Validating image=%s GPU=%s\n' "$image" "$gpu"
exec "${docker_cmd[@]}" run --rm --pull=never \
  --gpus "device=${gpu}" \
  --shm-size=1g \
  --network=none \
  --mount "type=bind,src=${script_dir}/check_training_gpu.py,dst=/validation/check_training_gpu.py,readonly" \
  --entrypoint /bin/bash \
  "$image" -c '
    set -euo pipefail
    python3.12 --version
    python3.12 -m pip check
    python3.12 -c "import ssl, cv2, numpy, yaml, ultralytics, onnx, onnxruntime; print(\"Application imports: PASS\")"
    exec python3.12 /validation/check_training_gpu.py
  '
