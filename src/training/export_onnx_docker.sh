#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${IMAGE:-flowmeter_train:v1.1}"
checkpoint=""
output=""
extra_args=()

usage() {
  echo 'Usage: bash src/training/export_onnx_docker.sh --checkpoint MODEL.pt [--output MODEL.onnx] [--opset 17]'
}

while (( $# )); do
  case "$1" in
    --checkpoint|--output)
      (( $# >= 2 )) || { usage >&2; exit 2; }
      if [[ "$1" == '--checkpoint' ]]; then checkpoint="$2"; else output="$2"; fi
      shift 2
      ;;
    --checkpoint=*) checkpoint="${1#*=}"; shift ;;
    --output=*) output="${1#*=}"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

[[ -n "$checkpoint" && -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; usage >&2; exit 2; }
checkpoint="$(realpath -- "$checkpoint")"
output="${output:-${checkpoint%.*}.onnx}"
output="$(realpath -m -- "$output")"
[[ "$output" != "$checkpoint" && "${output%.*}.json" != "$checkpoint" ]] || { echo 'Output would overwrite checkpoint.' >&2; exit 2; }
output_dir="$(dirname -- "$output")"
mkdir -p -- "$output_dir"

exec docker run --rm --pull=never --network=none \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env PYTHONPATH=/app --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$project_dir/src,dst=/app/src,readonly" \
  --mount "type=bind,src=$checkpoint,dst=/checkpoint.pt,readonly" \
  --mount "type=bind,src=$output_dir,dst=/export" \
  --workdir /app --entrypoint python3.12 "$image" \
  -B -m src.training.export_onnx --checkpoint /checkpoint.pt \
  --output "/export/$(basename -- "$output")" "${extra_args[@]}"
