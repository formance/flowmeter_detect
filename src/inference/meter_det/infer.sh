#!/usr/bin/env bash
set -euo pipefail

# Run inside a container with the project mounted at /app.
cd -- "${PROJECT_DIR:-/app}"
if (( $# == 0 )); then
  set -- --source data/test.jpg
fi
exec python3.12 -m src.inference.meter_det.infer_stage1 \
  --weights weights/gauge_detection_model.pt --output outputs/stage1.png "$@"
