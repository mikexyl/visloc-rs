#!/usr/bin/env bash
# Ground-rig preset: ground-01, left camera + IMU, official ground calibration.
# Pass --output with a fresh directory; trailing arguments override defaults.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_local_graco_vio.sh" \
    --bag /data/graco/ground-01 \
    --calibration-dir /data/graco/ground-calibration \
    --camera-mode mono "$@"
