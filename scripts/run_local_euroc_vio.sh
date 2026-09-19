#!/usr/bin/env bash
# Build and run full-sequence Basalt VIO, then score against EuRoC ground truth.
# Usage: bash scripts/run_local_euroc_vio.sh [sequence_dir] [output_dir]
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sequence_dir="$(realpath "${1:-/data/euroc/MH_01_easy}")"
output_dir="$(realpath -m "${2:-$repo_dir/target/euroc_vio/$(basename "$sequence_dir")_$(date -u +%Y%m%dT%H%M%SZ)}")"
cd "$repo_dir"
for sensor in cam0 cam1 imu0; do
    test -f "$sequence_dir/mav0/$sensor/data.csv" || {
        echo "Missing EuRoC sensor manifest: $sequence_dir/mav0/$sensor/data.csv" >&2
        exit 1
    }
done
if [[ -e "$output_dir" ]]; then
    echo "Output already exists: $output_dir (choose a new directory)" >&2
    exit 1
fi
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --locked --release \
    --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse
mkdir -p "$output_dir"
git rev-parse HEAD > "$output_dir/revision.txt"
/usr/bin/time -v -o "$output_dir/timing.txt" \
    target/release/examples/basalt_euroc_vio_demo \
    --euroc-dir "$sequence_dir" \
    --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
    --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
    --out-dir "$output_dir" --no-trace --no-marg-data \
    > "$output_dir/run.log" 2>&1
ground_truth="$sequence_dir/mav0/state_groundtruth_estimate0/data.csv"
if [[ -f "$ground_truth" ]]; then
    python3 scripts/evaluate_euroc_trajectory.py \
        --ground-truth-csv "$ground_truth" \
        --trajectory "$output_dir/trajectory.tum" --tum-time-unit s \
        --out-json "$output_dir/evaluation.json"
fi
cat "$output_dir/summary.txt"
echo "Results: $output_dir"
