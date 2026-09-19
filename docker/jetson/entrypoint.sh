#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 2 ]]; then
    echo 'Usage: docker run ... IMAGE /dataset /output [VIO options]'
    echo 'Mount an extracted EuRoC sequence at /dataset (read-only) and a writable /output.'
    echo 'Extra options include --max-frames 80; default is the full sequence.'
    exit 0
fi
sequence_dir=$1
output_dir=$2
shift 2
for sensor in cam0 cam1 imu0; do
    test -f "$sequence_dir/mav0/$sensor/data.csv" || {
        echo "Missing manifest: $sequence_dir/mav0/$sensor/data.csv" >&2
        exit 1
    }
done
mkdir -p "$output_dir"
if [[ -e "$output_dir/trajectory.tum" ]]; then
    echo "Output already contains a trajectory: $output_dir" >&2
    exit 1
fi
calibration=${VISLOC_CALIBRATION:-/opt/visloc-rs/configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json}
config=${VISLOC_CONFIG:-/opt/visloc-rs/configs/basalt/variants/official_euroc_ds/euroc_config.json}
/usr/bin/time -v -o "$output_dir/timing.txt" basalt_euroc_vio_demo \
    --euroc-dir "$sequence_dir" --out-dir "$output_dir" \
    --calibration "$calibration" --config "$config" \
    --no-trace --no-marg-data "$@" 2>&1 | tee "$output_dir/run.log"
if [[ -f "$sequence_dir/mav0/state_groundtruth_estimate0/data.csv" ]]; then
    python3 /opt/visloc-rs/scripts/evaluate_euroc_trajectory.py \
        --ground-truth-csv "$sequence_dir/mav0/state_groundtruth_estimate0/data.csv" \
        --trajectory "$output_dir/trajectory.tum" --tum-time-unit s \
        --out-json "$output_dir/evaluation.json"
fi
