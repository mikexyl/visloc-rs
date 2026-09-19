#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 /path/to/EuRoC/sequence /path/to/new/output [VIO options]" >&2
    exit 1
fi
sequence_dir=$(realpath "$1")
mkdir -p "$2"
output_dir=$(realpath "$2")
shift 2
docker run --rm --init --user "$(id -u):$(id -g)" \
    --cap-drop ALL --security-opt no-new-privileges --network none \
    --mount "type=bind,src=$sequence_dir,dst=/dataset,readonly" \
    --mount "type=bind,src=$output_dir,dst=/output" \
    "${VISLOC_IMAGE:-visloc-rs:jetson}" /dataset /output "$@"
