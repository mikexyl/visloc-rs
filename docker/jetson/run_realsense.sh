#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
output_dir=${1:-results/realsense-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
name=${VISLOC_CONTAINER:-visloc-realsense}
# The RSUSB backend needs only USB device access; no privileged container or host networking.
mode=()
if [[ ${VISLOC_DETACH:-0} == 1 ]]; then mode=(-d); fi
exec docker run "${mode[@]}" --rm --init --name "$name" --network bridge \
    --add-host host.docker.internal:host-gateway \
    --env "RERUN_CONNECT=${RERUN_CONNECT-rerun+http://192.168.0.243:9878/proxy}" \
    --env "BROWSER_CONNECT=${BROWSER_CONNECT:-}" \
    --device /dev/bus/usb --mount type=bind,src=/run/udev,dst=/run/udev,readonly \
    --mount "type=bind,src=$output_dir,dst=/output" \
    "${VISLOC_IMAGE:-visloc-rs:realsense}" \
    --output /output "$@"
