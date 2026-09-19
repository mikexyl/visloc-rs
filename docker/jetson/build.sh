#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ $(uname -m) != aarch64 ]]; then
    echo 'Run this build on the ARM64 Jetson.' >&2
    exit 1
fi
docker build --network host -f docker/jetson/Dockerfile \
    --build-arg BUILD_JOBS="${BUILD_JOBS:-4}" \
    --build-arg TARGET_CPU="${TARGET_CPU:-cortex-a78}" \
    --build-arg SOURCE_REVISION="${SOURCE_REVISION:-unknown}" \
    -t "${VISLOC_IMAGE:-visloc-rs:jetson}" .
