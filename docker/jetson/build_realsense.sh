#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
bash docker/jetson/build.sh
docker build --network host -f docker/jetson/Dockerfile.realsense \
    -t visloc-rs:realsense .
