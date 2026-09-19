#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export VISLOC_IMAGE=${VISLOC_IMAGE:-visloc-rs:realsense-browser}
export BROWSER_CONNECT=${BROWSER_CONNECT:-http://host.docker.internal:8090}
export RERUN_CONNECT=${RERUN_CONNECT-}
exec bash docker/jetson/run_realsense.sh "$@"
