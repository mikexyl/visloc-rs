#!/usr/bin/env bash
set -euo pipefail
cd /home/mikexyl/workspaces/visloc-rs
mkdir -p results/map-resource-image-smoke-20260917
docker run --rm --network none \
  --mount type=bind,src=/home/mikexyl/data/euroc/MH_01_easy_80,dst=/dataset,readonly \
  --mount type=bind,src=/home/mikexyl/workspaces/visloc-rs/results/map-resource-image-smoke-20260917,dst=/output \
  --entrypoint bash visloc-rs:realsense-browser -lc '
    set -e
    python3 -m py_compile /opt/visloc-rs/tools/browser_viewer/live_server.py /opt/visloc-rs/tools/browser_viewer/map_store.py /opt/visloc-rs/tools/browser_viewer/resource_monitor.py
    mkdir -p /opt/visloc-rs/target/release/examples
    ln -s /usr/local/bin/basalt_stream_vio /opt/visloc-rs/target/release/examples/basalt_stream_vio
    python3 /opt/visloc-rs/tools/browser_viewer/test_online_euroc.py --euroc-dir /dataset --frames 80 --output /output --url http://127.0.0.1:9
  ' > results/map-resource-image-smoke-20260917/run.log 2>&1
cat results/map-resource-image-smoke-20260917/summary.json
docker image inspect --format '{{.Id}}' visloc-rs:realsense-browser
docker image inspect --format '{{.Id}}' visloc-rs:jetson
systemctl is-active visloc-viewer
docker ps --format '{{.Names}} {{.Image}}'
