#!/usr/bin/env bash
# Setup/build, then sensor-only GRACO VIO with a saved Rerun recording.
# Additional arguments are passed to run_graco_vio.py.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -x target/graco-venv/bin/python ]]; then
    uv venv target/graco-venv --python python3
fi
uv pip install --python target/graco-venv/bin/python -r scripts/requirements-graco.txt
features=basalt-lm-workspace-reuse
for argument in "$@"; do
    if [[ "$argument" == --online-loop-config* ]]; then
        features+=,tensorrt-loop
    fi
done
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --locked --release \
    --example basalt_stream_vio --features "$features"
exec target/graco-venv/bin/python scripts/run_graco_vio.py "$@"
