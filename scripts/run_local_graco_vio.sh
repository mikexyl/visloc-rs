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
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --locked --release \
    --example basalt_stream_vio --features basalt-lm-workspace-reuse
exec target/graco-venv/bin/python scripts/run_graco_vio.py "$@"
