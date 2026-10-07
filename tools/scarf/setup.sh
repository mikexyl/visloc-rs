#!/usr/bin/env bash
# Isolated Python/CUDA mapper; no ROS nodes or OpenVINS build required.
set -euo pipefail
unset PYTHONPATH
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RUNTIME="$REPO/.runtime/scarf"
DA3_REV=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
mkdir -p "$RUNTIME"
git -C "$REPO" submodule update --init third_party/ScaRF-SLAM
if [[ ! -d "$RUNTIME/Depth-Anything-3" ]]; then
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$RUNTIME/Depth-Anything-3"
    git -C "$RUNTIME/Depth-Anything-3" checkout "$DA3_REV"
fi
if [[ $(git -C "$RUNTIME/Depth-Anything-3" rev-parse HEAD) != "$DA3_REV" ]] ||
   [[ -n $(git -C "$RUNTIME/Depth-Anything-3" status --porcelain) ]]; then
    echo 'Depth Anything 3 must be clean at the pinned revision; refusing to reset local changes.' >&2
    exit 1
fi
if [[ ! -x "$RUNTIME/venv/bin/python" ]]; then
    uv venv --python 3.11 "$RUNTIME/venv"
fi
uv pip install --python "$RUNTIME/venv/bin/python" -r "$REPO/tools/scarf/requirements.txt"
# DA3's PyTorch SDPA path does not need xformers or its web application extras.
uv pip install --python "$RUNTIME/venv/bin/python" --no-deps -e "$RUNTIME/Depth-Anything-3"
"$RUNTIME/venv/bin/python" - "$RUNTIME" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
model = snapshot_download('depth-anything/DA3-LARGE', revision='c54c26b16ec04d218e8d584ecf4bce082a9fcc20')
link = root / 'DA3-LARGE'
if not link.exists():
    link.symlink_to(model, target_is_directory=True)
PY
