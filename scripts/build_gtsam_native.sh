#!/usr/bin/env bash
# Build the pinned C++ dependency locally; no system installation or Python.
set -euo pipefail
VISLOC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VISLOC_GTSAM_BUILD="$VISLOC_REPO/.runtime/gtsam-native"
VISLOC_GTSAM_REV=71a25ca36c084cbad1f872e812d6d97fbadfdb05
mkdir -p "$VISLOC_GTSAM_BUILD"
if [[ ! -d "$VISLOC_GTSAM_BUILD/source/.git" ]]; then
  git clone --depth 1 --branch 4.3.0 https://github.com/borglab/gtsam.git "$VISLOC_GTSAM_BUILD/source"
fi
[[ $(git -C "$VISLOC_GTSAM_BUILD/source" rev-parse HEAD) == "$VISLOC_GTSAM_REV" ]] || {
  echo "GTSAM checkout does not match pinned revision $VISLOC_GTSAM_REV" >&2
  exit 1
}
[[ -z $(git -C "$VISLOC_GTSAM_BUILD/source" status --porcelain --untracked-files=no) ]] || {
  echo "GTSAM dependency has tracked modifications; refusing an unrecorded build" >&2
  exit 1
}
cmake -S "$VISLOC_GTSAM_BUILD/source" -B "$VISLOC_GTSAM_BUILD/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$VISLOC_GTSAM_BUILD/install" \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
  -DGTSAM_BUILD_PYTHON=OFF -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_POSE3_EXPMAP=ON -DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON \
  -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF -DGTSAM_USE_BOOST_FEATURES=OFF \
  -DGTSAM_SUPPORT_NESTED_DISSECTION=OFF -DGTSAM_BUILD_WITH_CCACHE=OFF
cmake --build "$VISLOC_GTSAM_BUILD/build" --parallel "${VISLOC_GTSAM_BUILD_JOBS:-4}"
cmake --install "$VISLOC_GTSAM_BUILD/build"
