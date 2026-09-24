#!/usr/bin/env bash
# Workspace-only dependencies for the validated Ubuntu 22.04 / ROS Humble build.
set -euo pipefail
VISLOC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$VISLOC_REPO"
test -f /opt/ros/humble/setup.bash || { echo 'ROS2 Humble is required.' >&2; exit 1; }
command -v uv >/dev/null
command -v cargo >/dev/null
/usr/bin/python3 - <<'PY'
import subprocess
v=subprocess.check_output(['rustc','--version'],text=True).split()[1]
assert tuple(map(int,v.split('.')[:2])) >= (1,85), 'visloc_ros requires Rust >=1.85'
PY
if [ ! -f .runtime/ros2-venv/bin/activate ]; then
  uv venv --python /usr/bin/python3 --system-site-packages .runtime/ros2-venv
fi
uv pip install --python .runtime/ros2-venv/bin/python \
  colcon-cargo==0.2.0 colcon-ros-cargo==0.2.0 cargo-ament-build==0.1.11 \
  empy==3.3.4 catkin-pkg lark PyYAML
mkdir -p .runtime/ros2_deps
if [ ! -d .runtime/ros2_deps/rosidl_rust/.git ]; then
  git clone https://github.com/ros2-rust/rosidl_rust .runtime/ros2_deps/rosidl_rust
  git -C .runtime/ros2_deps/rosidl_rust checkout --detach 19d57818dab3b51e418c0893b70b3a1a8b64c495
fi
test "$(git -C .runtime/ros2_deps/rosidl_rust rev-parse HEAD)" = 19d57818dab3b51e418c0893b70b3a1a8b64c495 || {
  echo 'Existing rosidl_rust checkout differs from ros2/dependencies.lock.json; leaving it untouched.' >&2; exit 1;
}
# rclrs 0.7 links test_msgs even outside its tests. Extract the pinned library
# locally rather than changing the system ROS installation.
mkdir -p .runtime/ros2_system_deps .runtime/clang
if [ ! -f .runtime/ros2_system_deps/opt/ros/humble/lib/libtest_msgs__rosidl_typesupport_c.so ]; then
  (cd .runtime/ros2_system_deps
   apt-get download 'ros-humble-test-msgs=1.2.3-1jammy.20260907.205609'
   dpkg-deb -x ros-humble-test-msgs_*.deb .)
fi
if [ ! -x .runtime/clang/usr/bin/clang-14 ]; then
  (cd .runtime/clang
   apt-get download 'clang-14=1:14.0.0-1ubuntu1.1' 'libclang-cpp14=1:14.0.0-1ubuntu1.1' 'libclang-common-14-dev=1:14.0.0-1ubuntu1.1'
   for archive in ./*.deb; do dpkg-deb -x "$archive" .; done)
fi
printf '%s\n' 'Workspace dependencies prepared. Build with bash scripts/build_multi_robot_ros2.sh.'
