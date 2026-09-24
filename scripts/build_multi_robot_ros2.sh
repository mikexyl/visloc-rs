#!/usr/bin/env bash
set -eo pipefail
VISLOC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VISLOC_ROS_INSTALL=${VISLOC_ROS_INSTALL:-"$VISLOC_REPO/.runtime/ros2_install"}
VISLOC_ROS_BUILD=${VISLOC_ROS_BUILD:-"$VISLOC_REPO/.runtime/ros2_build"}
cd "$VISLOC_REPO"
source /opt/ros/humble/setup.bash
source .runtime/ros2-venv/bin/activate
python -m colcon --log-base .runtime/ros2_log build \
  --base-paths .runtime/ros2_deps/rosidl_rust/rosidl_generator_rs ros2/visloc_msgs \
  --build-base "$VISLOC_ROS_BUILD" --install-base "$VISLOC_ROS_INSTALL" \
  --allow-overriding rosidl_generator_rs \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -Wno-dev
source "$VISLOC_ROS_INSTALL/setup.bash"
export AMENT_PREFIX_PATH="$VISLOC_REPO/.runtime/ros2_system_deps/opt/ros/humble:$AMENT_PREFIX_PATH"
export LD_LIBRARY_PATH="$VISLOC_REPO/.runtime/ros2_system_deps/opt/ros/humble/lib:$VISLOC_REPO/.runtime/clang/usr/lib/llvm-14/lib:$VISLOC_REPO/.runtime/clang/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CLANG_PATH="$VISLOC_REPO/.runtime/clang/usr/bin/clang-14"
export LIBCLANG_PATH=/lib/x86_64-linux-gnu
python scripts/ros2_cargo_paths.py
cd ros2
RUSTFLAGS='-C target-feature=+avx2,+fma' python -m colcon --log-base ../.runtime/ros2_log build \
  --base-paths visloc_ros --build-base "$VISLOC_ROS_BUILD" --install-base "$VISLOC_ROS_INSTALL" \
  --cargo-args --release
