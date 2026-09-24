# Source this file before running the ROS2 binaries or Python replay tools.
VISLOC_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
VISLOC_ROS_INSTALL=${VISLOC_ROS_INSTALL:-"$VISLOC_REPO/.runtime/ros2_install"}
export VISLOC_ROS_INSTALL
source "$VISLOC_ROS_INSTALL/setup.bash"
export AMENT_PREFIX_PATH="$VISLOC_REPO/.runtime/ros2_system_deps/opt/ros/humble:$AMENT_PREFIX_PATH"
export LD_LIBRARY_PATH="$VISLOC_REPO/.runtime/ros2_system_deps/opt/ros/humble/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
