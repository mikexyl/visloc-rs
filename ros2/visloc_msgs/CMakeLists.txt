cmake_minimum_required(VERSION 3.16)
project(visloc_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_generator_rs REQUIRED)
# Regenerate interfaces after a branch update adds/removes messages, including
# when an existing colcon build directory is reused.
file(GLOB messages CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg srv/*.srv)
rosidl_generate_interfaces(${PROJECT_NAME} ${messages})
ament_export_dependencies(rosidl_default_runtime)
ament_package()
