"""Launch native Rust nodes for live sensors or an external replay publisher."""
import json
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def nodes(context):
    path = Path(LaunchConfiguration('mission').perform(context))
    mission = json.loads(path.read_text())
    return [Node(package='visloc_ros', executable='backend', output='screen',
                 additional_env={'VISLOC_BACKEND_CONFIG': mission['backend_config']})] + [
        Node(package='visloc_ros', executable='robot', output='screen',
             additional_env={'VISLOC_ROBOT_CONFIG': robot['config']}) for robot in mission['robots']]

def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('mission'), OpaqueFunction(function=nodes)])
