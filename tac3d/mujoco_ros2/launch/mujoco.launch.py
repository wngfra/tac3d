from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    simulator_node = LifecycleNode(
        package="mujoco_ros2",
        name="mujoco_simulator_node",
        executable="simulator",
        namespace="tac3d",
        output="screen",
    )

    ld.add_action(simulator_node)

    return ld
