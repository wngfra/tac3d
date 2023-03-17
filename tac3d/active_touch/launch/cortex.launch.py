from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # Cortex node
    cortex_node = Node(
        package="active_touch",
        name="cortex_node",
        executable="cortex",
        namespace="tac3d",
        output="screen",
    )

    ld.add_action(cortex_node)

    return ld
