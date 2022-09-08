from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    tactile_interface_node = Node(
        namespace='tac3d',
        name='sensor_interface_node',
        package='sensor_interfaces',
        executable='tactile_interface',
        parameters=[
            {'animated' : False},
            {'baudrate' : 115200},
            {'dim'      : [-1, -1]},
            {'port'     : ''}
        ],
        output='screen',
        emulate_tty=True
    )

    motion_controller_node = Node(
        namespace='tac3d',
        name='motion_controller_node',
        package='exp_ctrl',
        executable='motion_controller',
        arguments=[],
    )

    ld.add_action(tactile_interface_node)
    ld.add_action(motion_controller_node)

    return ld