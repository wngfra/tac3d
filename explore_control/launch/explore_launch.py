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
            {'baudrate' : 115200},
            {'device'   : 'cpu'},
            {'dim'      : [10, 10]},
            {'port'     : ''}
        ],
        output='screen',
        emulate_tty=True
    )

    motion_controller_node = Node(
        namespace='tac3d',
        name='motion_controller_node',
        package='explore_control',
        executable='motion_controller',
        arguments=[],
    )

    ld.add_action(tactile_interface_node)
    ld.add_action(motion_controller_node)

    return ld