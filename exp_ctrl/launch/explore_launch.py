from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode


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


    ld.add_action(tactile_interface_node)

    return ld