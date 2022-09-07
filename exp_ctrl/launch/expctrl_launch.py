from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():
    ld = LaunchDescription()

    sensor_interface_node = LifecycleNode(
        namespace='tac3d',
        name='sensor_interface_node',
        package='sensor_interfaces',
        executable='SensorInterface',
        parameters=[
            {'animated' : False},
            {'baudrate' : 115200},
            {'dim'      : None},
            {'port'     : None}
        ],
        output='screen',
        emulate_tty=True
    )


    ld.add_action(sensor_interface_node)

    return 