from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # MuJoCo simulation node
    mj_sim_node = LifecycleNode(
        package="mujoco_ros2",
        name="mujoco_simulator_node",
        executable="simulator",
        namespace="tac3d",
        output="screen",
    )

    def wait_for_mj_sim_node(context):
        while not context.runner.is_shutdown():
            if context.runner.is_process_active(mj_sim_node):
                break
            context.step()

    wait_for_mj_sim_node_handler = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=mj_sim_node, on_start=wait_for_mj_sim_node
        )
    )

    # rqt visualization node
    rqt_node = Node(
        package="rviz2",
        name="rviz2_node",
        executable="rviz2",
        # parameters=[{"topic": "/tact3d/mujoco_simulator/tactile_image"}],
        namespace="tac3d",
        on_exit=[wait_for_mj_sim_node_handler],
    )

    # Tactile encoding node
    tacnet_node = Node(
        package="active_touch",
        name="tacnet_node",
        executable="tacnet",
        namespace="tac3d",
        output="screen",
        on_exit=[wait_for_mj_sim_node_handler],
    )

    ld.add_action(mj_sim_node)
    ld.add_action(rqt_node)
    ld.add_action(tacnet_node)

    return ld
