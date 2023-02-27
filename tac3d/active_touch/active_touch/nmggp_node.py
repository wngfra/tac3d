import numpy as np
import nengo

import rclpy
from rclpy.Node import Node
from rclpy.Node import Publisher
from rclpy.Node import Subscriber

from mujoco_interfaces.msg import Locus


class NMGGP(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self._sub = self.create_subscription(
            Locus, "mujoco_simulator/tactile_sensor", self.subscribe, 10
        )

    def subscribe(self, msg: Locus):
        data = msg.data


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = NMGGP("nmggp_node")
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
