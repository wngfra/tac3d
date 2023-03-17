import numpy as np
import rclpy
import std_msgs.msg
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

from mujoco_interfaces.msg import MotorSignal

_FREQ = 1000
t = 0


class CartesianMotion(Node):
    def __init__(self, qos_profile):
        super().__init__("cartesian_motion_node")
        self.i = 0
        self.pub = self.create_publisher(
            MotorSignal, "/tac3d/mujoco_simulator/motor_signal", qos_profile
        )

        timer_period = 1.0 / _FREQ
        self.tmr = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        global t
        t += 0.1
        msg = MotorSignal()
        msg.header = std_msgs.msg.Header()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()

        signal = np.array(
            [
                np.random.normal(0, 0.5, size=1),
                np.random.normal(0, 0.5, size=1),
                np.random.normal(0, 0.5, size=1),
                0,
                0,
                0,
            ],
            dtype=np.float64,
        )
        msg.spike_signal = signal.tolist()
        self.pub.publish(msg)


def main(argv=None):
    rclpy.init(args=argv)

    custom_qos_profile = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)
    executor = rclpy.executors.MultiThreadedExecutor()
    node = CartesianMotion(custom_qos_profile)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
