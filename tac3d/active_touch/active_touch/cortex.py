from collections import deque

import numpy as np
import rclpy
import std_msgs.msg
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from mujoco_interfaces.msg import Locus, MotorSignal, RobotState

_SIM_RATE = 200
_FREQ = 100
t = 0


class Cortex(Node):
    def __init__(self, qos_profile):
        super().__init__("cortex_node")
        self._contact_force = 0.0
        self._ee_pose = deque(maxlen=int(_SIM_RATE * 0.2))

        self.i = 0

        # Motor signal publisher
        self._ms_pub = self.create_publisher(
            MotorSignal, "mujoco_simulator/motor_signal", qos_profile
        )
        # Robot state subscription
        self._rs_sub = self.create_subscription(
            RobotState, "mujoco_simulator/robot_state", self.subscribe_rs, 10
        )
        # Tactile encoding subscription
        self._te_sub = self.create_subscription(
            Locus, "active_touch/tacnet_encoding", self.subscribe_te, 10
        )
        # Timer for publishing motor signal
        timer_period = 1.0 / _FREQ
        self.tmr = self.create_timer(timer_period, self.timer_callback)

        # Desired pose
        self._desired_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def subscribe_rs(self, msg):
        xpos = msg.site_position
        xquat = msg.site_quaternion
        xrpy = R.from_quat(xquat).as_euler("xyz", degrees=True)
        pose = np.concatenate([xpos, xrpy])
        self._ee_pose.appendleft(pose)

        self._contact_force = msg.contact_force

    def subscribe_te(self, msg):
        pass

    def timer_callback(self):
        global t
        t += 0.1
        msg = MotorSignal()
        msg.header = std_msgs.msg.Header()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()

        signal = np.zeros(6, dtype=np.float64)
        # FIXME PID control for constant contact force
        signal[2] -= np.exp(-2e-3 * t)
        signal[1] = np.sin(0.5 * t)
        msg.spike_signal = signal.tolist()
        self._ms_pub.publish(msg)


def main(argv=None):
    rclpy.init(args=argv)

    custom_qos_profile = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)
    executor = rclpy.executors.MultiThreadedExecutor()
    node = Cortex(custom_qos_profile)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
