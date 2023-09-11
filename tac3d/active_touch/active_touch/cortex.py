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
    """Cortex node controls the robot using the tactile feedback."""

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
            RobotState, "mujoco_simulator/robot_state", self.subscribe_robot_state, 10
        )

        # Tactile encoding subscription
        topic_names_and_types = self.get_topic_names_and_types()
        for topic_name, _ in topic_names_and_types:
            if "tacnet/output" in topic_name:
                self.create_subscription(
                    Locus, topic_name, self.subscribe_tactile_output, 10
                )

        # Timer for publishing motor signal
        timer_period = 1.0 / _FREQ
        self.tmr = self.create_timer(timer_period, self.timer_callback)

        # Desired pose
        self._desired_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def subscribe_robot_state(self, msg):
        xpos = msg.site_position
        xquat = msg.site_quaternion
        xrpy = R.from_quat(xquat).as_euler("xyz", degrees=True)
        pose = np.concatenate([xpos, xrpy])
        self._ee_pose.appendleft(pose)

        self._contact_force = msg.contact_force

    def subscribe_tactile_output(self, msg):
        pass

    def timer_callback(self):
        global t
        t += 0.1
        msg = MotorSignal()
        msg.header = std_msgs.msg.Header()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()

        signal = np.zeros(6, dtype=np.float64)
        if self._contact_force < 1e-3:
            signal[2] -= 1
        else:
            signal[2] += -1 * (1e-2 - self._contact_force)

            signal[0] += 1 * np.cos(0.1 * np.pi * t)
            signal[1] += 1 * np.sin(0.1 * np.pi * t)

        msg.signal = signal.tolist()
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
