import threading
import weakref
from collections import deque
from typing import Optional

import mujoco
import numpy as np
import rclpy
import sensor_msgs.msg
import std_msgs.msg
from mujoco import viewer
from mujoco_interfaces.msg import Locus, MotorSignal, RobotState
from rclpy.lifecycle import (
    Node,
    Publisher,
    State,
    TransitionCallbackReturn,
)
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from mujoco_ros2.sim_helper import normalize, qpo_from_site_xpos

_DEFAULT_XML_PATH = "/workspace/src/tac3d/models/scene.xml"
_HEIGHT, _WIDTH = 15, 15
_TIMER_RATE = 200
_IKSITE_TYPE = 2
_CONTROL_STEP = 1e-3


class Simulator(Node):
    def __init__(self, node_name: str, xml_path: str):
        """Construct the lifecycle simulator node.

        Args:
            node_name (str): Name of the simulator node.
            xml_path (str): Path to the MuJoCo XML file.
        """
        super().__init__(node_name)
        self._xml_path = xml_path
        self._m: Optional[mujoco.MjModel] = None
        self._d: Optional[mujoco.MjData] = None
        self._viewer_thread: Optional[threading.Thread] = None

        self._img_pub: Optional[Publisher] = None
        self._locus_pub: Optional[Publisher] = None
        self._rs_pub: Optional[Publisher] = None
        self._ms_sub: Optional[Subscription] = None
        self._timer: Optional[Timer] = None

        # Store motor controls
        self._ctrls = deque(maxlen=20)
        self._default_ctrl: Optional[np.ndarray] = None

        if self.trigger_configure() == TransitionCallbackReturn.SUCCESS:
            self.trigger_activate()

    @property
    def m(self):
        return weakref.proxy(self._m)

    @property
    def d(self):
        return weakref.proxy(self._d)

    @property
    def sensordata(self):
        return self._d.sensordata

    @property
    def time(self):
        return self._d.time

    def controller_callback(self, m: mujoco.MjModel, d: mujoco.MjData):
        """Controller callback function to set motor signals."""
        # TODO include IK, numerical IK iteration is too slow for spiking control
        if len(self._ctrls) > 0:
            cmd = self._ctrls.pop()
            d.ctrl[: len(cmd)] += _CONTROL_STEP * cmd

    def reset_simulator(self, key_id: int):
        """Reset the MjData key frame by id.

        Args:
            key_id (int): Key frame id as defined in XML file.
        """
        mujoco.mj_resetDataKeyframe(self._m, self._d, key_id)
        mujoco.mj_forward(self._m, self._d)
        self._default_ctrl = self._d.ctrl

    def install_control(self, msg):
        """Install the control if there is at least one non-zero command.

        Args:
            msg (mujoco_interfaces.msg.MotorSignal): 6D MotorSignal message
        """
        signal = msg.spike_signal
        if np.any(signal):
            self._ctrls.appendleft(signal)

    def publish_sensordata(self):
        """Publish a new message when enabled."""
        header = std_msgs.msg.Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        sensor_msg = Locus()
        sensor_msg.header = header
        sensor_msg.height = _HEIGHT
        sensor_msg.width = _WIDTH
        sensor_msg.data = self.sensordata.tolist()
        self._locus_pub.publish(sensor_msg)

        img_msg = sensor_msgs.msg.Image()
        img_msg.header = header
        img_msg.height = _HEIGHT
        img_msg.width = _WIDTH
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = True
        img_msg.step = _WIDTH
        img_msg.data = normalize(self.sensordata).tolist()
        self._img_pub.publish(img_msg)

        rs_msg = RobotState()
        rs_msg.header = header
        rs_msg.joint_name = [self._m.joint(i).name for i in range(7)]
        rs_msg.joint_position = self._d.qpos[:7].tolist()
        rs_msg.joint_velocity = self._d.qvel[:7].tolist()
        IKSite_ids = [
            i for i in range(self._m.nsite) if self._m.site(i).type == _IKSITE_TYPE
        ]
        rs_msg.site_name = [self._m.site(j).name for j in IKSite_ids]
        rs_msg.site_position = self._d.site_xpos[IKSite_ids].flatten().tolist()
        rs_msg.site_quaternion = (
            np.asarray(
                [
                    R.from_matrix(self._d.site_xmat[j].reshape((3, 3))).as_quat()
                    for j in IKSite_ids
                ]
            )
            .flatten()
            .tolist()
        )
        self._rs_pub.publish(rs_msg)

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure the node, after a configuring transition is requested.
        on_configure callback is being called when the lifecycle node enters the "configuring" state.

        Args:
            state (State): Previous state.

        Returns:
            TransitionCallbackReturn: The state machine either invokes a transition to the "inactive" state or stays in "unconfigured" depending on the return value.
            TransitionCallbackReturn.SUCCESS transitions to "inactive".
            TransitionCallbackReturn.FAILURE transitions to "unconfigured".
            TransitionCallbackReturn.ERROR or any uncaught exceptions to "errorprocessing"
        """
        try:
            self._m = mujoco.MjModel.from_xml_path(self._xml_path)
            self._d = mujoco.MjData(self._m)
            self.reset_simulator(0)
            self.get_logger().info("Loaded MuJoCo XML from {}.".format(self._xml_path))
        except ValueError:
            self.get_logger().error("Error loading XML from {}".format(self._xml_path))
            return TransitionCallbackReturn.ERROR

        # Create the GUI thread
        self._viewer_thread = threading.Thread(
            target=viewer.launch,
            args=(
                self._m,
                self._d,
            ),
        )
        # Create tactile image publisher
        self._img_pub = self.create_lifecycle_publisher(
            sensor_msgs.msg.Image,
            "mujoco_simulator/tactile_image",
            qos_profile_sensor_data,
        )
        # Create tactile signal (locus) publisher
        self._locus_pub = self.create_lifecycle_publisher(
            Locus,
            "mujoco_simulator/tactile_sensor",
            qos_profile_sensor_data,
        )
        # Create robot state publisher
        self._rs_pub = self.create_lifecycle_publisher(
            RobotState,
            "mujoco_simulator/robot_state",
            20,
        )
        # Create motor signal subscription
        self._ms_sub = self.create_subscription(
            MotorSignal,
            "mujoco_simulator/motor_signal",
            self.install_control,
            qos_profile_sensor_data,
        )
        # Create sensor data timer for the above publishers
        self._timer = self.create_timer(1.0 / _TIMER_RATE, self.publish_sensordata)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node, after a activating transition is requested.

        Args:
            state (State): Previous state.

        Returns:
            TransitionCallbackReturn: _description_
        """
        self._viewer_thread.start()
        self.get_logger().info("Simulation started successfully.")
        callback = self.controller_callback
        mujoco.set_mjcb_control(callback)
        return super().on_activate(state)

    def on_error(self, state: State) -> TransitionCallbackReturn:
        """_summary_

        Args:
            state (State): Previous state.

        Returns:
            TransitionCallbackReturn: _description_
        """
        self.get_logger().error("Error running the node.")
        return TransitionCallbackReturn.ERROR

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node and threads, and remove MuJoCo objects, after a shuttingdown transition is requested.

        Args:
            state (State): Previous state.

        Returns:
            TransitionCallbackReturn: SUCCESS
        """
        try:
            self.get_logger().info("Node is shutting down.")
            self.destroy_publisher(self._locus_pub)
            self.destroy_timer(self._timer)

            self._viewer_thread.join()

            del self._m
            del self._d

            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error("Failed to shutdown the node {}".format(e))
            return TransitionCallbackReturn.FAILURE


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = Simulator("mujoco_simulator_node", _DEFAULT_XML_PATH)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
