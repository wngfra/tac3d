import threading
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from mujoco_ros2.sim_helper import normalize, nullspace_method

_DEFAULT_XML_PATH = "/workspace/src/tac3d/models/scene.xml"
_HEIGHT, _WIDTH = 15, 15
_TIMER_RATE = 200
_IKSITE_TYPE = 2
_CTRL_SCALE = 1e-3
_QOS_PROFILE = QoSProfile(depth=20, reliability=QoSReliabilityPolicy.RELIABLE)

_SITE_NAME = "attachment_site"
_JOINTS = ["joint%d" % (i + 1) for i in range(7)]


@dataclass
class IK_Solver:
    jac: np.ndarray = None
    jacp: np.ndarray = None
    jacr: np.ndarray = None
    site_id: int = None
    dof_indices: np.ndarray = None
    ref_pose: np.ndarray = None


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

    def init_ik(self):
        self.ik = IK_Solver()

        dtype = self._d.qpos.dtype
        self.ik.jac = np.empty((6, self._m.nv), dtype=dtype)
        self.ik.jacp, self.ik.jacr = self.ik.jac[:3], self.ik.jac[3:]
        self.ik.site_id = mujoco.mj_name2id(
            self._m, mujoco.mjtObj.mjOBJ_SITE, _SITE_NAME
        )

        indexer = [self._m.joint(jn).jntid for jn in _JOINTS]
        self.ik.dof_indices = np.asarray(indexer).flatten()
        self.ik.ref_xpos = self._d.site_xpos[self.ik.site_id]
        self.ik.ref_xmat = self._d.site_xmat[self.ik.site_id]

    def controller_callback(self, m: mujoco.MjModel, d: mujoco.MjData):
        """Controller callback function to set motor signals.

        Args:
            m (mujoco.MjModel): Robot model.
            d (mujoco.MjData): Binded data.
        """

        # Compute translational error
        dtype = d.qpos.dtype
        err = np.zeros(6, dtype=dtype)
        err_pos, err_rot = err[:3], err[3:]
        err_pos[:] = self.ik.ref_xpos - d.site_xpos[self.ik.site_id]

        # Compute rotation error
        site_xmat = d.site_xmat[self.ik.site_id]
        site_xquat = np.empty(4, dtype=dtype)
        target_quat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)
        mujoco.mju_mat2Quat(site_xquat, site_xmat)
        mujoco.mju_mat2Quat(target_quat, self.ik.ref_xmat)
        mujoco.mju_negQuat(neg_site_xquat, site_xquat)
        mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
        mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
        
        # Add control to error vector.
        if len(self._ctrls) > 0:
            ctrl = self._ctrls.pop() * _CTRL_SCALE
            err_pos += ctrl[:3]
            err_rot += ctrl[3:]
        if np.linalg.norm(err_pos) > 1e-6:
            # Compute IK and set control
            mujoco.mj_jacSite(m, d, self.ik.jacp, self.ik.jacr, self.ik.site_id)
            jac_joints = self.ik.jac[:, self.ik.dof_indices]
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=1e-2
            )
            d.ctrl[: len(update_joints)] += update_joints

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
        signal = np.asarray(msg.spike_signal)
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
        rs_msg.contact_force = np.mean(self.sensordata)
        rs_msg.n_contacts = self._d.ncon
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
            _QOS_PROFILE,
        )
        # Create tactile signal (locus) publisher
        self._locus_pub = self.create_lifecycle_publisher(
            Locus,
            "mujoco_simulator/tactile_sensor",
            _QOS_PROFILE,
        )
        # Create robot state publisher
        self._rs_pub = self.create_lifecycle_publisher(
            RobotState,
            "mujoco_simulator/robot_state",
            _QOS_PROFILE,
        )
        # Create motor signal subscription
        self._ms_sub = self.create_subscription(
            MotorSignal,
            "mujoco_simulator/motor_signal",
            self.install_control,
            _QOS_PROFILE,
        )
        # Create sensor data timer for the above publishers
        self._timer = self.create_timer(1.0 / _TIMER_RATE, self.publish_sensordata)

        self.init_ik()

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
