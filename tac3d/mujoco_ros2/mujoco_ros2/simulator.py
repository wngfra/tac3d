import weakref
import threading
from typing import Optional

import numpy as np
import mujoco
from mujoco import viewer

import rclpy
from rclpy.lifecycle import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.timer import Timer

import sensor_msgs.msg
from mujoco_interfaces.msg import Locus


_DEFAULT_XML_PATH = "/workspace/src/tac3d/models/scene.xml"
_HEIGHT, _WIDTH = 15, 15


def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


class Simulator(Node):
    def __init__(self, node_name: str, xml_path: str):
        """Construct the lifecycle simulator node.

        Args:
            node_name (str): Name of the simulator node.
            xml_path (str): Path to the MuJoCo XML file.
        """
        super().__init__(node_name)
        self._xml_path = xml_path
        self._pub: Optional[Publisher] = None
        self._img_pub: Optional[Publisher] = None
        self._timer: Optional[Timer] = None
        self._m: Optional[mujoco.MjModel] = None
        self._d: Optional[mujoco.MjData] = None

        if self.trigger_configure() == TransitionCallbackReturn.SUCCESS:
            self.trigger_activate()

    @property
    def m(self) -> weakref.ref:
        return weakref.ref(self._model)

    @property
    def d(self) -> weakref.ref:
        return weakref.ref(self._d)

    @property
    def sensordata(self):
        return self._d.sensordata.astype(np.float32)

    @property
    def time(self):
        return self._d.time

    def control(self, controller_callback=None):
        """Install controller callback.

        Args:
            controller_callback (function, optional): Control callback function, set to None to uninstall any controller. Defaults to None.
        """
        mujoco.set_mjcb_control(controller_callback)

    def reset_simulator(self, key_id: int):
        """Reset the MjData key frame by id.

        Args:
            key_id (int): Key frame id as defined in XML file.
        """
        mujoco.mj_resetDataKeyframe(self._m, self._d, key_id)
        mujoco.mj_forward(self._m, self._d)

    def publish_sensordata(self):
        """Publish a new message when enabled."""
        msg = Locus()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = _HEIGHT
        msg.width = _WIDTH
        msg.data = self.sensordata.tolist()
        self._pub.publish(msg)

        img_msg = sensor_msgs.msg.Image()
        img_msg.header = msg.header
        img_msg.height = _HEIGHT
        img_msg.width = _WIDTH
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = True
        img_msg.step = _WIDTH
        img_msg.data = normalize(self.sensordata).tolist()
        self._img_pub.publish(img_msg)

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

        # Launch the GUI thread
        self._viewer_thread = threading.Thread(
            target=viewer.launch,
            args=(
                self._m,
                self._d,
            ),
        )

        self._pub = self.create_lifecycle_publisher(
            Locus, "mujoco_simulator/tactile_sensor", 10
        )
        self._img_pub = self.create_lifecycle_publisher(
            sensor_msgs.msg.Image, "mujoco_simulator/tactile_image", 10
        )
        self._timer_ = self.create_timer(5e-3, self.publish_sensordata)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node, after a activating transition is requested.

        Args:
            state (State): _description_

        Returns:
            TransitionCallbackReturn: _description_
        """
        self._viewer_thread.start()
        self.get_logger().info("Simulation started successfully.")
        return super().on_activate(state)

    def on_error(self, state: State) -> TransitionCallbackReturn:
        """_summary_

        Args:
            state (State): _description_

        Returns:
            TransitionCallbackReturn: _description_
        """
        self.get_logger().error("Error running the node.")
        return TransitionCallbackReturn.ERROR

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shut down the node, after a shuttingdown transition is requested.

        Args:
            state (State): _description_

        Returns:
            TransitionCallbackReturn: _description_
        """
        self.get_logger().info("Node is shutting down.")
        self.destroy_publisher(self._pub)
        self.destroy_timer(self._timer)

        self._viewer_thread.join()

        del self._m
        del self._d

        return TransitionCallbackReturn.SUCCESS


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
