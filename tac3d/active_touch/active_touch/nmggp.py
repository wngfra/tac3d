import pdb
import threading
from typing import Optional

import nengo
import numpy as np
import rclpy
import sensor_msgs.msg
from mujoco_interfaces.msg import Locus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

_HEIGHT, _WIDTH = 15, 15
_SIZE_IN = _HEIGHT * _WIDTH
UINT8_MAX, UINT8_MIN = np.iinfo(np.uint8).max, np.iinfo(np.uint8).min
_RATE = 100
_THRESHOLD = -0.5
_ENS_PARAMS = dict(
    neuron_type=nengo.LIFRate(),
    radius=1.0,
    intercepts=nengo.dists.Uniform(_THRESHOLD, -0.5),
    max_rates=nengo.dists.Uniform(_RATE, _RATE),
)


def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


def log(node: Node, x):
    node.get_logger().info("data: {}".format(x))


class NMGGP(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._conn: Optional[nengo.Connection] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._tactile_data = np.zeros(_SIZE_IN)

        # Create a Nengo network
        self.create_network()

        self._sub = self.create_subscription(
            Locus,
            "mujoco_simulator/tactile_sensor",
            self.subscribe,
            qos_profile_sensor_data,
        )
        self._pub = self.create_publisher(
            sensor_msgs.msg.Image, "active_touch/tactile_encoding", 10
        )

        # Start the threaded Nengo simulation
        self._simulator_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self._simulator_thread.start()

    def __del__(self):
        self._simulator_thread.join()
        self.destroy_subscription(self._sub)
        self.destroy_publisher(self._pub)

    def create_network(self):
        with nengo.Network() as self._net:
            self._stim = nengo.Node(
                output=self.input_func, size_in=_SIZE_IN, label="Stimulus Node"
            )
            inp = nengo.Ensemble(
                n_neurons=self._stim.size_out,
                dimensions=self._stim.size_out,
                **_ENS_PARAMS,
                label="Input Ensemble",
            )
            weights = np.ones((self._stim.size_out, inp.n_neurons))
            self._conn = nengo.Connection(
                self._stim, inp, synapse=0.01, transform=weights
            )
            self._probe = nengo.Probe(inp, label="Input Probe", synapse=None)

    def input_func(self, t, x):
        rate = self._tactile_data.flatten()
        rate[rate < 0] = 0
        rate *= _RATE

        return rate

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                output = self._sim.data[self._probe][-1]
                
                # Publish the tactile percept
                data = normalize(output)
                self.publish(data)

    def publish(self, data: np.ndarray or list):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        msg = sensor_msgs.msg.Image()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = _HEIGHT
        msg.width = _WIDTH
        msg.encoding = "mono8"
        msg.is_bigendian = True
        msg.step = _WIDTH
        msg.data = data
        self._pub.publish(msg)

    def subscribe(self, msg: Locus):
        # Update the tactile data
        self._tactile_data = np.asarray(msg.data)


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
