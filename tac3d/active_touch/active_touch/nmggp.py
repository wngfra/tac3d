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
_MAX_RATES = 200


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
        self._conn: Optional[dict[str, nengo.Connection]] = None
        self._net: Optional[nengo.Network] = None
        self._probe: Optional[dict[str, nengo.Probe]] = None
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
        self._conn = dict()
        self._probe = dict()

        with nengo.Network() as self._net:
            self._stim = nengo.Node(
                output=self.input_func, size_in=_SIZE_IN, label="Stimulus Node"
            )
            inp = nengo.Ensemble(
                n_neurons=self._stim.size_out,
                dimensions=1,
                radius=1.0,
                intercepts=nengo.dists.Gaussian(0, 0.1),
                max_rates=_MAX_RATES * np.ones(self._stim.size_out),
                neuron_type=nengo.PoissonSpiking(nengo.LIFRate()),
                label="Input Ensemble",
            )
            hidden = nengo.Ensemble(n_neurons=120, dimensions=1)
            out = nengo.Ensemble(n_neurons=16, dimensions=1)

            self._conn["stim2inp"] = nengo.Connection(
                self._stim,
                inp.neurons,
                transform=1,
                synapse=None,
                label="stim2inp",
            )
            self._conn["inp2hidden"] = nengo.Connection(
                inp,
                hidden,
                learning_rule_type=nengo.PES(learning_rate=1e-4),
                label="inp2hidden",
            )
            self._conn["hidden2output"] = nengo.Connection(
                hidden,
                out,
                learning_rule_type=nengo.PES(learning_rate=1e-4),
                label="hidden2output",
            )

            self._probe["Input Spikes"] = nengo.Probe(
                inp.neurons, label="Input Spikes", synapse=0.01
            )
            self._probe["Hidden Spikes"] = nengo.Probe(
                hidden.neurons, label="Hidden Spikes", synapse=0.01
            )
            self._probe["Output Spikes"] = nengo.Probe(
                out.neurons, label="Output Spikes", synapse=0.01
            )

    def input_func(self, t, x):
        stimuli = self._tactile_data.flatten()
        return stimuli

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                output = self._sim.data[self._probe["Input Spikes"]][-1]
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
