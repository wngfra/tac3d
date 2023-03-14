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
from active_touch.net_helper import (
    image_height,
    image_width,
    image_size,
    layer_confs,
    conn_confs,
    normalize,
    gen_transform,
)

default_intercepts = nengo.dists.Choice([0, 0.1])


class NMGGP(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._layers: Optional[list[nengo.Node | nengo.Ensemble]] = None
        self._conns: Optional[list[nengo.Connection]] = None
        self._probes: Optional[list[nengo.Probe]] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._tactile_data = np.zeros(image_size)

        # Create encoder network
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

    def input_func(self, t):
        return self._tactile_data.ravel()

    def create_network(self):
        self._layers = dict()
        self._conns = dict()
        self._probes = dict()

        with nengo.Network("active touch") as self._net:
            stim = nengo.Node(output=self.input_func, label="stimulus_node")
            self._layers["stimulus_node"] = stim

            # Create layers
            for k, layer_conf in enumerate(layer_confs):
                layer_conf = dict(layer_conf)  # Copy layer configuration
                name = layer_conf.pop("name")
                n_neurons = layer_conf.pop("n_neurons")
                intercepts = layer_conf.pop("intercepts", default_intercepts)
                max_rates = layer_conf.pop("max_rates", None)
                radius = layer_conf.pop("radius", 1.0)
                neuron_type = layer_conf.pop("neuron")
                on_chip = layer_conf.pop("on_chip", True)
                block = layer_conf.pop("block", None)
                learning_rule = layer_conf.pop("learning_rule", None)
                loc = "chip" if on_chip else "host"

                assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(layer_conf)
                )

                if neuron_type is None:
                    assert not on_chip, "Nodes can only be run off-chip"
                    layer = nengo.Node(size_in=n_neurons, label=name)
                    self._layers[name] = layer
                else:
                    layer = nengo.Ensemble(
                        n_neurons,
                        1,
                        radius=radius,
                        intercepts=intercepts,
                        neuron_type=neuron_type,
                        label=name,
                    )
                    self._layers[name] = layer.neurons

                    # Add a probe so we can measure individual layer rates
                    probe = nengo.Probe(
                        self._layers[name], synapse=0.01, label="%s_probe" % name
                    )
                    self._probes[name] = probe

            for k, conn_conf in enumerate(conn_confs):
                conn_conf = dict(conn_conf)  # Copy connection configuration
                pre = conn_conf.pop("pre")
                post = conn_conf.pop("post")
                synapse = conn_conf.pop("synapse", 0)
                transform = conn_conf.pop("transform", gen_transform())
                learning_rule = conn_conf.pop("learning_rule", None)
                name = "conn_{}-{}".format(pre, post)

                assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(layer_conf)
                )

                conn = nengo.Connection(
                    self._layers[pre],
                    self._layers[post],
                    transform=transform(
                        (self._layers[post].size_in, self._layers[pre].size_in)
                    ),
                    learning_rule_type=learning_rule,
                    synapse=synapse,
                    label=name,
                )
                # transforms[name] = transform
                self._conns[name] = name

                probe = nengo.Probe(
                    conn, "weights", synapse=0.01, label="weights_{}".format(name)
                )
                self._probes[name] = probe

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                output = self._sim.data[self._probes["output_layer"]][-1]
                # Publish the tactile percept
                self.publish(normalize(output))

    def publish(self, data: np.ndarray or list):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        msg = sensor_msgs.msg.Image()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = image_height
        msg.width = image_width
        msg.encoding = "mono8"
        msg.is_bigendian = True
        msg.step = image_width
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
