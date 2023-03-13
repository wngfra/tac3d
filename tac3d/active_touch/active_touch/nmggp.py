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
    HEIGHT,
    WIDTH,
    default_intercepts,
    default_neuron,
    default_transform,
    layer_confs,
    normalize,
)

_SIZE_IN = HEIGHT*WIDTH


class NMGGP(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._conns: Optional[list[nengo.Connection]] = None
        self._probes: Optional[list[nengo.Probe]] = None
        self._transforms: Optional[list[nengo.transforms.Transform]] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._tactile_data = np.zeros(_SIZE_IN)

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
        self._conns = []
        self._probes = []
        self._transforms = []

        with nengo.Network() as self._net:
            stim = nengo.Node(output=self.input_func, label="Stimulus Node")
            shape_in = nengo.transforms.ChannelShape((_SIZE_IN,))
            x = stim

            # Create layers
            for k, layer_conf in enumerate(layer_confs):
                layer_conf = dict(layer_conf)  # copy, so we don't modify the original
                name = layer_conf.pop("name")
                intercepts = layer_conf.pop("intercepts", default_intercepts)
                max_rates = layer_conf.pop("max_rates", None)
                neuron_type = layer_conf.pop("neuron", default_neuron)
                on_chip = layer_conf.pop("on_chip", True)
                block = layer_conf.pop("block", None)
                recurrent = layer_conf.pop("recurrent", False)
                learning_rule = layer_conf.pop("learning_rule", None)
                recurrent_learning_rule = layer_conf.pop(
                    "recurrent_learning_rule", None
                )

                # Create layer transform
                if "filters" in layer_confs:
                    # Convolutional layer
                    pass
                else:
                    # Dense layer
                    n_neurons = layer_conf.pop("n_neurons")
                    shape_out = nengo.transforms.ChannelShape((n_neurons,))
                    if name != "input_layer":
                        transform = nengo.Dense(
                            (shape_out.size, shape_in.size),
                            init=default_transform,
                        )
                    else:
                        transform = 1
                    if recurrent:
                        transform_reccurent = nengo.Dense(
                            (shape_in.size, shape_out.size),
                            init=default_transform,
                        )

                    loc = "chip" if on_chip else "host"

                assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(layer_conf)
                )

                if neuron_type is None:
                    assert not on_chip, "Nodes can only be run off-chip"
                    y = nengo.Node(size_in=shape_out.size, label=name)
                else:
                    ens = nengo.Ensemble(
                        shape_out.size,
                        1,
                        max_rates=max_rates,
                        intercepts=intercepts,
                        neuron_type=neuron_type,
                        label=name,
                    )
                    y = ens.neurons

                    # Add a probe so we can measure individual layer rates
                    probe = nengo.Probe(y, synapse=0.01, label="%s_p" % name)
                    self._probes.append(probe)

                conn = nengo.Connection(
                    x, y, transform=transform, learning_rule_type=learning_rule
                )
                self._conns.append(conn)
                self._transforms.append(transform)

                if recurrent:
                    conn_recurrent = nengo.Connection(
                        y,
                        x,
                        transform=transform_reccurent,
                        learning_rule_type=recurrent_learning_rule,
                    )
                    self._conns.append(conn_recurrent)
                    self._transforms.append(transform_reccurent)

                x = y
                shape_in = shape_out

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                output = self._sim.data[self._probes[0]][-1]
                # Publish the tactile percept
                self.publish(normalize(output))

    def publish(self, data: np.ndarray or list):
        if isinstance(data, np.ndarray):
            data = data.tolist()
        msg = sensor_msgs.msg.Image()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = HEIGHT
        msg.width = WIDTH
        msg.encoding = "mono8"
        msg.is_bigendian = True
        msg.step = WIDTH
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
