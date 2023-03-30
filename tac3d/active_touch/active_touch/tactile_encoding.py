from collections import deque
import threading
from typing import Optional

import nengo
import numpy as np
import rclpy
import sensor_msgs.msg
import std_msgs.msg
from mujoco_interfaces.msg import Locus, RobotState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from active_touch.tacnet_helper import (
    default_intercepts,
    default_neuron,
    default_rates,
    image_height,
    image_width,
    image_size,
    layer_confs,
    conn_confs,
    learning_confs,
    normalize,
    gen_transform,
)


class TactileEncoding(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._layers: Optional[list[nengo.Node | nengo.Ensemble]] = None
        self._conns: Optional[list[nengo.Connection]] = None
        self._probes: Optional[list[nengo.Probe]] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._tactile_data = np.zeros(image_size)
        self._robot_state = deque(maxlen=40)

        # Create encoder network
        self.create_network()

        self._rs_sub = self.create_subscription(
            RobotState,
            "mujoco_simulator/robot_state",
            self.subscribe_rs,
            qos_profile_sensor_data,
        )
        self._sensor_sub = self.create_subscription(
            Locus,
            "mujoco_simulator/tactile_sensor",
            self.subscribe_sensor,
            qos_profile_sensor_data,
        )
        self._img_pub = self.create_publisher(
            sensor_msgs.msg.Image, "active_touch/tacnet_output", 20
        )
        self._encoding_pub = self.create_publisher(
            Locus, "active_touch/tacnet_encoding", 20
        )
        self._pc2_pub = self.create_publisher(
            sensor_msgs.msg.PointCloud2, "active_touch/tacnet_pointcloud2", 20
        )

        # Start the threaded Nengo simulation
        self._simulator_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self._simulator_thread.start()

    def __del__(self):
        self._simulator_thread.join()
        self.destroy_subscription(self._sensor_sub)
        self.destroy_publisher(self._img_pub)
        self.destroy_publisher(self._encoding_pub)

    def input_func(self, t):
        if len(self._tactile_data) == 0:
            return np.zeros(image_size)
        return self._tactile_data.ravel()

    def state_func(self, t):
        if len(self._robot_state) == 0:
            return np.zeros(7)
        return self._robot_state.pop().ravel()

    def create_network(self):
        self._layers = dict()
        self._conns = dict()
        self._probes = dict()

        with nengo.Network("tacnet") as self._net:
            # Create layers
            for k, layer_conf in enumerate(layer_confs):
                layer_conf = dict(layer_conf)  # Copy layer configuration
                name = layer_conf.pop("name")
                n_neurons = layer_conf.pop("n_neurons", 1)
                dimensions = layer_conf.pop("dimensions", 1)
                encoders = layer_conf.pop(
                    "encoders", nengo.dists.ScatteredHypersphere(surface=True)
                )
                intercepts = layer_conf.pop("intercepts", default_intercepts)
                max_rates = layer_conf.pop("max_rates", default_rates)
                radius = layer_conf.pop("radius", 1.0)
                neuron_type = layer_conf.pop("neuron", default_neuron)
                on_chip = layer_conf.pop("on_chip", False)
                block = layer_conf.pop("block", None)
                output = layer_conf.pop("output", None)
                size_in = layer_conf.pop("size_in", None)

                assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(layer_conf)
                )

                if neuron_type is None or n_neurons == 0:
                    assert not on_chip, "Nodes can only be run off-chip"
                    match name:
                        case "input":
                            output = self.input_func
                        case "state":
                            output = self.state_func
                    layer = nengo.Node(output=output, label=name)
                    self._layers[name] = layer
                    self._probes[name] = nengo.Probe(
                        layer, synapse=0.01, label="%s_probe" % name
                    )
                else:
                    layer = nengo.Ensemble(
                        n_neurons,
                        dimensions=dimensions,
                        radius=radius,
                        encoders=encoders,
                        intercepts=intercepts,
                        max_rates=max_rates,
                        neuron_type=neuron_type,
                        normalize_encoders=True,
                        label=name,
                    )
                    self._layers[name] = layer
                    self._layers[name + "_neurons"] = layer.neurons

                    # Add a probe so we can measure individual layer rates
                    self._probes[name] = nengo.Probe(
                        layer, synapse=0.01, label="%s_probe" % name
                    )
                    self._probes[name + "_neurons"] = nengo.Probe(
                        layer.neurons, synapse=0.01, label="%s_neurons_probe" % name
                    )

            for k, conn_conf in enumerate(conn_confs):
                conn_conf = dict(conn_conf)  # Copy connection configuration
                pre = conn_conf.pop("pre")
                post = conn_conf.pop("post")
                synapse = conn_conf.pop("synapse", None)
                solver = conn_conf.pop("solver", None)
                transform = conn_conf.pop("transform", gen_transform())
                learning_rule = conn_conf.pop("learning_rule", None)
                name = "{}2{}".format(pre, post)

                assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(conn_conf)
                )

                conn = nengo.Connection(
                    self._layers[pre],
                    self._layers[post],
                    transform=transform(
                        (self._layers[post].size_in, self._layers[pre].size_in)
                    ),
                    synapse=synapse,
                    label=name,
                )
                if solver:
                    conn.solver = solver
                if learning_rule:
                    conn.learning_rule_type = learning_rule
                self._conns[name] = conn

                self._probes[name] = nengo.Probe(
                    conn, "weights", synapse=0.01, label="weights_{}".format(name)
                )

            # Connect learning rule
            for k, learning_conf in enumerate(learning_confs):
                learning_conf = dict(learning_conf)
                pre = learning_conf.pop("pre")
                post = learning_conf.pop("post")
                transform = learning_conf.pop(
                    "transform", gen_transform("identity_excitation")
                )
                nengo.Connection(
                    self._layers[pre],
                    self._conns[post].learning_rule,
                    transform=transform,
                )

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                output = self._sim.data[self._probes["output"]][-1]
                coding = self._sim.data[self._probes["coding"]][-1]
                # Publish the reconstruction
                self.publish_touch(output, coding)

    def publish_touch(self, image: np.ndarray, encoding: np.ndarray):
        image = normalize(image).tolist()
        img_msg = sensor_msgs.msg.Image()
        img_msg.header = std_msgs.msg.Header(frame_id="world")
        img_msg.height = image_height
        img_msg.width = image_width
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = True
        img_msg.step = image_width
        img_msg.data = image
        self._img_pub.publish(img_msg)

        height, width = encoding.shape
        encoding = encoding.tolist()
        ec_msg = Locus()
        ec_msg.header = std_msgs.msg.Header(frame_id="world")
        ec_msg.height = height
        ec_msg.width = width
        ec_msg.data = encoding
        self._encoding_pub.publish(ec_msg)

    def publish_pc2(self, points: np.ndarray):
        ros_dtype = sensor_msgs.msg.PointField.FLOAT32
        dtype = np.float32
        data = points.astype(dtype).tobytes()
        itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

        data = points.astype(dtype).tobytes()
        fields = [
            sensor_msgs.PointField(
                name=n, offset=i * itemsize, datatype=ros_dtype, count=1
            )
            for i, n in enumerate("xyz")
        ]

        msg = sensor_msgs.msg.PointCloud2(
            header=std_msgs.msg.Header(frame_id="world"),
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=itemsize * points.shape[1],
            row_step=itemsize * points.shape[0] * points.shape[1],
            data=data,
        )
        self._pc2_pub.publish(msg)

    def subscribe_rs(self, msg: RobotState):
        # Update the robot state
        xpos = msg.site_position
        xquat = msg.site_quaternion
        pose = np.concatenate([xpos, xquat])
        self._robot_state.appendleft(pose)

    def subscribe_sensor(self, msg: Locus):
        # Update the tactile data
        self._tactile_data = np.asarray(msg.data)


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = TactileEncoding("tactile_encoding_node")
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
