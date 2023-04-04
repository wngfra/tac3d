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
    dim_states,
    layer_confs,
    conn_confs,
    learning_confs,
    normalize,
    gen_transform,
    log,
)

_DURATION = 0.5


class Tacnet(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._layers: Optional[dict[str, nengo.Node | nengo.Ensemble]] = None
        self._conns: Optional[dict[str, nengo.Connection]] = None
        self._probes: Optional[dict[str, nengo.Probe]] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._tactile_data = np.zeros(image_size)
        self._touch_sites = deque(maxlen=40)
        self._last_touch: Optional[np.ndarray] = None

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
        self.destroy_subscription(self._rs_sub)
        self.destroy_subscription(self._sensor_sub)
        self.destroy_publisher(self._img_pub)
        self.destroy_publisher(self._encoding_pub)
        self.destroy_publisher(self._pc2_pub)

    def input_func(self, t):
        if len(self._tactile_data) == 0:
            return np.zeros(image_size)
        return self._tactile_data.ravel()

    def state_func(self, t):
        # TODO consider the working memory decay
        if len(self._touch_sites) == 0:
            return np.zeros(dim_states)
        self._last_touch = self._touch_sites.pop().ravel()
        return self._last_touch

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
                output = layer_conf.pop("output", None)
                size_in = layer_conf.pop("size_in", None)
                size_out = layer_conf.pop("size_out", None)

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

                    layer = nengo.Node(
                        output=output, size_in=size_in, size_out=size_out, label=name
                    )
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
                name = learning_conf.pop("name")
                pre = learning_conf.pop("pre")
                post = learning_conf.pop("post")
                transform = learning_conf.pop(
                    "transform", gen_transform("identity_excitation")
                )
                self._conns[name] = nengo.Connection(
                    self._layers[pre],
                    self._conns[post].learning_rule,
                    transform=transform,
                )

    def run_simulation(self):
        with nengo.Simulator(self._net, progress_bar=False) as self._sim:
            while rclpy.ok():
                self._sim.run(_DURATION)
                output = self._sim.data[self._probes["output_ens"]][-1]
                coding = self._sim.data[self._probes["coding_ens"]][-1]
                # Publish the reconstruction
                self.publish(output, coding, self._last_touch)

    def publish(
        self, image: np.ndarray, encoding: np.ndarray, points: np.ndarray = None
    ):
        header = std_msgs.msg.Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        # Prepare reconstruction tactile image
        image = normalize(image).tolist()
        img_msg = sensor_msgs.msg.Image()
        img_msg.header = header
        img_msg.height = image_height
        img_msg.width = image_width
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = True
        img_msg.step = image_width
        img_msg.data = image
        self._img_pub.publish(img_msg)

        # Prepare tactile percept as Locus
        dim = int(np.sqrt(encoding.size))
        encoding = encoding.tolist()
        ec_msg = Locus()
        ec_msg.header = header
        ec_msg.height = dim
        ec_msg.width = dim
        ec_msg.data = encoding
        self._encoding_pub.publish(ec_msg)

        # Prepare tactile spatial memory as point cloud
        if points is not None:
            points = points[:3]
            ros_dtype = sensor_msgs.msg.PointField.FLOAT32
            dtype = np.float32
            itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.
            pc2_msg = sensor_msgs.msg.PointCloud2(
                header=header,
                height=1,
                width=1,
                is_dense=True,
                is_bigendian=False,
                fields=[
                    sensor_msgs.msg.PointField(
                        name=n, offset=i * itemsize, datatype=ros_dtype, count=1
                    )
                    for i, n in enumerate("xyz")
                ],
                point_step=itemsize * 1,
                row_step=itemsize * points.shape[0] * 1,
                data=points.astype(dtype).tobytes(),
            )
            # FIXME cannot be visualized by rviz2
            # self._pc2_pub.publish(pc2_msg)

    def subscribe_rs(self, msg: RobotState):
        # Update the robot state
        xpos = msg.site_position
        xquat = msg.site_quaternion
        pose = np.concatenate([xpos, xquat])
        theta_2 = np.arccos(xquat[0])
        gamma = 2 * np.arccos(xquat[3] / np.sin(theta_2))
        self._touch_sites.appendleft(pose)

    def subscribe_sensor(self, msg: Locus):
        # Update the tactile data
        self._tactile_data = np.asarray(msg.data)


def main(args=None):
    rclpy.init(args=args)

    executor = rclpy.executors.MultiThreadedExecutor()
    node = Tacnet("tacnet_node")
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
