from collections import deque
import threading
from typing import Optional

import nengo
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from mujoco_interfaces.msg import Locus, RobotState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from active_touch.tacnet_helper import (
    default_intercepts,
    default_neuron,
    default_rates,
    stim_shape,
    layer_confs,
    conn_confs,
    learning_confs,
    normalize,
    gen_transform,
    log,
)

_DURATION = 0.1
NODE_CONFS = [
    # Subscriber configs
    dict(
        name="robot_state_sub",
        msg_type=RobotState,
        topic="mujoco_simulator/robot_state",
        qos_profile=qos_profile_sensor_data,
        callback="subscribe_robot_state",
    ),
    dict(
        name="tactile_sensor_sub",
        msg_type=Locus,
        topic="mujoco_simulator/tactile_sensor",
        qos_profile=qos_profile_sensor_data,
        callback="subscribe_tactile_sensor",
    ),
    # Publisher configs
    dict(
        name="tacnet_encoded_pub",
        msg_type=Image,
        topic="active_touch/tacnet/encoded",
    ),
    dict(
        name="tacnet_output_pub",
        msg_type=Locus,
        topic="active_touch/tacnet/output",
    ),
    dict(
        name="tacnet_point_pub",
        msg_type=PointStamped,
        topic="active_touch/tacnet/point",
    ),
]


class Tacnet(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._layers: Optional[dict[str, nengo.Node | nengo.Ensemble]] = None
        self._conns: Optional[dict[str, nengo.Connection]] = None
        self._probes: Optional[dict[str, nengo.Probe]] = None
        self._net: Optional[nengo.Network] = None
        self._sim: Optional[nengo.Simulator] = None
        self._mem = deque(maxlen=50)
        self._confs = dict()

        # Create encoder network
        self.create_network()

        # Create subscriber and publisher
        for conf in NODE_CONFS:
            conf = conf.copy()
            name = conf.pop("name")
            msg_type = conf.pop("msg_type")
            topic = conf.pop("topic")
            qos_profile = conf.pop("qos_profile", 20)
            callback = conf.pop("callback", None)
            if callback:
                sub = self.create_subscription(
                    msg_type, topic, getattr(self, callback), qos_profile
                )
                self._confs[name] = sub
            else:
                pub = self.create_publisher(msg_type, topic, qos_profile)
                self._confs[name] = pub

        # Start the threaded Nengo simulation
        self._simulator_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self._simulator_thread.start()

    def __del__(self):
        self._simulator_thread.join()
        for name, item in self._confs.items():
            if name.endswith("sub"):
                self.destroy_subscription(item)
            else:
                self.destroy_publisher(item)

    def input_func(self, t):
        # FIXME the poses are not yet memorized
        data = np.zeros(stim_shape)
        if len(self._mem) > 0:
            d = self._mem.pop()
            if "data" in d:
                data = d["data"]
        return data.ravel()

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
                    if isinstance(output, str):
                        output = getattr(self, output)

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
                dim_in = conn_conf.pop("dim_in", None)
                dim_out = conn_conf.pop("dim_out", None)
                synapse = conn_conf.pop("synapse", None)
                solver = conn_conf.pop("solver", None)
                transform = conn_conf.pop("transform", None)
                learning_rule = conn_conf.pop("learning_rule", None)
                name = "{}2{}".format(pre, post)

                assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
                    [name], list(conn_conf)
                )

                if dim_in is None:
                    pre_conn = self._layers[pre]
                else:
                    pre_conn = self._layers[pre][dim_in]
                if dim_out is None:
                    post_conn = self._layers[post]
                else:
                    post_conn = self._layers[post][dim_out]
                if callable(transform):
                    transform = transform((post_conn.size_in, pre_conn.size_in))

                conn = nengo.Connection(
                    self._layers[pre],
                    self._layers[post],
                    transform=transform,
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
        with nengo.Simulator(self._net, progress_bar=False, optimize=True) as self._sim:
            while rclpy.ok():
                self._sim.run(_DURATION)
                hidden = self._sim.data[self._probes["hidden_neurons"]][-1]
                output = self._sim.data[self._probes["output_neurons"]][-1]

                # Publish simulation results
                self.publish_image("tacnet_encoded_pub", hidden)
                # self.publish_output(output)
                self._sim.clear_probes()

    def publish_image(self, pub_name, image: np.ndarray):
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        # Prepare reconstruction tactile image
        dim = int(np.sqrt(image.size))
        img_msg = Image()
        img_msg.header = header
        img_msg.height = dim
        img_msg.width = dim
        img_msg.encoding = "mono8"
        img_msg.is_bigendian = True
        img_msg.step = dim
        img_msg.data = normalize(image).tolist()
        self._confs[pub_name].publish(img_msg)

    def publish_output(self, output: np.ndarray, point: np.ndarray = None):
        header = Header()
        header.frame_id = "world"
        header.stamp = self.get_clock().now().to_msg()

        # Prepare tactile percept as Locus
        dim = int(np.sqrt(output.size))
        output = output.tolist()
        ec_msg = Locus()
        ec_msg.header = header
        ec_msg.height = dim
        ec_msg.width = dim
        ec_msg.data = output
        # self._confs["te_pub"].publish(ec_msg)

        # Prepare tactile spatial memory as point cloud
        if point:
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.point.x = point[0]
            point_msg.point.y = point[1]
            point_msg.point.z = point[2]
            self._confs["tp_pub"].publish(point_msg)

    def subscribe_robot_state(self, msg: RobotState):
        # Update the robot state
        xpos = msg.site_position
        xquat = msg.site_quaternion
        pose = np.concatenate([xpos, xquat])
        theta_2 = np.arccos(xquat[0])
        gamma = 2 * np.arccos(xquat[3] / np.sin(theta_2))
        if len(self._mem) == 0 or "pose" in self._mem[-1]:
            self._mem.appendleft({"pose": pose})
        else:
            self._mem[-1]["pose"] = pose

    def subscribe_tactile_sensor(self, msg: Locus):
        # Update the tactile data
        data = np.asarray(msg.data)
        if len(self._mem) == 0 or "data" in self._mem[-1]:
            self._mem.appendleft({"data": data})
        else:
            self._mem[-1]["data"] = data


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
