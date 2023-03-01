import threading
import pdb
import numpy as np
import nengo

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from mujoco_interfaces.msg import Locus

_HEIGHT, _WIDTH = 15, 15
_SIZE_IN = _HEIGHT * _WIDTH


class NMGGP(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._net = nengo.Network()
        self._sim = nengo.Simulator(self._net)

        self._sub = self.create_subscription(
            Locus,
            "mujoco_simulator/tactile_sensor",
            self.subscribe,
            qos_profile_sensor_data,
        )
        self._pub = self.create_publisher(Locus, "active_touch/tactile_encoding", 10)

        # Create a Nengo SNN model and start the threaded simulation
        self.create_network()
        self._simulation_thread = threading.Thread(
            target=self.run_simulation, daemon=True
        )
        self._simulation_thread.start()

    def __del__(self):
        self._simulation_thread.join()
        self.destroy_subscription(self._sub)
        self.destroy_publisher(self._pub)

    def create_network(self):
        with self._net:
            self._inp = nengo.Node(size_in=_SIZE_IN)
            ens = nengo.Ensemble(36, self._inp.size_out)
            nengo.Connection(self._inp, ens)
            self._probe = nengo.Probe(ens)
    
    def poisson_output(self, x):
        rate = x.flatten()
        rate[rate < 0] = 0
        rate *= 100  # scale up the rates
        return np.random.poisson(rate)

    def run_simulation(self):
        with self._sim:
            while rclpy.ok():
                self._sim.run(0.1)
                pdb.set_trace()
                output = self._sim.data[self._probe][-1]
                # Do something with the output, e.g. publish it
                self.get_logger().info("Output: %s" % output)

    def subscribe(self, msg: Locus):
        # Update the stimulus for the Nengo model
        stimuli = msg.data
        pdb.set_trace()
        self._inp.output = self.poisson_output(stimuli)



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
