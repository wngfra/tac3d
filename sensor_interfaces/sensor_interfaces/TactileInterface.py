import multiprocessing
import struct
from collections import deque

import cv2
import nengo
import numpy as np
import pylab as plt
import rclpy
import serial
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from sensor_interfaces.DigitalProcessor import Grayscale

params = {
    'dt': 2.5e-2,
}


class TactileInterface(Node):
    def __init__(self):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('baudrate', 115200),
                ('device', 'cpu'),
                ('dim', [100, 100]),
                ('port', ''),
            ],
        )

        # Get parameters
        baudrate = self.get_parameter('tac3d.baudrate').value
        device = str(self.get_parameter('tac3d.device').value)
        dim = self.get_parameter('tac3d.dim').value
        port = str(self.get_parameter('tac3d.port').value)
        # Setup
        self._activated = True
        self._buffer = deque(maxlen=10)
        self._dim = dim
        self.bridge = CvBridge()
        
        try:
            if len(port) < 1:
                raise serial.SerialException("No serial port found.")
            # Initialize interface in serial mode
            self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            self._update_process = multiprocessing.Process(target=self._serial2buffer)
            self._update_process.start()
            self.get_logger().info('The sensor interface operating in serial mode.')
        except serial.SerialException as e:
            # Initialize interface in simulation mode
            # self._dsps = [Grayscale(keep_dim=False)]
            self._tacImg_pub = self.create_publisher(Image, '/sensors/tactile_sensors/image', 10)
            self._sub = self.create_subscription(Image, '/sensors/tactile_sensors/raw', self._subscriber_callback, 10)
            self.get_logger().warn('{}\nThe sensor interface operating in simulation mode.'.format(e))

        # Prepare SNN model and output publisher
        self.create_tacnet(n_neurons=[100, 12, 36])
        self.device = device
        self._timer = self.create_timer(params['dt'], self._timer_callback)
        self._net_pub = self.create_publisher(Float32MultiArray, '/perception/touch', 10)

    def __del__(self):
        self._activated = False

        if self.mode == 'serial':
            self._update_process.join()
            self._ser.close()
        else:
            self._sim.close()
            self._sub.destroy()

    def _timer_callback(self):
        self._sim.run(params['dt'])
        output = self._sim.data[self._probes[-1]].flatten()
        msg = Float32MultiArray(data=output)
        self._net_pub.publish(msg)

    def _subscriber_callback(self, msg):
        """Callback function for the tactile image subscriber.

        Args:
            msg (Image): Incoming message as Image.
        """
        self._dim = [msg.height, msg.width]
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            values = data.view('<f4').reshape(self._dim) / 1.0 * 255
            
        except ValueError as e:
            self.get_logger().error('Invalid data received from sensor: {}'.format(e))
            values = np.zeros(self._dim)
        
        if hasattr(self, '_dsps'):
            values = [dsp(values) for dsp in self._dsps]
        self._buffer.append(values)
        imgMsg = self.bridge.cv2_to_imgmsg(values)
        self._tacImg_pub.publish(imgMsg)
            
    def _serial2buffer(self):
        """Update the buffer from the serial port.
        """

        while self._activated:
            if not self._ser.is_open:
                self._ser.open()
            try:
                # Convert bytes to numpy int array
                stream = self._ser.readline().decode('utf-8')
                data = [int(d) for d in stream.split(' ')[:-1]]
                values = np.asarray(data, dtype=int).reshape(self._dim)
                self._buffer.append(values)
            except ValueError as e:
                self.get_logger().error('Invalid data received from sensor: {}'.format(e))

    def create_tacnet(self, n_neurons):
        """Create the TacNet(SNN) for asyncronous tactile perception.

        Args:
            n_neurons (list or int): Number of neurons in each layer.
            device (str, optional): Device to run nengo simulator. Defaults to 'cpu'.
        """

        self.net = nengo.Network(label='TacNet')
        self._probes = []
        with self.net:
            inp = nengo.Node(output=self.input_func)
            self._layers = [inp]
            for n in n_neurons:
                self._layers.append(nengo.Ensemble(
                    n_neurons=n, dimensions=inp.size_out, neuron_type=nengo.AdaptiveLIF()))

            for i in range(len(self._layers)-1):
                nengo.Connection(
                    self._layers[i], self._layers[i+1], synapse=None)

            for layer in self._layers:
                self._probes.append(nengo.Probe(layer, synapse=1e-2))

    @property
    def device(self):
        if not hasattr(self, '_device'):
            self.device = 'cpu'
        return self._device

    @device.setter
    def device(self, device):
        """Set device and create the simulator.

        Args:
            device (string): Name of the device to run the simulator.

        Raises:
            ModuleNotFoundError: If the package for the device is not installed.
            ValueError: If the device is not supported.
        """
        
        try:
            match device:
                case 'cpu':
                    from nengo import Simulator as Simulator
                case 'loihi':
                    from nengo_loihi import Simulator as Simulator
                case _:
                    raise ValueError('Invalid device: {}'.format(device))
        except ModuleNotFoundError as e:
            self.get_logger().error('{}\nFailed to load simulator on {}'.format(e, device))
        
        self._device = device
        self._sim = Simulator(self.net, dt=1e-4, seed=0, progress_bar=False)

    def input_func(self, t):
        data = self.read_buffer(pop=True)
        return data.flatten()
    
    @property
    def mode(self):
        if hasattr(self, '_ser'):
            return 'serial'
        else:
            return 'simulation'
        
    def read_buffer(self, pop=False):
        """Read the buffer(deque) from the right.

        Args:
            pop (bool, optional): Option to pop the buffer. Defaults to False.

        Returns:
            np.ndarray: Buffer data.
        """
        try:
            if pop:
                data = self._buffer.pop()
            else:
                data = self._buffer[-1]
        except IndexError as e:
            data = np.zeros(self._dim)

        return data


def main(args=None):
    rclpy.init(args=args)
    node = TactileInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
