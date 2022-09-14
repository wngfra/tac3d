import threading
from collections import deque
import time
import nengo
import numpy as np
import pylab as plt
import rclpy
import serial
from matplotlib.animation import FuncAnimation
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


duration = 2.5e-2


class TactileInterface(Node):
    def __init__(self):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('animated', False),
                ('baudrate', 115200),
                ('device', 'cpu'),
                ('dim', [25, 20]),
                ('port', ''),
            ],
        )

        # Get parameters
        animated = self.get_parameter('tac3d.animated').value
        baudrate = self.get_parameter('tac3d.baudrate').value
        device = str(self.get_parameter('tac3d.device').value)
        dim = self.get_parameter('tac3d.dim').value
        port = str(self.get_parameter('tac3d.port').value)
        # Setup
        self._activated = True
        self._animated = animated
        self._buffer = deque(maxlen=10)
        self._dim = dim
        self._device = device

        # Set interface operating mode
        try:
            if len(port) < 1:
                raise serial.SerialException("No serial port found.")
            # Initialize interface in serial mode
            self._baudrate = baudrate
            self._mode = 'serial'
            self._port = port
            self._ser = serial.Serial(
                port=self._port, baudrate=self._baudrate, timeout=1)
            self._update_thread = threading.Thread(target=self._serial2buffer)
            self._update_thread.start()
            self.get_logger().info('The sensor interface operating in serial mode.')
        except serial.SerialException as e:
            # Initialize interface in simulation mode
            self._mode = 'simulation'
            self._sub = self.create_subscription(
                Image, '/sensors/tactile_image', self._subscriber_callback, 10)
            self.get_logger().warn('{}'.format(e))
            self.get_logger().info('The sensor interface operating in simulation mode.')

        # Prepare SNN model
        self.create_tacnet(n_neurons=[100, 12, 36])
        self.set_device()
        # Prepare SNN output publisher
        self._timer = self.create_timer(duration, self._timer_callback)
        self._net_pub = self.create_publisher(
            Float32MultiArray, '/perception/touch', 10)

        if self._animated:
            self.animate()

    def __del__(self):
        self._activated = False

        if hasattr(self, '_anim'):
            self._anim.event_source.stop()
        if hasattr(self, '_update_thread'):
            self._update_thread.join()
        if hasattr(self, '_ser'):
            self._ser.close()
        if hasattr(self, '_sim'):
            self._sim.close()
        if hasattr(self, '_sub'):
            self._sub.destroy()

    def _animation(self, i):
        data = self.read_buffer(pop=False)
        self._artists[0].set_data(data)

        k = 1
        for (_, _), label in np.ndenumerate(data):
            self._artists[k].set_text(label)
            k += 1

        return self._artists,

    def _timer_callback(self):
        # ! FIX: Setup simulator running, semi-syncrhonous with the subscriber
        self._sim.run(duration)
        output = self._sim.data[self._probes[-1]].flatten()
        msg = Float32MultiArray(data=output)
        self._net_pub.publish(msg)

    def _subscriber_callback(self, msg):
        """Callback function for the tactile image subscriber.

        Args:
            msg (Image): Incoming message as Image.
        """
        # TODO: setup dim with the first subscribed sensory msg
        self._dim = [msg.height, msg.width]
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            values = data.view('<f4').reshape(self._dim)
            self._buffer.append(values)
        except ValueError as e:
            self.get_logger().error('Invalid data received from sensor: {}'.format(e))

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

    def animate(self, figsize=(10, 10)):
        """Animate the tactile image.

        Args:
            figsize (tuple, optional): Figure size to animate tactile image. Defaults to (10, 10).
        """

        data = self.read_buffer(pop=False)

        self._fig, ax = plt.subplots(1, 1, figsize=figsize)
        self._artists = [ax.imshow(data)]

        for (j, i), label in np.ndenumerate(data):
            text = ax.text(i, j, label, ha='center', va='center',
                           color='red', size='x-large', weight='bold')
            self._artists.append(text)

        self._anim = FuncAnimation(
            self._fig, self._animation, interval=1, repeat=False)
        ax.set_title('Sensor View')
        plt.show(block=False)

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
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        self.set_device()

    def input_func(self, t):
        data = self.read_buffer(pop=True)
        return data.flatten()

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

    def set_device(self):
        """Set device. Currently supports CPU, GPU and Intel Loihi.
        """

        if self._device == 'loihi':
            try:
                import nengo_loihi
                from nengo_loihi import Simulator as Simulator
            except ModuleNotFoundError as e:
                self.get_logger().warn('Failed to import nengo_loihi.')
                self._device = 'cpu'

        if self._device == 'cpu':
            from nengo import Simulator as Simulator

        self._sim = Simulator(self.net, dt=1e-4, seed=0, progress_bar=False)


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
