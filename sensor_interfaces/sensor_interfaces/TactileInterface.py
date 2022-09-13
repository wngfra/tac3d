import threading
from collections import deque

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

n_neurons = [100, 12, 36]
dt = 1e-2

class TactileInterface(Node):
    def __init__(self):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('animated', False),
                ('baudrate', 115200),
                ('device', 'cpu'),
                ('dim', [-1, -1]),
                ('port', ''),
            ],
        )

        # Get parameters
        animated = self.get_parameter('tac3d.animated').value
        baudrate = self.get_parameter('tac3d.baudrate').value
        device = self.get_parameter('tac3d.device').value
        dim = self.get_parameter('tac3d.dim').value
        port = self.get_parameter('tac3d.port').value
        # Set parameters
        self._activated = True
        self._buffer = deque(maxlen=10)
        self._device = device

        # Initialize interface
        try:
            if len(port) < 1:
                raise serial.SerialException("No serial port found.")
            self._port = port
            self._baudrate = baudrate
            self._dim = dim
            self.init_serial()
            self.get_logger().warn('The sensor interface operating in serial mode.')
        except serial.SerialException as e:
            self.init_simulation()
            self.get_logger().warn('{} The sensor interface operating in simulation mode.'.format(e))

        # TODO: publish TacNet output periodically
        self._net_pub = self.create_publisher(Float32MultiArray, '/perception/touch')
        self._timer = self.create_timer(dt, self._timer_callback)

        if animated:
            self.animate()

    def __del__(self):
        if hasattr(self, '_anim'):
            self._anim.event_source.stop()

        self._activated = False

        match self._mode:
            case 'serial':
                self._update_thread.join()
                self._ser.close()
            case 'simulation':
                self._sub.destroy()
            case _:
                pass

    def _animation(self, i):
        self._artists[0].set_data(self._buffer)

        k = 1
        for (_, _), label in np.ndenumerate(self._buffer):
            self._artists[k].set_text(label)
            k += 1

        return self._artists,
    
    def _timer_callback(self):
        with self._simulator(self.net) as sim:
            sim.run(dt)
        # TODO: set msg.data as the TacNet output
        output = sim.data[self._probes[-1]]
        msg = Float32MultiArray(data=output)
        self._net_pub.publish(msg)

    def _tactile_callback(self, msg):
        """Callback function for the tactile image subscriber.

        Args:
            msg (Image): Incoming message as Image.
        """
        if self._dim is None or self._dim[0] < 0:
            self._dim = (msg.height, msg.width)
            self.create_tacnet(n_neurons=n_neurons)
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            values = data.view('<f4').reshape(self._dim)
            self._buffer.append(values)
        except ValueError as e:
            self.get_logger().error('Invalid data received from sensor: {}'.format(e))

    def _update_buffer(self):
        """Update the buffer from the serial port.
        """
        while self._activated:
            if not self._ser.is_open:
                self._ser.open()

            try:
                # Convert bytes to numpy int array
                stream = self._ser.readline().decode('utf-8')
                data = [int(d) for d in stream.split(' ')[:-1]]
                values = np.asarray(data, dtype=int).reshape(self.dim)
            except ValueError as e:
                self.get_logger().error('Invalid data received from sensor: {}'.format(e))

            self._buffer.append(values)

    def animate(self, figsize=(10, 10)):
        """Animate the tactile image.

        Args:
            figsize (tuple, optional): The figure size. Defaults to (10, 10).
        """

        self._fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title('Sensor View')
        self._artists = [ax.imshow(self._buffer)]
        for (j, i), label in np.ndenumerate(self._buffer):
            text = ax.text(i, j, label, ha='center', va='center',
                           color='red', size='x-large', weight='bold')
            self._artists.append(text)

        self._anim = FuncAnimation(
            self._fig, self._animation, interval=1, repeat=False)

        plt.show(block=False)

    def create_tacnet(self, n_neurons, device='cpu'):
        """Create the SNN for asyncronous tactile perception.
        """

        self.set_device()
        self.net = nengo.Network(label='TacNet')
        with self.net:
            inp = nengo.Node(output=self.input_func)
            self._layers = [inp]
            for n in n_neurons:
                self._layers.append(nengo.Ensemble(
                    n_neurons=n, dimensions=self._n_input, neuron_type=nengo.AdaptiveLIF()))

            for i in range(len(self._layers)-1):
                nengo.Connection(
                    self._layers[i], self._layers[i+1], synapse=None)

            self._probes = []
            for layer in self._layers:
                self._probes.append(nengo.Probe(layer, synapse=1e-2))

    def init_serial(self):
        """Initialize the serial port.
        """

        self._mode = 'serial'
        self._ser = serial.Serial(
            port=self._port, baudrate=self._baudrate, timeout=1)
        self.create_tacnet(n_neurons=n_neurons)

        self._update_thread = threading.Thread(target=self._update_buffer)
        self._update_thread.start()

    def init_simulation(self):
        """Initialize the sensory signal subscriber for simulation.
        """

        self._mode = 'simulation'
        self._dim = None
        self._sub = self.create_subscription(
            Image, '/sensors/tactile_image', self._tactile_callback, 10)

    def input_func(self, t):
        try:
            data = self._buffer.pop()
        except IndexError as e:
            data = np.zeros(self._dim)
        return data.flatten()

    def set_device(self):
        """Set device. Currently supports CPU emulation and Intel Loihi.
        """

        if self._device == 'loihi':
            try:
                import nengo_loihi
                nengo_loihi.set_defaults()
                from nengo_loihi import Simulator
            except ModuleNotFoundError as e:
                print('Failed to import nengo_loihi, falls back to cpu emulation.')
                self._device = 'cpu'
        if self._device == 'cpu':
            from nengo import Simulator

        self._simulator = Simulator


def main(args=None):
    rclpy.init(args=args)
    node = TactileInterface()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
