import numpy as np
import pylab as plt
import serial
import threading
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from sensor_msgs.msg import Image


class SensorInterface(LifecycleNode):
    def __init__(self):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('animated', False),
                ('baudrate', 115200),
                ('dim', None),
                ('port', None),
            ],
        )

        # Get parameters
        animated = self.get_parameter('animated').value
        baudrate = self.get_parameter('baudrate').value
        dim = self.get_parameter('dim').value
        port = self.get_parameter('port').value

        try:
            if port is None:
                raise serial.SerialException("No serial port found.")
            self._port = port
            self._baudrate = baudrate
            self._dim = dim
            self.init_serial()
            self.get_logger().info('The sensor interface operating in serial mode.')
            
        except serial.SerialException as e:
            self.init_simulation()
            self.get_logger().info('{}\nThe sensor interface operating in simulation mode.'.format(e))
                
        self._activated = True

        if animated:
            self.animate()

    @property
    def dim(self):
        return self._dim

    @property
    def mode(self):
        return self._mode

    @property
    def values(self):
        return self._values

    def __del__(self):
        if hasattr(self, '_anim'):
            self._anim.event_source.stop()

        self._activated = False

        match self.mode:
            case 'serial':
                self._update_thread.join()
                self._ser.close()
            case 'simulation':
                pass
            case _:
                pass

    def _update_values(self):
        while self._activated:
            if not self._ser.is_open:
                self._ser.open()
            
            # Convert bytes to numpy int array
            stream = self._ser.readline().decode('utf-8')
            data = [int(d) for d in stream.split(' ')[:-1]]
            values = np.asarray(data, dtype=int)

            try:
                self._values = values.reshape(self.dim)
            except ValueError as e:
                print(e)

    def _animation(self, i):
        self._artists[0].set_data(self.values)

        k = 1
        for (_, _), label in np.ndenumerate(self.values):
            self._artists[k].set_text(label)
            k += 1

        return self._artists,

    def animate(self, figsize=(10, 10)):
        self._fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title('Sensor View')
        self._artists = [ax.imshow(self.values)]
        for (j, i), label in np.ndenumerate(self.values):
            text = ax.text(i, j, label, ha='center', va='center',
                           color='red', size='x-large', weight='bold')
            self._artists.append(text)

        self._anim = FuncAnimation(
            self._fig, self._animation, interval=1, repeat=False)

        plt.show(block=False)

    def init_serial(self):
        self._mode = 'serial'
        self._values = -1*np.ones(self.dim, dtype=int)
        self._ser = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=1)

    def init_simulation(self):
        self._mode = 'simulation'
        self._sub = self.create_subscription(Image, '/sensors/tactile_image', self._tactile_callback, 20)

    def _tactile_callback(self, msg):
        self._values = np.asarray(msg.data, dtype=int).reshape(self._dim)


def main(args=None):
    rclpy.init(args=args)
    node = SensorInterface()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

if __name__ == '__main__':
    main()