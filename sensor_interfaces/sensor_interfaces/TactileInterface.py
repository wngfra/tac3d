from pickletools import uint8
import numpy as np
import pylab as plt
import serial
import threading
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image


class TactileInterface(Node):
    def __init__(self):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('animated', False),
                ('baudrate', 115200),
                ('dim', [-1, -1]),
                ('port', ''),
            ],
        )

        # Get parameters
        animated = self.get_parameter('tac3d.animated').value
        baudrate = self.get_parameter('tac3d.baudrate').value
        dim = self.get_parameter('tac3d.dim').value
        port = self.get_parameter('tac3d.port').value

        try:
            if len(port) < 1:
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

    def __del__(self):
        if hasattr(self, '_anim'):
            self._anim.event_source.stop()

        self._activated = False

        match self._mode:
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
        self._artists[0].set_data(self._values)

        k = 1
        for (_, _), label in np.ndenumerate(self._values):
            self._artists[k].set_text(label)
            k += 1

        return self._artists,

    def animate(self, figsize=(10, 10)):
        self._fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title('Sensor View')
        self._artists = [ax.imshow(self._values)]
        for (j, i), label in np.ndenumerate(self._values):
            text = ax.text(i, j, label, ha='center', va='center',
                           color='red', size='x-large', weight='bold')
            self._artists.append(text)

        self._anim = FuncAnimation(
            self._fig, self._animation, interval=1, repeat=False)

        plt.show(block=False)

    def init_serial(self):
        self._mode = 'serial'
        self._values = -1*np.ones(self._dim, dtype=int)
        self._ser = serial.Serial(port=self._port, baudrate=self._baudrate, timeout=1)

    def init_simulation(self):
        self._mode = 'simulation'
        self._dim = None
        self._sub = self.create_subscription(Image, '/sensors/tactile_image', self._tactile_callback, 10)

    def _tactile_callback(self, msg):
        if hasattr(self, '_dim'):
            self._dim = (msg.height, msg.width)
        
        data = np.frombuffer(msg.data, dtype=np.uint8)
        self._values = data.view('<f4').reshape(self._dim)

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