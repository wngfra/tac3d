import numpy as np
import pylab as plt
import serial
import threading
from matplotlib import artist, style
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node



class SensorInterface(Node):
    def __init__(self, shape, port=None, baudrate=115200, animated=False):
        super().__init__('sensor_interface_node')

        self.declare_parameters(
            namespace='tac3d',
            parameters=[
                ('mode', '')
            ],
        )

        try:
            if port is None:
                raise serial.SerialException("No serial port specified.")
            self.__ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            self.__mode = 'serial'
        except serial.SerialException as e:
            self.get_logger().info('{}\nThe sensor interface operating in simulation mode.'.format(e))

            self.__mode = 'simulation'
            self.init_simulation()

        self.__shape = shape
        self.__values = -1*np.ones(shape, dtype=int)

        self.__activated = True
        self.__update_thread = threading.Thread(
            target=self.__update_values, args=(), daemon=True)
        self.__update_thread.start()

        if animated:
            self.animate()

    @property
    def mode(self):
        return self.__mode

    @property
    def values(self):
        return self.__values

    def __del__(self):
        if hasattr(self, '__anim'):
            self.__anim.event_source.stop()

        self.__activated = False
        self.__update_thread.join()

        if self.__mode == 'serial':
            self.__ser.close()

    def __update_values(self):
        while self.__activated:
            match self.__mode:
                case 'serial':
                    if not self.__ser.is_open:
                        self.__ser.open()

                    # Convert bytes to numpy int array
                    stream = self.__ser.readline().decode('utf-8')
                    data = [int(d) for d in stream.split(' ')[:-1]]
                    values = np.asarray(data, dtype=int)
                    try:
                        self.__values = values.reshape(self.__shape)
                    except ValueError as e:
                        print(e)
                case 'simulation':
                    values = np.random.randint(0, 1024, self.__shape)
                case _:
                    print('Sensor interface mode is not specified.')
                    values = -1*np.ones(self.__shape)

    def __animation(self, i):
        self.__artists[0].set_data(self.values)

        k = 1
        for (_, _), label in np.ndenumerate(self.values):
            self.__artists[k].set_text(label)
            k += 1

        return self.__artists,

    def animate(self, figsize=(10, 10)):
        self.__fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title('Sensor View')
        self.__artists = [ax.imshow(self.values)]
        for (j, i), label in np.ndenumerate(self.values):
            text = ax.text(i, j, label, ha='center', va='center',
                           color='red', size='x-large', weight='bold')
            self.__artists.append(text)

        self.__anim = FuncAnimation(
            self.__fig, self.__animation, interval=1, repeat=False)

        plt.show(block=False)

    def init_simulation(self):
        pass


def print_values(sensor, duration, delay=0.1):
    import time
    t0 = time.time()
    while time.time() - t0 < duration:
        print(sensor.values)
        time.sleep(delay)


if __name__ == '__main__':
    import sys
    import glob

    if len(sys.argv) > 2:
        ports = glob.glob('/dev/ttyACM[0-9]*')
        p = None

        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                p = port
                break
            except:
                pass
    else:
        p = None

    sensor = SensorInterface((5, 5), p)
    print(sensor.mode)

    print_values(sensor, 60.0)
