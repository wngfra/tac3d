import numpy as np
import pylab as plt
import serial
import threading
from matplotlib.animation import FuncAnimation


class Sensor:
    def __init__(self, shape, port='/dev/ttyACM0', baudrate=115200):
        self.__ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.__shape = shape
        self.__values = np.random.random(shape)

        self.__activated = True
        self.__update_thread = threading.Thread(target=self.__update_values, args=(), daemon=True)
        self.__update_thread.start()
        self.animate()

    def __del__(self):
        self.__activated = False
        self.__update_thread.join()
        self.__ser.close()

    def __update_values(self):
        while self.__activated:
            if not self.__ser.is_open:
                self.__ser.open()

            # Convert bytes to numpy int array
            stream = self.__ser.readline().decode('utf-8')
            data = [int(d) for d in stream.split(' ')[:-1]]
            values = np.asarray(data, dtype=int)

            try:
                values = values.reshape(self.__shape)
                self.__values = values
            except ValueError as e:
                print(e)

    def __animation(self, i):
        self.__im.set_data(self.values)
        return self.__im

    def animate(self, figsize=(6, 6)):
        self.__fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.__im = ax.imshow(self.values)
        ax.set_title('Sensor View')

        self.__anim = FuncAnimation(self.__fig, self.__animation, interval=10)
        
        plt.show(block=False)

    @property
    def values(self):
        return self.__values


if __name__ == '__main__':
    sensor = Sensor((5, 5), '/dev/ttyACM0')