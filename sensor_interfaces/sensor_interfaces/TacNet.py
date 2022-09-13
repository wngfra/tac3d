import nengo
import numpy as np
import pylab as plt


class TacNet:
    def __init__(self, ):
        self._device = device
        self.set_device()

        self._n_input = np.prod(n_input)
        self._n_neurons = n_neurons


    def set_device(self):
        """ Set device. Currently supports CPU emulation and Intel Loihi.
        """
        if self._device == 'loihi':
            try:
                import nengo_loihi
                nengo_loihi.set_defaults()
                from nengo_loihi import Simulator
            except ModuleNotFoundError as e:
                print('Failed to import nengo_loihi, falls back to emulation.')
                self._device = 'sim'
        if self._device == 'sim':
            from nengo import Simulator

        self._simulator = Simulator

    def run(self, ):
        with self._simulator(self.net) as sim:
            sim.run(duration)

        return sim.trange(), [sim.data[probe] for probe in self._probes]


if __name__ == '__main__':
    net = TacNet([20, 30], [10, 5])
    trange, probes = net.run(10.0)
    plt.plot(trange, probes[2])
    plt.show()
