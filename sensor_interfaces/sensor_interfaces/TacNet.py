import nengo
import numpy as np
import pylab as plt


class TacNet:
    def __init__(self, ):
        self._device = device
        self.set_device()

        self._n_input = np.prod(n_input)
        self._n_neurons = n_neurons

    def run(self, ):
        with self._simulator(self.net) as sim:
            sim.run(duration)

        return sim.trange(), [sim.data[probe] for probe in self._probes]


if __name__ == '__main__':
    net = TacNet([20, 30], [10, 5])
    trange, probes = net.run(10.0)
    plt.plot(trange, probes[2])
    plt.show()
