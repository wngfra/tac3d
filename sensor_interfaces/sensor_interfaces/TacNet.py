import nengo
import numpy as np
import pylab as plt


class TacNet:
    def __init__(self, n_input, n_hidden, n_output, device='sim'):
        self._device = device
        self.set_device()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self._net = nengo.Network(label='TacNet')

        with self._net:
            inp = nengo.Node(output=np.zeros((2)))
            in_ens = nengo.Ensemble(n_neurons=n_input, dimensions=2)
            hid_ens = nengo.Ensemble(n_neurons=n_hidden, dimensions=2, neuron_type=nengo.AdaptiveLIF())
            out_ens = nengo.Ensemble(n_neurons=n_output, dimensions=2, neuron_type=nengo.AdaptiveLIF())

            nengo.Connection(inp, in_ens)
            nengo.Connection(in_ens, hid_ens)
            nengo.Connection(hid_ens, out_ens)

            in_probe = nengo.Probe(in_ens, synapse=1e-2, label='in_probe')
            hid_probe = nengo.Probe(hid_ens, synapse=1e-2, label='hid_probe')
            out_probe = nengo.Probe(out_ens, synapse=1e-2, label='out_probe')
            
            self._probes = [in_probe, hid_probe, out_probe]

    def set_device(self):
        """ Set up simulation device. Currently supports CPU emulation and Intel Loihi.
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

    def run(self, duration):
        with self._simulator(self._net) as sim:
            sim.run(duration)

        return sim.trange(), [sim.data[probe] for probe in self._probes]

if __name__=='__main__':
    net = TacNet(10, 5, 2)
    trange, probes = net.run(10.0)
    plt.plot(trange, probes[2])
    plt.show()