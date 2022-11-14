# Copyright (C) 2022 wngfra
#
# tac3d is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tac3d is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tac3d. If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from brian2 import *
from brian2tools import brian_plot
from configs import *


def generate_conns(N, M, mode):
    match mode:
        case 'random':
            rng = np.random.default_rng(np.random.PCG64DXSM())
            i = np.arange(N)
            j = np.repeat(np.arange(M), N // M)
            rng.shuffle(j)
            return i, j
        case _:
            return None, None


class TacNet(object):
    def __init__(self, num_neurons: list) -> None:
        """Constructor of the Tactile Encoding Network.

        Args:
            num_neurons (list): _description_
        """
        self._eqs = equations
        self._events = events
        self._params = params
        self._params['num_neurons'] = num_neurons

        # Define NeuronGroups
        self.layers = {}
        for i, n_neuron in enumerate(num_neurons):
            layer_name = 'L'+str(i+1)
            event_name = 'event'+str(i+1)
            if event_name in self._events:
                event_label = self._events[event_name][0]
                event_trigger = self._events[event_name][1]
                event_operation = self._events[event_name][2]
                event = {event_label: event_trigger}
            else:
                event = None
            self.layers[layer_name] = NeuronGroup(n_neuron, model=self._eqs['L'+str(i+1)],
                                                  method='euler',
                                                  threshold='v > V_theta',
                                                  reset='v = V_res',
                                                  refractory='tau_r',
                                                  events=event,
                                                  namespace=self._params,
                                                  name=layer_name)
            if event is not None:
                self.layers[layer_name].run_on_event(
                    event_label, event_operation)

        # Define Synapses
        self.synapses = {}
        for i in range(len(num_neurons)):
            for j in range(len(num_neurons)):
                syn_name = 'Syn'+str(i+1)+str(j+1)
                pre_name = 'Pre'+str(i+1)+str(j+1)
                post_name = 'Post'+str(i+1)+str(j+1)
                if syn_name in self._eqs:
                    if pre_name in self._eqs:
                        on_pre = self._eqs[pre_name]
                    else:
                        on_pre = None
                    if post_name in self._eqs:
                        on_post = self._eqs[post_name]
                    else:
                        on_post = None
                    self.synapses[syn_name] = Synapses(self.layers['L'+str(i+1)],
                                                       self.layers['L' +
                                                                   str(j+1)],
                                                       model=self._eqs[syn_name],
                                                       on_pre=on_pre,
                                                       on_post=on_post,
                                                       namespace=self._params,
                                                       method='euler',
                                                       name=syn_name)

        # Connect synapses
        for synapse in self.synapses.values():
            if 'mode' in connections[synapse.name]:
                mode = connections[synapse.name]['mode']
            else:
                mode = 'full'
            if 'p' in connections[synapse.name]:
                p = connections[synapse.name]['p']
            else:
                p = 1
            if 'condition' in connections[synapse.name]:
                condition = connections[synapse.name]['condition']
            else:
                condition = None
            i, j = generate_conns(synapse.source.N,
                                  synapse.target.N, mode=mode)
            synapse.connect(i=i, j=j, condition=condition, p=p)

        # Define Monitors
        self.mons = dict()
        for layer in self.layers.values():
            mon_name = 'SpikeMonitor_'+layer.name
            self.mons[mon_name] = SpikeMonitor(
                layer, record=True, name=mon_name)
        for k, v in monitors.items():
            mon_name = 'StateMonitor_'+k
            if k in self.layers:
                G = self.layers[k]
                record = True
            if k in self.synapses:
                G = self.synapses[k]
                record = np.arange(G.source.N*G.target.N)
            self.mons[mon_name] = StateMonitor(
                G, v, record=record, name=mon_name)

        # Add all groups to the network
        self.net = Network()
        for layer in self.layers.values():
            self.net.add(layer)
        for synapse in self.synapses.values():
            self.net.add(synapse)
        for monitor in self.mons.values():
            self.net.add(monitor)

    def run(self, net_input, duration, save_state=None):
        """_summary_

        Args:
            net_input (_type_): _description_
            duration (_type_): _description_
            save_state (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._params['I'] = net_input
        self.net.run(duration=duration, report='stdout')
        return self.mons
