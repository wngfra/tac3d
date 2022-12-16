# Copyright 2022 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from brian2 import *
from configs import connections, equations, events, initial_values, monitors, params


def generate_conns(N, M, mode):
    """Generate connection patterns for the Synapses.

    Args:
        N (int): Number of incoming synapses.
        M (int): Number of outgoing synapses.
        mode (str): Connection mode.

    Returns:
        tuple: The source and target index tuple (i, j). 
    """
    match mode:
        case 'random':
            rng = np.random.default_rng(np.random.PCG64DXSM())
            i = np.arange(N)
            j = np.repeat(np.arange(M), N // M)
            rng.shuffle(j)
            return i, j
        case _:
            return (None, None)


class TacNet(object):
    def __init__(self, num_neurons: list) -> None:
        """Constructor of the Tactile Encoding Network.

        Args:
            num_neurons (list): List of numbers of neurons of each layer. 'L1' is the input layer.
        """
        # autopep8: off
        # Define NeuronGroups (layers)
        self.layers = {}
        for i, n_neuron in enumerate(num_neurons):
            layer_name = 'L'+str(i+1)
            event_name = 'event'+str(i+1)
            # TODO: add support to multiple events
            if event_name in events:
                event_label = events[event_name][0]
                event_trigger = events[event_name][1]
                event_operation = events[event_name][2]
                event = {event_label: event_trigger}
            else:
                event = None
            self.layers[layer_name] = NeuronGroup(n_neuron, model=equations[layer_name],
                                                  method='euler',
                                                  threshold='v > V_theta',
                                                  reset='v = V_res',
                                                  refractory='tau_r',
                                                  events=event,
                                                  namespace=params,
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
                if syn_name in equations:
                    if pre_name in equations:
                        on_pre = equations[pre_name]
                    else:
                        on_pre = None
                    if post_name in equations:
                        on_post = equations[post_name]
                    else:
                        on_post = None
                    self.synapses[syn_name] = Synapses(self.layers['L'+str(i+1)],
                                                       self.layers['L'+str(j+1)],
                                                       model=equations[syn_name],
                                                       on_pre=on_pre,
                                                       on_post=on_post,
                                                       namespace=params,
                                                       method='euler',
                                                       name=syn_name)
        # autopep8: on

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

        self.initiate(initial_values)

    def initiate(self, initial_values: dict):
        """Set initial values for the simulator.

        Args:
            initial_values (dict): _description_
        """
        for layer in self.layers.values():
            ivs = initial_values[0][layer.name]
            for k, v in ivs.items():
                setattr(layer, k, v)
        for synapse in self.synapses.values():
            ivs = initial_values[1][synapse.name]
            for k, v in ivs.items():
                setattr(synapse, k, v)

    def run(self, net_input, duration, clean=False, save_state=None):
        """Run the simulation for a duration and save the state(optional).

        Args:
            net_input (TimedArray): Spike inputs.
            duration (Quantity): Time to run the simulation.
            save_state (str): Filepath to save the state. Defaults to None.

        Returns:
            dict: A dictionary of Monitors. 
        """
        params['I'] = net_input
        self.net.run(duration=duration, report='stdout')
        return self.mons
