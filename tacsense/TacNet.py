# Copyright 2022 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from brian2 import *
from configs import connections, equations, events, initial_values, monitors, params


# autopep8: off
events = {"event3": ["increase_threshold", "v > v_th", "v_th += delta_theta*rand()"]}

params = {
    # Model constants
    "C_mem": 200 * pF,  # Membrane capacitance
    "delta_theta": 5 * mV,  # Adaptive threshold incremental scale
    "g_l": 7 * nS,  # Leak conductance
    "J_C": 1,  # Scale of the calcium variable
    "tau_c": 60 * ms,  # Calcium variable time constant
    "tau_e": 10 * ms,  # Excitatory synaptic time constant
    "tau_i": 5 * ms,  # Inhibitory synaptic time constant
    "tau_r": 5 * ms,  # Refractory period
    "tau_theta": 5 * ms,  # Adaptive threshold time constant
    "V_ir": -80 * mV,  # Inhibitory reverse potential
    "V_res": -60 * mV,  # Resting potential
    "V_theta": -50 * mV,  # Spiking threshold
    "w_e": 35 * nS,  # Excitatory conductance increment
    "w_i": 30 * nS,  # Inhibitory conductance increment
    "X_max": 1,  # Synaptic variable maximum
    "X_min": 0,  # Synaptic variable minimum
    # Simulation parameters
    "defaultclock.dt": 0.1 * ms,  # Time step
}

# Thresholds and plasticity parameters
params["a"] = 0.1 * params["X_max"]
params["b"] = 0.1 * params["X_max"]
params["alpha"] = 3.5 * params["X_max"] * Hz
params["beta"] = 3.5 * params["X_max"] * Hz
params["theta_hup"] = 12 * params["J_C"]
params["theta_lup"] = 3 * params["J_C"]
params["theta_hdown"] = 4 * params["J_C"]
params["theta_ldown"] = 3 * params["J_C"]
params["theta_v"] = 0.8 * params["V_theta"]
params["theta_X"] = 0.5 * params["X_max"]

equations = {
    # Neuronal models
    "L1": """
        dv/dt = (g_l*(V_res - v) + I(t,i))/C_mem : volt (unless refractory)
        x : 1
        y : 1
        """,
    "L2": """
        dv/dt = (g_l*(V_res - v) - g_e*v)/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        sum_w : 1
        x : 1
        y : 1
        """,
    "L3": """
        dv/dt = (g_l*(V_res - v) - g_e*v + g_i*(V_ir - v))/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        dg_i/dt = -g_i/tau_i : siemens
        dv_th/dt = (V_theta - v_th)/tau_theta : volt
        is_winner : boolean
        sum_w : 1
        """,
    # Synaptic models
    "Syn12": """
        w : 1
        sum_w_post = w : 1 (summed)
        """,
    "Syn23": """
        count : 1
        X_condition : 1
        dc/dt = -c/tau_c + J_C*count*Hz: 1 (clock-driven)
        dX/dt = (alpha*int(X > theta_X)*int(X < X_max) - beta*int(X <= theta_X)*int(X > X_min))*(1 - X_condition) : 1 (clock-driven)
        w = int(X >= 0.5) : 1
        sum_w_post = w : 1 (summed)
        """,
    "Syn33": """
        w : 1
        """,
    # Synaptic events
    "Pre12": """
        g_e_post += w_e/sum_w_post
        """,
    "Pre23": """
        g_e_post += w_e*int(sum_w_post >= 1)/(sum_w_post + 1e-12)*X
        X += a*int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) - b*int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        X = clip(X, X_min, X_max)
        X_condition = int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) + int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        """,
    "Pre33": """
        g_i_post += w_i*(i-j)
        """,
    "Post23": """
        count += 1
        X_condition = 0
        """,
}

connections = {
    "Syn12": {"mode": "gaussian"},
    "Syn23": {"mode": "full"},
    "Syn33": {"mode": "different"},
}

monitors = {
    "L1": ["v"],
    "L2": ["v", "g_e"],
    "L3": ["v", "v_th", "g_e", "g_i", "sum_w"],
    "Syn23": ["X", "w", "c"],
}

initial_values = {
    "L1": {"v": "V_res + rand()*(V_theta - V_res)"},
    "L2": {"v": "V_res + rand()*(V_theta - V_res)"},
    "L3": {
        "v": "V_res + rand()*(V_theta - V_res)",
        "v_th": "V_theta",
        "g_e": 0 * nS,
        "g_i": 0 * nS,
        "is_winner": False,
    },
    "Syn12": {"w": 1},
    "Syn23": {"count": 0, "c": 2, "X": "rand()*X_max", "delay": "rand()*tau_r"},
    "Syn33": {"w": 1},
}
# autopep8: on


def generate_conns(N, M, mode="full"):
    """Generate connection patterns for the Synapses.

    Parameters
    ----------
        N : int
            Number of incoming synapses.
        M : int
            Number of outgoing synapses.
        mode : {'different', 'full', 'random'}, default='full'
            Connection mode.

    Returns
    -------
        i, j, condition, p: (int, int, str, float)
            Synaptic connection configs.

    """
    i, j = None, None
    condition = None
    p = 1.0

    match mode:
        case "different":
            condition = "i != j"
        case "full":
            i = np.repeat(np.arange(N), M)
            j = np.tile(np.arange(M), N)
        case "gaussian":
            p = "1.0 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2)/(2*rf_size**2))"
        case "random":
            p = "rand()"

    return i, j, condition, p


class TacNet(object):
    def __init__(self, num_neurons: list) -> None:
        """Constructor of the Tactile Encoding Network.

        Parameters
        ----------
            num_neurons : list of int
                List of numbers of neurons of each layer. 'L1' is the input layer.
        """
        try:
            device.reinit()
            device.activate()
        except _:
            pass
        # autopep8: off
        # Define NeuronGroups (layers)
        neuron_groups = {}
        for index, n_neuron in enumerate(num_neurons):
            layer_name = "L%d" % (index + 1)
            event_name = "event%d" % (index + 1)

            if event_name in events:
                event_label = events[event_name][0]
                event_trigger = events[event_name][1]
                event_operation = events[event_name][2]
                event = {event_label: event_trigger}
            else:
                event = None
            neuron_groups[layer_name] = NeuronGroup(
                n_neuron,
                model=equations[layer_name],
                method="euler",
                threshold="v > V_theta",
                reset="v = V_res",
                refractory="tau_r",
                events=event,
                namespace=params,
                name=layer_name,
            )
            if event is not None:
                neuron_groups[layer_name].run_on_event(event_label, event_operation)

        # Assign coordinates to L1 neurons
        l1_size = sqrt(len(neuron_groups["L1"]))
        l2_size = sqrt(len(neuron_groups["L2"]))
        rf_size = l1_size / l2_size

        neuron_groups["L1"].x = "i // l1_size"
        neuron_groups["L1"].y = "i % l1_size"
        neuron_groups["L2"].x = "rf_size // 2 + rf_size * (i // l2_size)"
        neuron_groups["L2"].y = "rf_size // 2 + rf_size * (i % l2_size)"

        # Define Synapses
        synapses = {}
        for source in range(len(num_neurons)):
            for target in range(len(num_neurons)):
                link = "%d%d" % (source + 1, target + 1)
                syn_name = "Syn" + link
                pre_name = "Pre" + link
                post_name = "Post" + link
                if syn_name in equations:
                    if pre_name in equations:
                        on_pre = equations[pre_name]
                    else:
                        on_pre = None
                    if post_name in equations:
                        on_post = equations[post_name]
                    else:
                        on_post = None
                    synapses[syn_name] = Synapses(
                        neuron_groups["L%d" % (source + 1)],
                        neuron_groups["L%d" % (target + 1)],
                        model=equations[syn_name],
                        on_pre=on_pre,
                        on_post=on_post,
                        namespace=params,
                        method="euler",
                        name=syn_name,
                    )
        # autopep8: on

        # Connect synapses
        for synapse in synapses.values():
            mode = connections[synapse.name]["mode"]
            source, target, condition, p = generate_conns(
                synapse.source.N, synapse.target.N, mode=mode
            )
            synapse.connect(i=source, j=target, condition=condition, p=p)

        # Define Monitors
        self.mons = dict()
        for layer in neuron_groups.values():
            mon_name = "SpikeMonitor_" + layer.name
            self.mons[mon_name] = SpikeMonitor(layer, record=True, name=mon_name)
        for k, v in monitors.items():
            mon_name = "StateMonitor_" + k
            if k in neuron_groups:
                G = neuron_groups[k]
                record = True
            if k in synapses:
                G = synapses[k]
                record = np.arange(G.source.N * G.target.N)
            self.mons[mon_name] = StateMonitor(G, v, record=record, name=mon_name)

        # Add all groups to the network
        self.net = Network()
        for layer in neuron_groups.values():
            self.net.add(layer)
        for synapse in synapses.values():
            self.net.add(synapse)
        for monitor in self.mons.values():
            self.net.add(monitor)

        self.initiate(initial_values)

    def initiate(self, initial_values: dict):
        """Set initial values for the simulator.

        Parameters
            initial_values (dict): _description_
        """
        for k, iv_dict in initial_values.items():
            for attr, iv in iv_dict.items():
                setattr(self.net[k], attr, iv)

    def run(self, net_input, duration, save_state=None):
        """Run the simulation for a duration and save the state(optional).

        Parameters
            net_input (TimedArray): Spike inputs.
            duration (Quantity): Time to run the simulation.
            save_state (str): Filepath to save the state. Defaults to None.

        Returns
            dict: A dictionary of Monitors.
        """
        params["I"] = net_input
        self.net.run(duration=duration)
        return self.mons
