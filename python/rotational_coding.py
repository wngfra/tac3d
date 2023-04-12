# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import nengo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from TouchDataset import TouchDataset
from custom_learning_rules import SynapticSampling

font = {"weight": "normal", "size": 30}

matplotlib.rc("font", **font)


def gen_transform(pattern="random", weights=None):
    W: Optional[np.ndarray] = None

    def inner(shape, weights=weights):
        """_summary_

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            _type_: _description_
        """
        match pattern:
            case "identity_excitation":
                W = 1
            case "identity_inhibition":
                W = -1
            case "exclusive_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = -np.ones((shape[0], shape[0])) + 2 * np.eye(shape[0])
                W[W < 0] *= 3
            case "custom":
                if weights is None:
                    raise ValueError("No weights provided!")
                W = weights
            case "zero":
                W = 0
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Uniform(-1, 1),
                )
        return W

    return inner


class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]


# Function to inhibit the error population after 15s
def inhib(t):
    return 2 if t > duration * 0.6 else 0.0


_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1, 1))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
height, width = X_train[0].shape
image_size = X_train[0].size

# Simulation parameters
dt = 1e-3
max_rate = 150 # Hz
amp = 1.0
rate_target = max_rate * amp  # must be in amplitude scaled units

n_features = 10
n_hidden_neurons = 100
wta_neurons = 64
n_latent_variables = 16
presentation_time = 0.1
duration = 10

learning_rate = 2e-8
delay = Delay(1, timesteps=int(0.1 / dt))

layer_confs = [
    dict(
        name="state_node",
        neuron=None,
        output=lambda t: y_train[int(t / presentation_time)],
    ),
    dict(
        name="state",
        n_neurons=wta_neurons,
        dimensions=1,
    ),
    dict(
        name="delay_node",
        neuron=None,
        output=delay.step,
        size_in=1,
    ),
    dict(
        name="delayed_state",
        n_neurons=10,
        dimensions=1,
    ),
    dict(
        name="delta_state",
        n_neurons=10,
        dimensions=1,
    ),
    dict(
        name="stimulus",
        neuron=None,
        output=lambda t: X_train[int(t / presentation_time)].ravel(),
    ),
    dict(
        name="stim",
        neuron=nengo.PoissonSpiking(nengo.LIFRate(amplitude=amp)),
        n_neurons=image_size,
        dimensions=1,
        on_chip=False,
    ),
    dict(
        name="hidden",
        n_neurons=n_hidden_neurons,
        dimensions=1,
    ),
    dict(
        name="output",
        n_neurons=image_size,
        dimensions=1,
    ),
    dict(
        name="WTA",
        n_neurons=wta_neurons,
        dimensions=n_latent_variables,
    ),
    dict(
        name="reconstruction_error",
        n_neurons=image_size,
        dimensions=1,
    ),
]

conn_confs = [
    # state variable: angle
    dict(
        pre="state_node",
        post="state",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_node",
        post="delay_node",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delay_node",
        post="delayed_state",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state",
        post="delta_state",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delayed_state",
        post="delta_state",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="stimulus",
        post="stim_neurons",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="stim_neurons",
        post="hidden_neurons",
        synapse=0.01,
    ),
    dict(
        pre="hidden_neurons",
        post="output",
        transform=gen_transform(),
        synapse=0.01,
    ),
    dict(
        pre="hidden_neurons",
        post="WTA_neurons",
        learning_rule=nengo.BCM(learning_rate=learning_rate),
    ),
    dict(
        pre="WTA_neurons",
        post="WTA_neurons",
        transform=gen_transform("exclusive_inhibition"),
        synapse=0.01,
    ),
    dict(
        pre="stim_neurons",
        post="reconstruction_error_neurons",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="output_neurons",
        post="reconstruction_error_neurons",
    ),
]

learning_confs = []


default_neuron = nengo.AdaptiveLIF(amplitude=amp)
default_rates = nengo.dists.Choice([max_rate])
default_intercepts = nengo.dists.Choice([0])

# Create the Nengo model
with nengo.Network(label="tacnet") as model:
    layers = dict()
    connections = dict()
    probes = dict()

    # Create layers
    for k, layer_conf in enumerate(layer_confs):
        layer_conf = dict(layer_conf)  # Copy layer configuration
        name = layer_conf.pop("name")
        n_neurons = layer_conf.pop("n_neurons", 1)
        dimensions = layer_conf.pop("dimensions", 1)
        encoders = layer_conf.pop(
            "encoders", nengo.dists.ScatteredHypersphere(surface=True)
        )
        max_rates = layer_conf.pop("max_rates", default_rates)
        radius = layer_conf.pop("radius", 1.0)
        neuron_type = layer_conf.pop("neuron", default_neuron)
        on_chip = layer_conf.pop("on_chip", False)
        block = layer_conf.pop("block", None)
        output = layer_conf.pop("output", None)
        size_in = layer_conf.pop("size_in", None)

        assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"

            layer = nengo.Node(output=output, size_in=size_in, label=name)
            layers[name] = layer
            probe = nengo.Probe(layer, synapse=0.01, label="%s_probe" % name)
            probes[name] = probe
        else:
            layer = nengo.Ensemble(
                n_neurons,
                dimensions=dimensions,
                radius=radius,
                encoders=encoders,
                intercepts=default_intercepts,
                neuron_type=neuron_type,
                max_rates=max_rates,
                normalize_encoders=True,
                label=name,
            )
            layers[name] = layer
            layers[name + "_neurons"] = layer.neurons

            # Add a probe so we can measure individual layer rates
            probe = nengo.Probe(layer, synapse=0.01, label="%s_probe" % name)
            probes[name] = probe
            probe = nengo.Probe(
                layer.neurons, synapse=0.01, label="%s_neurons_probe" % name
            )
            probes[name + "_neurons"] = probe

    for k, conn_conf in enumerate(conn_confs):
        conn_conf = dict(conn_conf)  # Copy connection configuration
        pre = conn_conf.pop("pre")
        post = conn_conf.pop("post")
        synapse = conn_conf.pop("synapse", None)
        solver = conn_conf.pop("solver", None)
        transform = conn_conf.pop("transform", gen_transform())
        learning_rule = conn_conf.pop("learning_rule", None)
        name = "{}2{}".format(pre, post)
        function = conn_conf.pop("function", None)

        assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )
        conn = nengo.Connection(
            layers[pre],
            layers[post],
            function=function,
            transform=transform((layers[post].size_in, layers[pre].size_in)),
            synapse=synapse,
            label=name,
        )
        if solver:
            conn.solver = solver
        if learning_rule:
            conn.learning_rule_type = learning_rule
        connections[name] = conn

        probe = nengo.Probe(
            conn, "weights", synapse=0.01, label="weights_{}".format(name)
        )
        probes[name] = probe

    # Connect learning rule
    for k, learning_conf in enumerate(learning_confs):
        learning_conf = dict(learning_conf)
        pre = learning_conf.pop("pre")
        post = learning_conf.pop("post")
        transform = learning_conf.pop("transform", 1)
        nengo.Connection(
            layers[pre],
            connections[post].learning_rule,
            transform=transform,
        )

"""Run in non-GUI mode
"""
with nengo.Simulator(model, dt=dt) as sim:
    sim.run(duration)

conn_name = "{}2{}".format("hidden_neurons", "WTA_neurons")
ens_names = ["state", "WTA", "hidden"]

plt.figure(figsize=(5, 10))
# Find weight row with max variance
neuron = np.argmax(np.mean(np.var(sim.data[probes[conn_name]], axis=0), axis=1))
plt.plot(sim.trange(), sim.data[probes[conn_name]][:, neuron, :])
plt.xlabel("time (s)")
plt.ylabel("weights")

fig, axs = plt.subplots(len(ens_names), 1, figsize=(5*len(ens_names), 10))
for i, ens_name in enumerate(ens_names):
    axs[i].plot(sim.trange(), sim.data[probes[ens_name]])
    axs[i].set_ylabel(ens_name)
    axs[i].set_xlabel("time (s)")
    axs[i].grid()
plt.tight_layout()
plt.show()
