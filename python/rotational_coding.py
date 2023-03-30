# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import nengo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from custom_learning_rules import MirroredSTDP
from TouchDataset import TouchDataset

import nengo_dl

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
            case "cyclic_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                xmax = shape[1] // 2
                x = np.abs(np.arange(shape[0]) - xmax)
                W = np.empty((shape[0], shape[0]))
                for i in range(shape[0]):
                    W[i, :] = np.roll(x, i)
                W = -W
                W[W == 0] = 1
                W *= 0.2
            case "custom":
                if weights is None:
                    raise ValueError("No weights provided!")
                W = weights
            case "zero":
                W = 0
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Gaussian(0, 1e-3),
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
    return 2 if t > 20.0 else 0.0


_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1, 1))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
height, width = X_train[0].shape
image_size = X_train[0].size

# Simulation parameters
dt = 1e-4
max_rate = 100
amp = 1.0 / max_rate
rate_target = max_rate * amp  # must be in amplitude scaled units

n_features = 10
n_hidden_neurons = 100
n_coding_neurons = 64
presentation_time = 0.5
duration = 10

learning_rate = 1e-3
delay = Delay(1, timesteps=int(0.1 / dt))

layer_confs = [
    dict(
        name="inhibition",
        neuron=None,
        output=inhib,
        size_in=0,
    ),
    dict(
        name="state",
        neuron=None,
        output=lambda t: y_train[int(t / presentation_time)],
    ),
    dict(
        name="state_ens",
        n_neurons=n_coding_neurons,
        dimensions=1,
        radius=np.pi,
    ),
    dict(
        name="delayed_state",
        neuron=None,
        output=delay.step,
        size_in=1,
        radius=np.pi,
    ),
    dict(
        name="delayed_state_ens",
        n_neurons=10,
        dimensions=1,
        neuron=nengo.LIF(),
        radius=np.pi,
    ),
    dict(
        name="delta_state_ens",
        n_neurons=10,
        dimensions=1,
        neuron=nengo.LIF(),
        radius=np.pi,
    ),
    dict(
        name="stimulus",
        neuron=None,
        output=lambda t: X_train[int(t / presentation_time)].ravel(),
    ),
    dict(
        name="stim_ens",
        n_neurons=image_size,
        dimensions=1,
        radius=1,
        max_rates=nengo.dists.Choice([rate_target]),
        on_chip=False,
    ),
    dict(
        name="hidden_ens",
        n_neurons=n_hidden_neurons,
        dimensions=1,
        radius=1,
        max_rates=nengo.dists.Choice([rate_target]),
    ),
    dict(
        name="output_ens",
        n_neurons=image_size,
        dimensions=1,
        radius=1,
        max_rates=nengo.dists.Choice([rate_target]),
    ),
    dict(
        name="coding_ens",
        n_neurons=n_coding_neurons,
        dimensions=1,
        radius=np.pi,
        max_rates=nengo.dists.Choice([rate_target]),
    ),
    dict(
        name="error_ens",
        n_neurons=n_coding_neurons,
        dimensions=1,
        radius=np.pi,
    ),
    dict(
        name="reconstruction_error_ens",
        n_neurons=image_size,
        dimensions=1,
        max_rates=nengo.dists.Choice([rate_target]),
        radius=1,
    ),
]

conn_confs = [
    # state variable: angle
    dict(
        pre="state",
        post="state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state",
        post="delayed_state",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delayed_state",
        post="delayed_state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_ens",
        post="delta_state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delayed_state_ens",
        post="delta_state_ens",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="stimulus",
        post="stim_ens_neurons",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="stim_ens_neurons",
        post="hidden_ens_neurons",
        synapse=0.01,
        learning_rule=nengo.PES(learning_rate=learning_rate),
    ),
    dict(
        pre="hidden_ens_neurons",
        post="output_ens_neurons",
        transform=gen_transform(
            "custom",
            weights=np.random.normal(0, 1e-3, size=(image_size, n_hidden_neurons)),
        ),
        synapse=0.01,
    ),
    dict(
        pre="hidden_ens_neurons",
        post="coding_ens_neurons",
        learning_rule=nengo.PES(learning_rate=learning_rate),
    ),
    dict(
        pre="coding_ens",
        post="error_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_ens",
        post="error_ens",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="stim_ens_neurons",
        post="reconstruction_error_ens_neurons",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="output_ens_neurons",
        post="reconstruction_error_ens_neurons",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="inhibition",
        post="error_ens_neurons",
        transform=gen_transform("custom", weights=np.ones((n_coding_neurons, 1)) * -3),
        synapse=0.01,
    ),
]

learning_confs = [
    dict(
        pre="error_ens",
        post="hidden_ens_neurons2coding_ens_neurons",
    ),
    dict(
        pre="error_ens",
        post="stim_ens_neurons2hidden_ens_neurons",
    ),
]


default_neuron = nengo.AdaptiveLIF()
default_intercepts = nengo.dists.Choice([0, 0.1])
default_rates = nengo.dists.Choice([200])

# Create the Nengo model
with nengo.Network(label="tactile_encoding_net") as model:
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
        intercepts = layer_conf.pop("intercepts", default_intercepts)
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
                intercepts=intercepts,
                neuron_type=neuron_type,
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

    layers["error_ens"].encoders = layers["state_ens"].encoders

"""Run in command line mode
"""
with nengo.Simulator(model) as sim:
    sim.run(duration)

conn_name = "{}2{}".format("hidden_ens_neurons", "coding_ens_neurons")
ens1 = "state_ens"
ens2 = "coding_ens"
ens3 = "error_ens"

plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
# Find weight row with max variance
neuron = np.argmax(np.mean(np.var(sim.data[probes[conn_name]], axis=0), axis=1))
plt.plot(sim.trange(), sim.data[probes[conn_name]][:, neuron, :])
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[probes[ens1]], label=ens1[:-4])
plt.plot(sim.trange(), sim.data[probes[ens2]], label=ens2[:-4])
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[probes[ens3]], label=ens3[:-4])
plt.xlabel("time (s)")
plt.legend()
plt.show()
