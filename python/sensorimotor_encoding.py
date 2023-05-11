# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import nengo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from BarGenerator import BarGenerator
from custom_learning_rules import SynapticSampling
from nengo_extras.plot_spikes import plot_spikes
from scipy.stats import multivariate_normal

from OrientationMap import OrientationMap

font = {"weight": "normal", "size": 30}

matplotlib.rc("font", **font)


def gen_transform(pattern=None, weights=None):
    def inner(shape, weights=weights):
        """Closure of the transform matrix generator.

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            inner: Function that returns the transform matrix.
        """
        W: Optional[np.ndarray] = None

        match pattern:
            case "identity_excitation":
                if 0 in shape:
                    W = 1
                else:
                    W = np.ones(shape)
            case "identity_inhibition":
                if 0 in shape:
                    W = -1
                else:
                    W = -np.ones(shape)
            case "orientation_map":
                m = np.sqrt(shape[1]).astype(int)
                n = np.sqrt(shape[0]).astype(int)
                orimap = OrientationMap((n, n), d=2, alpha=2, theta=22.5)
                return orimap.gen_transform((m, m))
            case "gaussian":
                M = np.sqrt(shape[1])
                N = np.sqrt(shape[0])
                W = np.empty(shape)
                for i in range(shape[0]):
                    col, row = i // N, i % N
                    mean = np.array([col * M // N, row * M // N])
                    rv = multivariate_normal(mean, cov=[[1, 0], [0, 1]])
                    pos = np.dstack((np.mgrid[0:M:1, 0:M:1]))
                    W[i, :] = rv.pdf(pos).ravel()
            case "custom":
                if weights is None:
                    raise ValueError("No weights provided!")
                W = weights
            case "zero":
                if 0 in shape:
                    W = 0
                else:
                    W = np.zeros(shape)
            case "exclusive_excitation":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = np.ones(shape) - np.eye(shape[0])
            case "exclusive_inhibition":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = -np.ones(shape) + 2 * np.eye(shape[0])
            case "circular_inhibition":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = np.empty(shape)
                weight = np.abs(np.arange(shape[0]) - shape[0] // 2)
                for i in range(shape[0]):
                    W[i, :] = -np.roll(weight, i + shape[0] // 2)
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Uniform(0, 1e-3),
                )
        return W

    return inner


# Delay node
class Delay:
    def __init__(self, dims, timesteps=50):
        self.history = np.zeros((timesteps, dims))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]


# Function to inhibit the error population after half of the duration
def inhib(t):
    return 2 if t > duration * 0.5 else 0.0


stim_shape = (15, 15)
stim_size = np.prod(stim_shape)
bg = BarGenerator(stim_shape)

# Prepare dataset
num_samples = 18
X_train, y_train = bg.gen_sequential_bars(
    num_samples=num_samples,
    dim=(5, 20),
    center=(7, 7),
    start_angle=0,
    step=360 / num_samples,
)
y_train = y_train / 90 - 1


# Simulation parameters
dt = 1e-3
max_rate = 100  # Hz
amp = 1.0
rate_target = max_rate * amp  # must be in amplitude scaled units

n_hidden_neurons = 225
n_wta_neurons = 18
n_state_neurons = 18
presentation_time = 0.5
duration = (num_samples - 1) * presentation_time
sample_every = 10 * dt

learning_rate = 5e-8
delay = Delay(1, timesteps=int(0.1 / dt))

# Default neuron parameters
default_neuron = nengo.AdaptiveLIF(amplitude=amp, tau_rc=0.02)
default_rates = nengo.dists.Choice([rate_target])
default_intercepts = nengo.dists.Choice([0])

# Define layers
layer_confs = [
    dict(
        name="state_node",
        neuron=None,
        output=lambda t: y_train[int(t / presentation_time) % len(y_train)],
    ),
    dict(
        name="delay_node",
        neuron=None,
        output=delay.step,
        size_in=1,
    ),
    dict(name="state", n_neurons=2 * n_state_neurons, dimensions=2),
    dict(
        name="delta_state",
        n_neurons=n_state_neurons,
        dimensions=1,
    ),
    dict(
        name="stimulus",
        neuron=None,
        output=lambda t: X_train[int(t / presentation_time) % len(X_train)].ravel(),
    ),
    dict(
        name="stim",
        n_neurons=stim_size,
        dimensions=2,
        on_chip=False,
    ),
    dict(
        name="hidden",
        n_neurons=n_hidden_neurons,
        dimensions=1,
    ),
    dict(
        name="wta",
        n_neurons=n_wta_neurons,
        dimensions=1,
    ),
]

# Define connections
conn_confs = [
    dict(
        pre="state_node",
        post="delay_node",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_node",
        post="state",
        dim_out=0,
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delay_node",
        post="state",
        dim_out=1,
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state",
        post="delta_state",
        solver=nengo.solvers.LstsqL2(weights=True),
        function=lambda x: x[0] - x[1],
    ),
    dict(
        pre="stimulus",
        post="stim_neurons",
        transform=gen_transform("identity_excitation"),
        synapse=0.001,
    ),
    dict(
        pre="stim_neurons",
        post="hidden_neurons",
        transform=gen_transform("orientation_map"),
        synapse=1e-3,
    ),
    dict(
        pre="hidden_neurons",
        post="wta_neurons",
        transform=gen_transform(),
        #learning_rule=SynapticSampling(),
        learning_rule=nengo.BCM(learning_rate=learning_rate),
        synapse=0.01,
    ),
    dict(
        pre="wta_neurons",
        post="wta_neurons",
        transform=gen_transform("circular_inhibition"),
        synapse=0.001,
    ),
]

learning_confs = []


# Create the Nengo model
with nengo.Network(label="tacnet", seed=1) as model:
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
            probe = nengo.Probe(
                layer, synapse=0.01, sample_every=sample_every, label="%s_probe" % name
            )
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
            probe = nengo.Probe(
                layer, synapse=0.01, sample_every=sample_every, label="%s_probe" % name
            )
            probes[name] = probe
            probe = nengo.Probe(
                layer.neurons,
                synapse=0.01,
                sample_every=sample_every,
                label="%s_neurons_probe" % name,
            )
            probes[name + "_neurons"] = probe

    for k, conn_conf in enumerate(conn_confs):
        conn_conf = dict(conn_conf)  # Copy connection configuration
        pre = conn_conf.pop("pre")
        post = conn_conf.pop("post")
        dim_in = conn_conf.pop("dim_in", None)
        dim_out = conn_conf.pop("dim_out", None)
        synapse = conn_conf.pop("synapse", None)
        solver = conn_conf.pop("solver", None)
        transform = conn_conf.pop("transform", None)
        learning_rule = conn_conf.pop("learning_rule", None)
        name = "{}2{}".format(pre, post)
        function = conn_conf.pop("function", None)

        assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )
        if dim_in is None:
            pre_conn = layers[pre]
        else:
            pre_conn = layers[pre][dim_in]
        if dim_out is None:
            post_conn = layers[post]
        else:
            post_conn = layers[post][dim_out]
        if transform is not None:
            transform = transform((post_conn.size_in, pre_conn.size_in))
        conn = nengo.Connection(
            pre_conn,
            post_conn,
            function=function,
            transform=transform,
            synapse=synapse,
            label=name,
        )
        if solver:
            conn.solver = solver
        if learning_rule:
            conn.learning_rule_type = learning_rule
        connections[name] = conn

        probe = nengo.Probe(
            conn,
            "weights",
            synapse=0.01,
            sample_every=sample_every,
            label="weights_{}".format(name),
        )
        probes[name] = probe

        # Debug
        try:
            probes["SS"] = nengo.Probe(
                conn.learning_rule,
                "cov",
                synapse=0.01,
            )
        except Exception:
            pass

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


def main():
    with nengo.Simulator(model, dt=dt, optimize=True) as sim:
        sim.run(duration)

    conn_name = "{}2{}".format("hidden_neurons", "wta_neurons")
    ens_names = ["stim_neurons", "hidden_neurons", "wta_neurons"]

    plt.figure(figsize=(5, 10))
    # Find weight row with max variance
    neuron = np.argmax(np.mean(np.var(sim.data[probes[conn_name]], axis=0), axis=1))
    plt.plot(sim.trange(sample_every), sim.data[probes[conn_name]][:, neuron, :])
    plt.xlabel("time (s)")
    plt.ylabel("weights")

    _, axs = plt.subplots(len(ens_names), 1, figsize=(5 * len(ens_names), 10))
    for i, ens_name in enumerate(ens_names):
        if "neurons" in ens_name:
            plot_spikes(
                sim.trange(sample_every=sample_every),
                sim.data[probes[ens_name]],
                ax=axs[i],
            )
            axs[i].set_ylabel("neuron")
        else:
            axs[i].plot(
                sim.trange(sample_every=sample_every), sim.data[probes[ens_name]]
            )
            axs[i].set_ylabel("encoder")
        axs[i].set_title(ens_name)
        axs[i].set_xlabel("time (s)")
        axs[i].grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
