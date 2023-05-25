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

from OrientationMap import sample_bipole_gaussian

font = {"weight": "normal", "size": 30}
matplotlib.rc("font", **font)

N_FILTERS = 18
KERN_SIZE = 5
STRIDES = (KERN_SIZE - 1, KERN_SIZE - 1)
STIM_SHAPE = (15, 15)


def gen_transform(pattern=None):
    def inner(shape):
        """Closure of the transform matrix generator.

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            inner: Function that returns the transform matrix.
        """
        W: Optional[np.ndarray] = None

        match pattern:
            case "orthogonal_excitation":
                W = np.zeros(shape)
                target = np.arange(shape[1])
                n = shape[1] // shape[0]
                np.random.shuffle(target)
                for i in range(shape[0]):
                    W[i, target[i * n : (i + 1) * n]] = np.random.uniform(0.5, 0.1)
                W[W < 0.1] = 1e-2
            case "bipolar_gaussian_conv":
                kernel = np.empty((KERN_SIZE, KERN_SIZE, 1, N_FILTERS))
                delta_phi = np.pi / (N_FILTERS + 1)
                for i in range(N_FILTERS):
                    kernel[:, :, 0, i] = sample_bipole_gaussian(
                        (KERN_SIZE, KERN_SIZE),
                        (KERN_SIZE // 2, KERN_SIZE // 2),
                        (3, 0.5),
                        i * delta_phi,
                        binary=False,
                    )
                conv = nengo.Convolution(
                    n_filters=N_FILTERS,
                    input_shape=STIM_SHAPE + (1,),
                    kernel_size=(KERN_SIZE, KERN_SIZE),
                    strides=STRIDES,
                    init=kernel,
                )
                return conv
            case "circular_inhibition":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = np.empty(shape)
                weight = np.abs(np.arange(shape[0]) - shape[0] // 2)
                for i in range(shape[0]):
                    W[i, :] = -np.roll(weight, i + shape[0] // 2)
                W *= 5 / np.max(np.abs(W))
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Uniform(0, 1e-5),
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


stim_size = np.prod(STIM_SHAPE)
bg = BarGenerator(STIM_SHAPE)

# Prepare dataset
num_samples = 18
X_train, y_train = bg.gen_sequential_bars(
    num_samples=num_samples,
    dim=(2, 20),
    shift=(1, 1),
    start_angle=0,
    step=360 / num_samples,
)
y_train = y_train / 90 - 1


# Simulation parameters
dt = 1e-3
K = (STIM_SHAPE[0] - KERN_SIZE) // STRIDES[0] + 1
n_hidden_neurons = K * K * N_FILTERS
n_coding_neurons = 4 * N_FILTERS
n_state_neurons = 16
decay_time = 0.05
presentation_time = 0.3 + decay_time  # Leave 0.05s for decay
duration = num_samples * presentation_time
sample_every = 1 * dt
learning_rule = SynapticSampling()

delay = Delay(1, timesteps=int(0.1 / dt))

# Default neuron parameters
max_rate = 60  # Hz
amp = 1.0
rate_target = max_rate * amp  # must be in amplitude scaled units
default_neuron = nengo.AdaptiveLIF(amplitude=amp, tau_rc=0.05)
default_rates = nengo.dists.Choice([rate_target])
default_intercepts = nengo.dists.Choice([0])
default_encoders = nengo.dists.ScatteredHypersphere(surface=True)


def stim_func(t):
    Xid = int(t / presentation_time) % len(X_train)
    stage = t - int(t / presentation_time) * presentation_time
    sample = X_train[Xid].ravel()
    if stage <= presentation_time - decay_time:
        return sample
    else:
        return np.zeros_like(sample)


def state_func(t):
    return 0


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
        output=stim_func,
    ),
    dict(
        name="stim",
        n_neurons=stim_size,
        dimensions=1,
        on_chip=False,
    ),
    dict(
        name="hidden",
        n_neurons=n_hidden_neurons,
        dimensions=1,
    ),
    dict(
        name="coding",
        n_neurons=n_coding_neurons,
        dimensions=N_FILTERS,
    ),
]

# Define connections
conn_confs = [
    dict(
        pre="state_node",
        post="delay_node",
        transform=1,
    ),
    dict(
        pre="state_node",
        post="state",
        dim_out=0,
        transform=1,
    ),
    dict(
        pre="delay_node",
        post="state",
        dim_out=1,
        transform=1,
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
        transform=1,
        synapse=0,
    ),
    dict(
        pre="stim_neurons",
        post="hidden_neurons",
        transform=gen_transform("bipolar_gaussian_conv"),
        synapse=5e-3,
    ),
    dict(
        pre="hidden_neurons",
        post="coding_neurons",
        transform=gen_transform("orthogonal_excitation"),
        learning_rule=learning_rule,
        synapse=5e-3,
    ),
    dict(
        pre="coding_neurons",
        post="coding_neurons",
        transform=gen_transform("circular_inhibition"),
        synapse=2e-3,
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
        layer_conf = dict(layer_conf)  # Copy configuration
        name = layer_conf.pop("name")
        n_neurons = layer_conf.pop("n_neurons", 1)
        dimensions = layer_conf.pop("dimensions", 1)
        encoders = layer_conf.pop("encoders", default_encoders)
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
        conn_conf = dict(conn_conf)
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
        if callable(transform):
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


def main(plot=False):
    with nengo.Simulator(model, dt=dt, optimize=True) as sim:
        sim.run(duration)

    conn_name = "{}2{}".format("hidden_neurons", "coding_neurons")
    ens_names = ["stim_neurons", "hidden_neurons", "coding_neurons"]

    # save_data(sim, ["hidden_neurons", "state"], "data/test_data.csv")

    if plot:
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


def save_data(sim, save_list, filename):
    import pandas as pd

    data = []
    labels = []
    for name in save_list:
        data.append(sim.data[probes[name]])
        labels += [name + "_{}".format(i) for i in range(data[-1].shape[1])]
    df = pd.DataFrame(
        np.hstack(data), index=sim.trange(sample_every=sample_every), columns=labels
    )
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main(True)
