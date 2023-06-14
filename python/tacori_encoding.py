# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import nengo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
from BarGenerator import BarGenerator
from learning_rules import SDSP
from nengo_extras.plot_spikes import plot_spikes
from orientation_map import sample_bipole_gaussian

font = {"weight": "normal", "size": 30}
matplotlib.rc("font", **font)

n_filters = 9
kern_size = 7
stim_shape = (20, 20)
stim_size = np.prod(stim_shape)
num_samples = 18

# Prepare dataset
bg = BarGenerator(stim_shape)
X_in, y_in = bg.generate_samples(
    num_samples=num_samples,
    dim=(3, 20),
    shift=(0, 0),
    start_angle=0,
    step=180 / num_samples,
    add_test=True,
)

# Simulation parameters
dt = 2e-3
filter_size = (stim_shape[0] - kern_size) // (kern_size - 1) + 1
n_hidden = filter_size * filter_size * n_filters
n_output = num_samples
decay_time = 0.02
presentation_time = 0.5 + decay_time
duration = X_in.shape[0] * presentation_time
sample_every = 5 * dt
learning_rule = SDSP()

# Default neuron parameters
max_rate = 60  # Hz
amp = 1.0
tau_rc = 0.02
rate_target = max_rate * amp  # must be in amplitude scaled units
default_neuron = nengo.LIF(amplitude=amp, tau_rc=tau_rc)
default_rates = nengo.dists.Choice([rate_target])
default_intercepts = nengo.dists.Choice([0])
default_encoders = nengo.dists.ScatteredHypersphere(surface=True)


def gen_transform(pattern=None, **kwargs):
    def inner(shape):
        """Closure of the transform matrix generator.

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            inner: Function that returns the transform matrix.
        """
        W: Optional[np.ndarray] = None

        match pattern:
            case "bipolar_gaussian_conv":
                try:
                    ksize = kwargs["ksize"]
                    nfilter = kwargs["nfilter"]
                except KeyError as _:
                    ksize = 5
                    nfilter = 10
                kernel = np.empty((ksize, ksize, 1, nfilter))
                delta_phi = np.pi / (nfilter + 1)
                strides = (ksize - 1, ksize - 1)
                for i in range(nfilter):
                    kernel[:, :, 0, i] = sample_bipole_gaussian(
                        (ksize, ksize),
                        (ksize // 2, ksize // 2),
                        (2, 0.5),
                        i * delta_phi,
                        binary=False,
                    )
                return nengo.Convolution(
                    n_filters=nfilter,
                    input_shape=stim_shape + (1,),
                    kernel_size=(ksize, ksize),
                    strides=strides,
                    init=kernel,
                )
            case "circular_inhibition":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = np.empty(shape)
                weight = np.abs(np.arange(shape[0]) - shape[0] // 2)
                for i in range(shape[0]):
                    W[i, :] = -np.roll(weight, i + shape[0] // 2)
            case _:
                if "weights" in kwargs:
                    W = kwargs["weights"]
                else:
                    W = nengo.Dense(
                        shape,
                        init=nengo.dists.Gaussian(0, 1e-2),
                    )
        return W

    return inner


def stim_func(t):
    Xid = int(t / presentation_time) % len(X_in)
    stage = t - int(t / presentation_time) * presentation_time
    sample = X_in[Xid].ravel()
    if stage <= presentation_time - decay_time:
        return sample
    else:
        return np.zeros_like(sample)


def target_func(t):
    target = y_in[int(t / presentation_time) % len(y_in)]
    stage = t - int(t / presentation_time) * presentation_time
    if t < 0.5 * duration and stage <= presentation_time - decay_time:
        n_target = int(target / (180 / n_output))
        spike = -30 * np.ones(n_output)
        spike[n_target] = 1
        return spike
    else:
        return np.zeros(n_output)


# Define layers
layer_confs = [
    # Input/Visual layers
    dict(
        name="target",
        neuron=None,
        output=target_func,
    ),
    dict(
        name="stimulus",
        neuron=None,
        output=stim_func,
    ),
    dict(
        name="visual",
        n_neurons=stim_size,
        dimensions=1,
    ),
    dict(
        name="view_target",
        n_neurons=n_output,
    ),
    dict(
        name="output",
        n_neurons=n_output,
        dimensions=1,
    ),
]

# Define connections
conn_confs = [
    # Input/Visual connections
    dict(
        pre="stimulus",
        post="visual_neurons",
        transform=1,
        synapse=0,
    ),
    dict(
        pre="target",
        post="view_target_neurons",
        transform=1,
        synapse=0,
    ),
    dict(
        pre="visual_neurons",
        post="output_neurons",
        transform=gen_transform(),
        learning_rule=learning_rule,
        synapse=0,
    ),
    dict(
        pre="output_neurons",
        post="output_neurons",
        transform=gen_transform("circular_inhibition"),
        synapse=5e-3,
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
            name_ = name + "_X"
            probes[name_] = nengo.Probe(
                conn.learning_rule,
                "X",
                synapse=0.01,
                sample_every=sample_every,
                label=f"{name_}",
            )
        connections[name] = conn

        # Probe weights
        name_ = name + "_weights"
        probes[name_] = nengo.Probe(
            conn,
            "weights",
            synapse=0.01,
            sample_every=sample_every,
            label=f"{name_}",
        )

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


def main(plot=False, savedata=False):
    with nengo.Simulator(model, dt=dt, optimize=True) as sim:
        sim.run(duration)

    ens_names = ["visual_neurons", "output_neurons"]
    conn_names = [f"{ens_names[0]}2{ens_names[1]}"]

    if savedata:
        with open("tacnet.pickle", "wb") as f:
            pickle.dump(sim, f)

    if plot:
        _, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        for conn_name in conn_names:
            axs[0].plot(
                sim.trange(sample_every=sample_every),
                sim.data[probes[conn_name + "_weights"]][:, 10, :],
            )
            axs[0].set_title("weight")
            axs[1].plot(
                sim.trange(sample_every=sample_every),
                sim.data[probes[conn_name + "_X"]][:, 10, :],
            )
            axs[1].set_title("synaptic variable")
        plt.xlabel("time (s)")
        plt.tight_layout()

        _, axs = plt.subplots(
            len(ens_names), 1, figsize=(5 * len(ens_names), 10), sharex=True
        )
        for i, ens_name in enumerate(ens_names):
            plot_spikes(
                sim.trange(sample_every=sample_every),
                sim.data[probes[ens_name]],
                ax=axs[i],
            )
            axs[i].set_ylabel("neuron index")
            axs[i].set_title(ens_name)
            axs[i].grid()
        plt.xlabel("time (s)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot=True, savedata=False)
