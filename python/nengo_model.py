 # Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import os

import nengo
import numpy as np
from BarGenerator import BarGenerator, gen_transform
from IPython.lib.deepreload import reload
from learning_rules import SDSP


# Define network parameters
stim_shape = (12, 12)
stim_size = np.prod(stim_shape)
num_samples = 16
repeats = 1

# Prepare dataset
bg = BarGenerator(stim_shape)
X_in, y_in = bg.generate_samples(
    num_samples=num_samples,
    dim=(3, 10),
    shift=(0, 0),
    start_angle=0,
    step=180 / num_samples,
    repeats=repeats,
    add_test=True,
)

# Simulation parameters
dt = 1e-3
n_output = num_samples
decay_time = 0.2
presentation_time = 1 + decay_time
duration = X_in.shape[0] * presentation_time
train_duration = repeats / (repeats + 1) * duration
sample_every = 10 * dt
learning_rule_option = {
    "SDSP": SDSP(
        J_C=20,
        tau_c=6e-2,
        tau_j=5e-2,
        X_coeff=(0.1, 0.1, 3.5, 3.5),  # (a, b, alpha, beta)
        theta_coeff=(13, 3, 4, 3),  # (theta_hup, theta_lup, theta_hdown, theta_ldown)
    ),
}

# Default neuron parameters
max_rates = 60  # Hz
amp = 1.0

tau_rc = 0.02
rate_target = max_rates * amp  # must be in amplitude scaled units
default_neuron = nengo.LIF(amplitude=amp, tau_rc=tau_rc)
default_rates = nengo.dists.Choice([rate_target])
default_intercepts = nengo.dists.Choice([0])
default_encoders = nengo.dists.ScatteredHypersphere(surface=True)


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
    spike = np.zeros(n_output + 1)
    if stage <= presentation_time - decay_time:
        n_target = int(target / (180 / n_output))
        spike[n_target] = 1
        spike[-1] = 1.0 if t < train_duration else 0.0
    return np.zeros(n_output + 1)


# Define layers
layer_confs = [
    dict(
        name="stimulus",
        neuron=None,
        output=stim_func,
    ),
    dict(
        name="input",
        n_neurons=stim_size,
        dimensions=1,
    ),
    dict(
        name="output",
        neuron=nengo.PoissonSpiking(default_neuron),
        n_neurons=n_output,
        dimensions=1,
    ),
    dict(
        name="switch",
        neuron=None,
        output=lambda t: 1.0 if t < train_duration else 1.0,
    ),
]

# Define connections
conn_confs = [
    # Input/Visual connections
    dict(
        pre="stimulus",
        post="input_neurons",
        transform=1,
        synapse=0,
    ),
    dict(
        pre="input_neurons",
        post="output_neurons",
        transform=gen_transform(),
        learning_rule=learning_rule_option["SDSP"],
        synapse=0,
    ),
    dict(
        pre="output_neurons",
        post="output_neurons",
        transform=1*(np.eye(n_output) - 1),
        synapse=1e-3,
    ),
]


learning_confs = [
    dict(
        pre="switch",
        post="input_neurons2output_neurons",
    ),
]

# Create the Nengo model
with nengo.Network(label="tacnet_train", seed=1) as model:
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
                sample_every=sample_every,
                synapse=0.01,
                label="%s_neurons_probe" % name,
            )
            probes[name + "_neurons"] = probe
            probe = nengo.Probe(
                layer.neurons,
                "output",
                sample_every=sample_every,
                synapse=0.01,
            )
            probes[name + "_voltages"] = probe

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
            try:
                name_ = name + "_X"
                probes[name_] = nengo.Probe(
                    conn.learning_rule,
                    "X",
                    synapse=0.01,
                    sample_every=sample_every,
                    label=f"{name_}",
                )
                name_ = name + "_C"
                probes[name_] = nengo.Probe(
                    conn.learning_rule,
                    "C",
                    synapse=0.01,
                    sample_every=sample_every,
                    label=f"{name_}",
                )
            except:
                pass
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