import os
from typing import Optional

import nengo
import numpy as np
from SynapticSampling import SynapticSampling
from TouchDataset import TouchDataset

_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-2.0, 2.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
image_size = X_train[0].size

# Simulation parameters
# dt = 1e-3
max_rate = 200
amp = 200.0 / max_rate
rate_target = max_rate * amp  # must be in amplitude scaled units

n_hidden = 18
n_output = 11  # Odd number of neurons for cyclic interpretation
presentation_time = 0.2
sigma = 10

default_neuron = nengo.AdaptiveLIF(amplitude=amp)
default_intercepts = nengo.dists.Choice([0, 0.1])


layer_confs = [
    dict(
        name="input_layer",
        n_neurons=image_size,
        radius=1,
        max_rates=nengo.dists.Choice([rate_target]),
        neuron=nengo.AdaptiveLIF(amplitude=amp),
        on_chip=False,
    ),
    dict(
        name="hidden_layer",
        n_neurons=n_hidden,
        radius=2,
        recurrent=False,
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        name="output_layer",
        n_neurons=n_output,
        radius=2,
        wta="self_cyclic",
        learning_rule=nengo.BCM(1e-9),
    ),
]


def motion_func(t):
    index = np.random.randint(0, n_output, size=1)
    M = np.zeros(n_output, dtype=int)
    M[index] = 1
    return M


def gen_wta_weights(shape, pattern):
    W: Optional[np.ndarray] = None
    match pattern:
        case "self_uniform":
            W = np.ones((shape[1], shape[1])) - np.eye(shape[1])
        case "self_cyclic":
            xmax = shape[1] // 2
            x = np.abs(np.arange(shape[1]) - xmax)
            W = np.empty((shape[1], shape[1]))
            for i in range(shape[1]):
                W[i, :] = np.roll(x, i)
    if W is not None:
        W *= -1
    return W


# Create the Nengo model
with nengo.Network(label="smc") as model:
    truth = nengo.Node(
        lambda t: y_train[int(t / presentation_time)]
        - np.floor(y_train[int(t / presentation_time)] / np.pi) * np.pi
    )
    stim = nengo.Node(lambda t: X_train[int(t / presentation_time)].ravel())

    connections = []
    transforms = []
    layer_probes = []
    conn_probes = []
    shape_in = nengo.transforms.ChannelShape((image_size,))
    x = stim

    # Create layers
    for k, layer_conf in enumerate(layer_confs):
        layer_conf = dict(layer_conf)  # copy, so we don't modify the original
        name = layer_conf.pop("name")
        intercepts = layer_conf.pop("intercepts", default_intercepts)
        max_rates = layer_conf.pop("max_rates", None)
        radius = layer_conf.pop("radius", 1.0)
        neuron_type = layer_conf.pop("neuron", default_neuron)
        on_chip = layer_conf.pop("on_chip", True)
        block = layer_conf.pop("block", None)
        recurrent = layer_conf.pop("recurrent", False)
        learning_rule = layer_conf.pop("learning_rule", None)
        recurrent_learning_rule = layer_conf.pop("recurrent_learning_rule", None)
        wta = layer_conf.pop("wta", False)

        # Create layer transform
        if "filters" in layer_confs:
            # Convolutional layer
            pass
        else:
            # Dense layer
            n_neurons = layer_conf.pop("n_neurons")
            shape_out = nengo.transforms.ChannelShape((n_neurons,))
            if name != "input_layer":
                transform = nengo.Dense(
                    (shape_out.size, shape_in.size),
                    init=nengo.dists.Gaussian(0., 0.3),
                )
            else:
                transform = 1
            if recurrent:
                transform_reccurent = nengo.Dense(
                    (shape_in.size, shape_out.size),
                    init=nengo.dists.Gaussian(0., 0.3),
                )
            if wta:
                transform_wta = gen_wta_weights((shape_in.size, shape_out.size), wta)

            loc = "chip" if on_chip else "host"

        assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            y = nengo.Node(size_in=shape_out.size, label=name)
        else:
            ens = nengo.Ensemble(
                shape_out.size,
                1,
                radius=radius,
                intercepts=intercepts,
                neuron_type=neuron_type,
                label=name,
            )
            y = ens.neurons

            # Add a probe so we can measure individual layer rates
            probe = nengo.Probe(y, synapse=0.01, label="%s_p" % name)
            layer_probes.append(probe)

        conn = nengo.Connection(
            x, y, transform=transform, learning_rule_type=learning_rule, synapse=0.01
        )
        transforms.append(transform)
        connections.append(conn)

        probe = nengo.Probe(conn, "weights", synapse=0.01, label="weights2{}".format(name))
        conn_probes.append(probe)

        if recurrent:
            conn_recurrent = nengo.Connection(
                y,
                x,
                transform=transform_reccurent,
                learning_rule_type=recurrent_learning_rule,
            )
            transforms.append(transform_reccurent)
            connections.append(conn_recurrent)
        if wta:
            conn_wta = nengo.Connection(y, y, transform=transform_wta)
            transforms.append(transform_wta)
            connections.append(conn_wta)
        x = y
        shape_in = shape_out


"""Run in command line mode

with nengo.Simulator(model) as sim:
    sim.run(10.0)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(1, 1, 1)
# Find weight row with max variance
neuron = np.argmax(np.mean(np.var(sim.data[conn_probes[1]], axis=0), axis=1))
plt.plot(sim.trange(), sim.data[conn_probes[1]][..., neuron])
plt.ylabel("Connection weight")
plt.show()
"""