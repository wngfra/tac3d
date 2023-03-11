import os
from typing import Optional

import nengo
import numpy as np
from SynapticSampling import SynapticSampling
from TouchDataset import TouchDataset

_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1.0, 1.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
image_size = X_train[0].size

# Simulation parameters
# dt = 1e-3
max_rate = 200
amp = 200.0 / max_rate
rate_target = max_rate * amp  # must be in amplitude scaled units

n_hidden = 36
n_output = 36
presentation_time = 0.2
sigma = 10

default_neuron = nengo.AdaptiveLIF(amplitude=amp)
default_intercepts = nengo.dists.Choice([0, 0.1])


layer_confs = [
    dict(
        name="input_layer",
        n_neurons=image_size,
        max_rates=nengo.dists.Choice([rate_target]),
        neuron=nengo.PoissonSpiking(nengo.LIFRate(amplitude=amp)),
        on_chip=False,
    ),
    dict(
        name="hidden_layer",
        n_neurons=n_hidden,
        recurrent=True,
        learning_rule=SynapticSampling(),
        recurrent_learning_rule=SynapticSampling()
    ),
    dict(
        name="output_layer",
        n_neurons=10,
    ),
]


def motion_func(t):
    index = np.random.randint(0, n_output, size=1)
    M = np.zeros(n_output, dtype=int)
    M[index] = 1
    return M


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
    shape_in = nengo.transforms.ChannelShape((image_size,))
    x = stim

    # Create layers
    for k, layer_conf in enumerate(layer_confs):
        layer_conf = dict(layer_conf)  # copy, so we don't modify the original
        name = layer_conf.pop("name")
        intercepts = layer_conf.pop("intercepts", default_intercepts)
        max_rates = layer_conf.pop("max_rates", None)
        neuron_type = layer_conf.pop("neuron", default_neuron)
        on_chip = layer_conf.pop("on_chip", True)
        block = layer_conf.pop("block", None)
        recurrent = layer_conf.pop("recurrent", False)
        learning_rule = layer_conf.pop("learning_rule", None)
        recurrent_learning_rule = layer_conf.pop("recurrent_learning_rule", None)

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
                    init=nengo.dists.Gaussian(0.5, 0.1),
                )
            else:
                transform = 1
            if recurrent:
                transform_reccurent = nengo.Dense(
                    (shape_in.size, shape_out.size),
                    init=nengo.dists.Gaussian(0.5, 0.1),
                )

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
                intercepts=intercepts,
                neuron_type=neuron_type,
                label=name,
            )
            y = ens.neurons

            # Add a probe so we can measure individual layer rates
            probe = nengo.Probe(y, synapse=0.01, label="%s_p" % name)
            layer_probes.append(probe)

        conn = nengo.Connection(x, y, transform=transform, learning_rule_type=learning_rule)
        transforms.append(transform)
        connections.append(conn)

        if recurrent:
            conn_recurrent = nengo.Connection(y, x, transform=transform_reccurent, learning_rule_type=recurrent_learning_rule)
            transforms.append(transform_reccurent)
            connections.append(conn_recurrent)
        x = y
        shape_in = shape_out


"""Run in command line mode
"""
with nengo.Simulator(model) as sim:
    sim.run(5.0)
