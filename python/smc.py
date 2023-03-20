import os
from typing import Optional

import nengo
import numpy as np
from SynapticSampling import SynapticSampling
from TouchDataset import TouchDataset


def gen_transform(pattern="random"):
    W: Optional[np.ndarray] = None

    def inner(shape):
        """_summary_

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            _type_: _description_
        """
        match pattern:
            case "identity_exhibition":
                W = 1
            case "identity_inhibition":
                W = 1
            case "uniform_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = np.ones((shape[0], shape[0])) - np.eye(shape[0])
            case "cyclic_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                xmax = shape[1] // 2
                x = np.abs(np.arange(shape[0]) - xmax)
                W = np.empty((shape[0], shape[0]))
                for i in range(shape[0]):
                    W[i, :] = np.roll(x, i)
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Gaussian(0.1, 0.2),
                )
        if "inhibition" in pattern:
            W *= -1
        return W

    return inner


_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-2.0, 2.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
image_size = X_train[0].size

# Simulation parameters
# dt = 1e-3
max_rate = 100
amp = 1.0 / max_rate
rate_target = max_rate * amp  # must be in amplitude scaled units

n_hidden = 32
n_output = 11  # Odd number of neurons for cyclic interpretation
presentation_time = 0.2

default_neuron = nengo.AdaptiveLIF(amplitude=amp)
default_intercepts = nengo.dists.Choice([0, 0.1])

layer_confs = [
    dict(
        name="input_layer",
        n_neurons=image_size,
        radius=1,
        max_rates=nengo.dists.Choice([rate_target]),
        neuron=nengo.PoissonSpiking(nengo.LIFRate()),
        on_chip=False,
    ),
    dict(
        name="hidden_layer",
        n_neurons=n_hidden,
        radius=2,
    ),
    dict(
        name="output_layer",
        n_neurons=image_size,
        radius=1,
    ),
    dict(
        name="coding_layer",
        n_neurons=n_output,
        radius=2,
    ),
]

conn_confs = [
    dict(
        pre="stimulus_node",
        post="input_layer",
        synapse=None,
        transform=gen_transform("identity_exhibition"),
        learning_rule=None,
    ),
    dict(
        pre="input_layer",
        post="hidden_layer",
        learning_rule=nengo.BCM(1e-9),
        synapse=0.1,
    ),
    dict(
        pre="hidden_layer",
        post="output_layer",
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        pre="input_layer",
        post="output_layer",
        transform=gen_transform("uniform_inhibition"),
        learning_rule=None,
    ),
    dict(
        pre="hidden_layer",
        post="coding_layer",
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        pre="coding_layer",
        post="coding_layer",
        transform=gen_transform("cyclic_inhibition"),
    ),
]


def motion_func(t):
    index = np.random.randint(0, n_output, size=1)
    M = np.zeros(n_output, dtype=int)
    M[index] = 1
    return M


# Create the Nengo model
with nengo.Network(label="smc") as model:
    layers = dict()
    connections = dict()
    probes = dict()

    truth = nengo.Node(
        lambda t: y_train[int(t / presentation_time)]
        - np.floor(y_train[int(t / presentation_time)] / np.pi) * np.pi,
        label="ground_truth_node",
    )
    stim = nengo.Node(
        lambda t: X_train[int(t / presentation_time)].ravel(), label="stimulus_node"
    )
    layers["stimulus_node"] = stim

    # Create layers
    for k, layer_conf in enumerate(layer_confs):
        layer_conf = dict(layer_conf)  # Copy layer configuration
        name = layer_conf.pop("name")
        n_neurons = layer_conf.pop("n_neurons")
        intercepts = layer_conf.pop("intercepts", default_intercepts)
        max_rates = layer_conf.pop("max_rates", nengo.dists.Choice([max_rate]))
        radius = layer_conf.pop("radius", 1.0)
        neuron_type = layer_conf.pop("neuron", default_neuron)
        on_chip = layer_conf.pop("on_chip", True)
        block = layer_conf.pop("block", None)
        learning_rule = layer_conf.pop("learning_rule", None)
        loc = "chip" if on_chip else "host"

        assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            layer = nengo.Node(size_in=n_neurons, label=name)
            layers[name] = layer
        else:
            layer = nengo.Ensemble(
                n_neurons,
                1,
                radius=radius,
                intercepts=intercepts,
                neuron_type=neuron_type,
                label=name,
            )
            layers[name] = layer.neurons

            # Add a probe so we can measure individual layer rates
            probe = nengo.Probe(layers[name], synapse=0.01, label="%s_probe" % name)
            probes[name] = probe

    for k, conn_conf in enumerate(conn_confs):
        conn_conf = dict(conn_conf)  # Copy connection configuration
        pre = conn_conf.pop("pre")
        post = conn_conf.pop("post")
        synapse = conn_conf.pop("synapse", 0)
        transform = conn_conf.pop("transform", gen_transform())
        learning_rule = conn_conf.pop("learning_rule", None)
        name = "conn_{}-{}".format(pre, post)

        assert len(conn_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )
        
        conn = nengo.Connection(
            layers[pre],
            layers[post],
            transform=transform((layers[post].size_in, layers[pre].size_in)),
            learning_rule_type=learning_rule,
            synapse=synapse,
            label=name,
        )
        # transforms[name] = transform
        connections[name] = name

        probe = nengo.Probe(
            conn, "weights", synapse=0.01, label="weights_{}".format(name)
        )
        probes[name] = probe


"""Run in command line mode
"""
with nengo.Simulator(model) as sim:
    sim.run(10.0)

conn_name = "conn_{}-{}".format("hidden_layer", "coding_layer")
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(1, 1, 1)
# Find weight row with max variance
neuron = np.argmax(np.mean(np.var(sim.data[probes[conn_name]], axis=0), axis=1))
plt.plot(sim.trange(), sim.data[probes[conn_name]][..., neuron])
plt.ylabel("Connection weight")
plt.show()

