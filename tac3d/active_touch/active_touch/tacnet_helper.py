from typing import Optional

import nengo
import numpy as np

image_height, image_width = 15, 15
image_size = image_height * image_width

# Constants
_MAX_RATE = 150
_AMP = 1.0

# Network params
n_hidden = 32
n_encoding = 11  # Odd number of neurons for cyclic interpretation
default_neuron = nengo.AdaptiveLIF()
default_rates = nengo.dists.Choice([_MAX_RATE])
default_intercepts = nengo.dists.Choice([0, 0.1])


def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


def log(node, x):
    node.get_logger().info("data: {}".format(x))


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


layer_confs = [
    dict(
        name="stimulus",
        neuron=None,
    ),
    dict(
        name="input",
        n_neurons=image_size,
        radius=1,
        neuron=nengo.PoissonSpiking(nengo.LIFRate()),
        on_chip=False,
    ),
    dict(
        name="hidden",
        n_neurons=n_hidden,
        neuron=default_neuron,
        radius=2,
    ),
    dict(name="output", n_neurons=image_size, radius=1, neuron=default_neuron),
    dict(
        name="encoding",
        n_neurons=n_encoding,
        radius=2,
        neuron=default_neuron,
    ),
]

conn_confs = [
    dict(
        pre="stimulus_node",
        post="input",
        synapse=None,
        transform=gen_transform("identity_exhibition"),
        learning_rule=None,
    ),
    dict(
        pre="input",
        post="hidden",
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        pre="hidden",
        post="output",
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        pre="input",
        post="output",
        transform=gen_transform("identity_inhibition"),
        learning_rule=None,
    ),
    dict(
        pre="hidden",
        post="encoding",
        learning_rule=nengo.BCM(1e-9),
    ),
    dict(
        pre="encoding",
        post="encoding",
        transform=gen_transform("cyclic_inhibition"),
    ),
]
