import nengo
import numpy as np

_HEIGHT, _WIDTH = 15, 15
_SIZE_IN = _HEIGHT * _WIDTH
UINT8_MAX, UINT8_MIN = np.iinfo(np.uint8).max, np.iinfo(np.uint8).min
_MAX_RATE = 150

default_neuron = nengo.AdaptiveLIF(amplitude=1)
default_intercepts = nengo.dists.Choice([0, 0.1])
default_transform = nengo.dists.Gaussian(0.5, 0.1)
default_rates = nengo.dists.Choice([_MAX_RATE])

def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


def log(node, x):
    node.get_logger().info("data: {}".format(x))


layer_confs = [
    dict(
        name="input_layer",
        n_neurons=_SIZE_IN,
        max_rates=default_rates,
        neuron=nengo.PoissonSpiking(nengo.LIFRate()),
        on_chip=False,
    ),
    dict(
        name="hidden_layer",
        n_neurons=36,
        max_rates=default_rates,
        recurrent=False,
        recurrent_learning_rule=nengo.BCM(),
    ),
    dict(
        name="output_layer",
        n_neurons=10,
        max_rates=default_rates,
    ),
]
