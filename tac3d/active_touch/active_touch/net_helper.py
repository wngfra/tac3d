import nengo
import numpy as np

HEIGHT, WIDTH = 15, 15
_SIZE_IN = HEIGHT * WIDTH
_MAX_RATE = 150
_AMP = 200.0 / _MAX_RATE
_RATE_TARGET = _MAX_RATE * _AMP 

# Default ensemble configs
default_intercepts = nengo.dists.Choice([0, 0.1])
default_neuron = nengo.AdaptiveLIF(amplitude=1)
default_rates = nengo.dists.Choice([_MAX_RATE])
default_transform = nengo.dists.Gaussian(0.5, 0.1)

# Default connection configs
default_learning_rule = nengo.BCM(1e-9)

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
