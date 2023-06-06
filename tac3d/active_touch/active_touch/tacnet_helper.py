from typing import Optional

import nengo
import numpy as np
from scipy.stats import multivariate_normal

stim_shape = (15, 15)
stim_size = np.prod(stim_shape)

# Constants
_MAX_RATE = 100
_AMP = 1.0
_RATE_TARGET = _MAX_RATE * _AMP

# Constatns for the convolutional layer
_N_FILTERS = 16
_KERN_SIZE = 5
_STRIDES = (_KERN_SIZE - 1, _KERN_SIZE - 1)

# Network params
dt = 1e-3
k = (stim_shape[0] - _KERN_SIZE) // _STRIDES[0] + 1
n_hidden = k * k * _N_FILTERS
n_output = 18
tau_rc = 0.02

# Default neuronal and synaptic parameters
default_neuron = nengo.AdaptiveLIF(amplitude=_AMP, tau_rc=tau_rc)
default_rates = nengo.dists.Choice([_RATE_TARGET])
default_intercepts = nengo.dists.Choice([0])
learning_rule = [nengo.BCM(5e-8), nengo.Oja(5e-6)]


def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


def log(node, x):
    node.get_logger().warn("DEBUG LOG: {}".format(x))


def sample_bipole_gaussian(shape, center, eigenvalues, phi, binary=False):
    """Sample from two 2D Gaussian with two peaks at the center of the bipole.
    Args:
        shape (tuple): Shape of the output array.
        center (tuple): Center of the bipole.
        eigenvalues (list like): Eigenvalues of the covariance matrix.
        phi (float): Rotation angle in radians.
    Returns:
        np.ndarray: 2D Gaussian array.
    """
    eigenvalues = np.asarray(eigenvalues)
    w0, w1 = eigenvalues.max(), eigenvalues.min()
    # Compute the mean (centre) of the two Gaussians
    mu = (
        np.tile(center, [2, 1])
        + np.asarray([[np.sin(phi), -np.cos(phi)], [-np.sin(phi), np.cos(phi)]]) * w1
    )
    # TODO: Check if two distributions are outside the shape.

    # Generate the meshgrid
    assert shape[0] == shape[1], "shape must be square"
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    pos = np.zeros(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Multivariate Gaussian ON cell
    V = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]).T
    D = [[w0, 0], [0, w1]]
    sigma = np.matmul(np.matmul(V, D), np.linalg.inv(V))
    rv_on = multivariate_normal(mu[0], sigma)
    rv_off = multivariate_normal(mu[1], sigma)
    pdf = rv_on.pdf(pos) - rv_off.pdf(pos)
    if binary:
        pdf[pdf < 0] = -1
        pdf[pdf > 0] = 1
    return pdf


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
            case "orthogonal_excitation":
                W = np.zeros(shape)
                target = np.arange(shape[1])
                n = shape[1] // shape[0]
                np.random.shuffle(target)
                for i in range(shape[0]):
                    W[i, target[i * n : (i + 1) * n]] = np.random.normal(0.3, 0.2)
                W[W < 0.1] = 0
            case "bipolar_gaussian_conv":
                try:
                    ksize = kwargs["ksize"]
                    nfilter = kwargs["nfilter"]
                except:
                    raise KeyError("Missing keyword arguments!")
                kernel = np.empty((ksize, ksize, 1, nfilter))
                delta_phi = np.pi / (nfilter + 1)
                for i in range(nfilter):
                    kernel[:, :, 0, i] = sample_bipole_gaussian(
                        (ksize, ksize),
                        (ksize // 2, ksize // 2),
                        (2, 0.5),
                        i * delta_phi,
                        binary=False,
                    )
                conv = nengo.Convolution(
                    n_filters=nfilter,
                    input_shape=stim_shape + (1,),
                    kernel_size=(ksize, ksize),
                    strides=_STRIDES,
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
                W = -np.expm1(-W)
            case _:
                if "weights" in kwargs:
                    W = kwargs["weights"]
                else:
                    W = nengo.Dense(
                        shape,
                        init=nengo.dists.Uniform(0, 0.3),
                    )
        return W

    return inner


class Delay:
    def __init__(self, dimensions, timesteps=50):
        self.history = np.zeros((timesteps, dimensions))

    def step(self, t, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        return self.history[0]


delay = Delay(n_output, timesteps=int(0.1 / dt))

# Define layers
layer_confs = [
    dict(
        name="stimulus",
        neuron=None,
        output="input_func",
    ),
    dict(
        name="visual",
        n_neurons=stim_size,
        dimensions=1,
    ),
    dict(
        name="hidden",
        n_neurons=n_hidden,
        dimensions=1,
    ),
    dict(
        name="output",
        n_neurons=n_output,
        dimensions=1,
    ),
]

# Define connections
conn_confs = [
    dict(
        pre="stimulus",
        post="visual_neurons",
        transform=1,
        synapse=0,
    ),
    # Encoding/Output connections
    dict(
        pre="stimulus",
        post="hidden_neurons",
        transform=gen_transform(
            "bipolar_gaussian_conv", ksize=_KERN_SIZE, nfilter=_N_FILTERS
        ),
        synapse=0,
    ),
    dict(
        pre="hidden_neurons",
        post="output_neurons",
        transform=gen_transform("othogonal_excitation"),
        learning_rule=learning_rule,
        synapse=2e-3,
    ),
    dict(
        pre="output_neurons",
        post="output_neurons",
        transform=gen_transform("circular_inhibition"),
        synapse=5e-3,
    ),
]

# Define learning rules
learning_confs = []
