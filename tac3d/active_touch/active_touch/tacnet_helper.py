from typing import Optional

import nengo
import numpy as np
from scipy.stats import multivariate_normal

image_height, image_width = 15, 15
image_size = image_height * image_width

# Constants
_MAX_RATE = 100
_AMP = 1.0 / _MAX_RATE
_RATE_TARGET = _MAX_RATE * _AMP

# Constatns for the convolutional layer
N_FILTERS = 18
KERN_SIZE = 7
STRIDES = (KERN_SIZE - 1, KERN_SIZE - 1)
STIM_SHAPE = (15, 15)

# Network params
dt = 1e-3
dim_states = 7
n_hidden_neurons = 100
n_coding_neurons = 36
learning_rate = 5e-9

# Default neuron parameters
default_neuron = nengo.AdaptiveLIF(amplitude=amp, tau_rc=0.05)
default_rates = nengo.dists.Choice([rate_target])
default_intercepts = nengo.dists.Choice([0])


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
            case "identity_excitation":
                if 0 in shape:
                    W = 1
                else:
                    W = np.ones(shape)
            case "bipolar_gaussian_conv":
                kernel = np.empty((KERN_SIZE, KERN_SIZE, 1, N_FILTERS))
                delta_phi = np.pi / (N_FILTERS + 1)
                for i in range(N_FILTERS):
                    kernel[:, :, 0, i] = sample_bipole_gaussian(
                        (KERN_SIZE, KERN_SIZE),
                        (KERN_SIZE // 2, KERN_SIZE // 2),
                        (2.0, 0.5),
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
                W /= np.max(np.abs(W))
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Gaussian(0.0, 0.2),
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


# Function to inhibit the error population after 15s
def inhib(t):
    return 2 if t > 20.0 else 0.0


delay = Delay(dim_states, timesteps=int(0.1 / dt))

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
    dict(name="state",
         n_neurons=2 * n_state_neurons, 
         dimensions=2
    ),
    dict(
        name="delta_state",
        n_neurons=n_state_neurons,
        dimensions=1,
    ),
    dict(
        name="stimulus",
        neuron=None,
        output=lambda t: X_train[int(t / presentation_time) % len(X_train)].ravel(),
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
        name="wta",
        n_neurons=n_wta_neurons,
        dimensions=1,
    ),
]

# Define connections
conn_confs = [
    dict(
        pre="state_node",
        post="delay_node",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_node",
        post="state",
        dim_out=0,
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delay_node",
        post="state",
        dim_out=1,
        transform=gen_transform("identity_excitation"),
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
        transform=gen_transform("identity_excitation"),
        synapse=0.001,
    ),
    dict(
        pre="stim_neurons",
        post="hidden_neurons",
        transform=gen_transform("bipolar_gaussian_conv"),
        synapse=1e-3,
    ),
    dict(
        pre="hidden_neurons",
        post="wta_neurons",
        transform=gen_transform(),
        learning_rule=SynapticSampling(),
        synapse=0,
    ),
    dict(
        pre="wta_neurons",
        post="wta_neurons",
        transform=gen_transform("circular_inhibition"),
        synapse=0.01,
    ),
    
]

# Define learning rules
learning_confs = []