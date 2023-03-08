import os
from typing import Optional
import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from TouchDataset import TouchDataset
from nengo_extras.plot_spikes import (
    plot_spikes,
)

_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")
_IMAGE_DIM: Optional[int] = None

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1.0, 1.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
_IMAGE_DIM = X_train[0].shape[0]

# Simulation parameters
dt = 1e-3
max_rates = 200
n_hidden = 4
n_output = 36
n_steps = 200
sigma = 1

ens_params = dict(radius=1, intercepts=nengo.dists.Gaussian(0, 0.1))
conn_config = dict(
    learning_rule_type=nengo.BCM(1e-5),
    synapse=dt,
)


def gen_gaussian_weights(mean, cov, size):
    x, y = np.mgrid[0:size:1, 0:size:1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    return rv.pdf(pos)


def gen_mean_weights(direction: int, size: int) -> np.ndarray:
    """Return the weights to compute mean for 2D input of size x size.

    Args:
        direction (int): 0 for y-axis and 1 for x-axis.
        size (int): Size of the weight matrix.

    Returns:
        np.ndarray: Weight matrix of dimension size x size.
    """
    x = np.linspace(-size / 2, size / 2, size)
    M = np.repeat(x[:, np.newaxis], size, axis=1)
    M = np.abs(M)
    return M if direction == 0 else M.T


def input_func(t):
    index = int(t / (dt * n_steps))
    return X_train[index].ravel()


def output_func(t):
    index = int(t / (dt * n_steps))
    theta = y_train[index]
    n = np.floor(theta / np.pi)
    return (theta - n * np.pi) / np.pi * 180


# Create the Nengo model
with nengo.Network(label="mvgg") as model:
    truth = nengo.Node(output_func)

    # Create input layer
    stim = nengo.Node(input_func)
    inp = nengo.Ensemble(
        n_neurons=stim.size_out,
        dimensions=1,
        max_rates=np.ones(stim.size_out) * max_rates,
        neuron_type=nengo.PoissonSpiking(nengo.LIFRate()),
        label="inp",
        **ens_params
    )
    conn_stim2inp = nengo.Connection(stim, inp.neurons, transform=1, synapse=None)

    # Create mean layer
    mean = nengo.Ensemble(
        n_neurons=5,
        dimensions=2,
    )
    conn_inp2meany = nengo.Connection(
        inp, mean[0], transform=gen_mean_weights(0, inp.size_out).ravel()
    )
    conn_inp2meanx = nengo.Connection(
        inp, mean[1], transform=gen_mean_weights(1, inp.size_out).ravel()
    )

    # Create hidden layer
    hidden = nengo.Ensemble(
        n_neurons=n_hidden * n_hidden,
        dimensions=n_hidden * n_hidden,
        neuron_type=nengo.LIF(),
        label="hidden",
        **ens_params
    )
    weights = np.empty((hidden.dimensions, inp.n_neurons))
    stride = (_IMAGE_DIM - n_hidden) // (n_hidden + 1)
    for i in range(n_hidden):
        for j in range(n_hidden):
            weight = gen_gaussian_weights(
                [(i + 1) * (stride + 1) - 1, (j + 1) * (stride + 1) - 1],
                [[sigma * sigma, 0], [0, sigma * sigma]],
                _IMAGE_DIM,
            )
            weights[i * n_hidden + j, :] = weight.ravel()
    conn_inp2hidden = nengo.Connection(inp.neurons, hidden, transform=weights)

    # Create WTA layer
    wta = nengo.Ensemble(
        n_neurons=n_output,
        dimensions=n_output,
        radius=2,
        intercepts=nengo.dists.Gaussian(0, 0.1),
        label="wta",
    )
    conn_hidden2wta = nengo.Connection(
        hidden,
        wta,
        solver=nengo.solvers.LstsqL2(weights=True),
        transform=np.random.random((wta.dimensions, hidden.dimensions)),
        label="hidden2out",
        **conn_config
    )
    conn_wta2wta = nengo.Connection(
        wta,
        wta,
        transform=-(np.ones((wta.dimensions, wta.dimensions)) - np.eye(wta.dimensions)),
    )

    # Create output layer
    out = nengo.Ensemble(
        n_neurons=n_output, dimensions=n_output, label="out", **ens_params
    )
    conn_wta2out = nengo.Connection(
        wta,
        out,
        transform=np.eye(n_output) - np.ones((n_output, n_output)),
    )

    # Create probes
    inp_p = nengo.Probe(inp, synapse=0.01)
    wta_p = nengo.Probe(wta, synapse=0.01)
    weights_p = nengo.Probe(conn_hidden2wta, "weights", synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(2.0)

plt.plot(sim.trange(), sim.data[weights_p][..., n_hidden])
plt.show()
