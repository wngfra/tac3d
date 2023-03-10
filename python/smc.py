import os
from typing import Optional
import nengo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from TouchDataset import TouchDataset
from nengo_extras.graphviz import net_diagram

_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")
_IMAGE_DIM: Optional[int] = None

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1.0, 1.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5, shuffle=True)
_IMAGE_DIM = X_train[0].shape[0]

# Simulation parameters
dt = 1e-3
max_rates = 200
n_mean = 10
n_encoding = 4
n_output = 36
n_steps = 500
sigma = 10

ens_params = dict(radius=1, intercepts=nengo.dists.Gaussian(0.1, 0.1))
conn_config = dict(
    learning_rule_type=nengo.BCM(1e-3),
    synapse=0.01,
)


def gen_gaussian_weights(mean, cov, size):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    pdf = rv.pdf(pos)
    pdf /= np.sum(pdf)
    sample = np.random.choice(
        size * size, p=pdf.ravel(), size=size * size // (n_encoding * n_encoding) + 1
    )
    M = np.zeros(size * size)
    M[sample] = 1
    norm_coeff = np.sum(M[M > 0])
    return M / norm_coeff


def input_func(t):
    index = int(t / (dt * n_steps))
    return X_train[index].ravel()


def motion_func(t):
    index = np.random.randint(0, n_output, size=1)
    M = np.zeros(n_output, dtype=int)
    M[index] = 1
    return M


def output_func(t):
    index = int(t / (dt * n_steps))
    theta = y_train[index]
    n = np.floor(theta / np.pi)
    return (theta - n * np.pi) / np.pi * 180


# Create the Nengo model
with nengo.Network(label="smc") as model:
    truth = nengo.Node(output_func)

    # Create input layer
    stim = nengo.Node(input_func)
    inp = nengo.Ensemble(
        n_neurons=stim.size_out,
        dimensions=stim.size_out,
        max_rates=nengo.dists.Choice([max_rates]),
        neuron_type=nengo.AdaptiveLIF(),
        label="inp",
        **ens_params
    )
    conn_stim2inp = nengo.Connection(
        stim, inp.neurons, transform=1, synapse=None, label="stim2inp"
    )

    # Create (encoding) layer
    encoding = nengo.Ensemble(
        n_neurons=n_encoding * n_encoding,
        dimensions=n_encoding * n_encoding,
        radius=1,
        max_rates=nengo.dists.Choice([max_rates]),
        intercepts=nengo.dists.Choice([0, 0.1]),
        neuron_type=nengo.AdaptiveLIF(),
        label="encoding",
    )
    weights = np.empty((encoding.dimensions, inp.dimensions))
    stride = (_IMAGE_DIM - n_encoding) // (n_encoding + 1)
    for i in range(n_encoding):
        for j in range(n_encoding):
            weight = gen_gaussian_weights(
                [(i + 1) * (stride + 1) - 1, (j + 1) * (stride + 1) - 1],
                [[sigma * sigma, 0], [0, sigma * sigma]],
                _IMAGE_DIM,
            )
            weights[i * n_encoding + j, :] = weight.ravel()

    conn_inp2encoding = nengo.Connection(
        inp.neurons,
        encoding.neurons,
        transform=np.random.randn(encoding.n_neurons, inp.n_neurons),
        synapse=0.01,
        learning_rule_type=nengo.PES(1e-5),
        label="inp2encoding",
    )

    # Create theta layer
    theta = nengo.Ensemble(
        n_neurons=10,
        dimensions=1,
        radius=10,
        intercepts=nengo.dists.Choice([0.1, 0.1]),
        encoders=nengo.dists.Uniform(-1, 1),
        neuron_type=nengo.AdaptiveLIF(),
        label="theta",
    )
    """
    conn_encoding2theta = nengo.Connection(
        encoding,
        theta.neurons,
        transform=nengo.dists.Gaussian(0.1, 0.5),
        synapse=0.01,
        #learning_rule_type=nengo.BCM(1e-5),
        label="encoding2theta",
    )
    """
    """ Self Inhibition with circular topology embedded
    fold2pi = theta.n_neurons // 2
    inh_weights = np.zeros((theta.n_neurons, theta.n_neurons))
    for i in range(theta.n_neurons):
        weight = np.concatenate([np.arange(fold2pi), np.flip(np.arange(fold2pi))])
        inh_weights[i, :] = -np.roll(weight, i)
    conn_theta2theta = nengo.Connection(
        theta.neurons, theta.neurons, transform=inh_weights, synapse=0.1
    )
    """

    # Create motion input
    motion_stim = nengo.Node(motion_func)
    dtheta = nengo.Ensemble(n_neurons=n_output, dimensions=1)
    conn_mstim2dtheta = nengo.Connection(motion_stim, dtheta.neurons)
    theta_new = nengo.Ensemble(n_neurons=theta.n_neurons, dimensions=1)
    conn_dtheta2tn = nengo.Connection(
        dtheta, theta_new.neurons, transform=nengo.dists.Gaussian(0, 0.01)
    )
    conn_theta2tn = nengo.Connection(
        theta, theta_new, transform=1, synapse=None, label="theta2theta_new"
    )

    # Create output layer
    out = nengo.Ensemble(
        n_neurons=inp.n_neurons,
        dimensions=inp.dimensions,
        neuron_type=nengo.SpikingRectifiedLinear(),
        label="out",
    )
    conn_encoding2out = nengo.Connection(
        encoding.neurons,
        out.neurons,
        transform=np.random.randn(out.n_neurons, encoding.n_neurons),
        learning_rule_type=nengo.PES(1e-5),
        label="encoding2out",
    )
    """
    conn_tn2out = nengo.Connection(
        theta_new, out.neurons, transform=nengo.dists.Gaussian(0, 0.1)
    )
    """

    # Create error ensemble
    error = nengo.Ensemble(
        n_neurons=inp.n_neurons, dimensions=inp.n_neurons, label="error"
    )
    conn_inp2error = nengo.Connection(inp, error, transform=-1, label="inp2error")
    conn_out2error = nengo.Connection(out, error, transform=1, label="out2error")

    # Connect learning rules
    nengo.Connection(error, conn_encoding2out.learning_rule)
    nengo.Connection(encoding, conn_inp2encoding.learning_rule)

    # Create probes
    inp_p = nengo.Probe(inp, synapse=0.01)
    theta_p = nengo.Probe(theta, synapse=0.01)
    weights_p = nengo.Probe(conn_encoding2out, "weights", synapse=0.01)


"""Run in command line mode
"""
with nengo.Simulator(model) as sim:
    sim.run(10.0)

plt.plot(sim.trange(), sim.data[weights_p][..., n_encoding])
plt.show()
