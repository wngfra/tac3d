import os
from typing import Optional
import nengo
import numpy as np
from TouchDataset import TouchDataset

_DATAPATH = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")
_IMAGE_DIM: Optional[int] = None

# Prepare dataset
dataset = TouchDataset(_DATAPATH, noise_scale=0.1, scope=(-1.0, 1.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5)
_IMAGE_DIM = X_train[0].shape[0]

# Simulation parameters
dt = 1e-3
max_rates = 100
n_steps = 100

ens_params = dict(radius=1, intercepts=nengo.dists.Gaussian(0, 0.1))
conn_config = dict(
    learning_rule_type=nengo.BCM(learning_rate=1e-9),
    synapse=dt,
)


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
        dimensions=stim.size_out,
        max_rates=np.ones(stim.size_out) * max_rates,
        neuron_type=nengo.PoissonSpiking(nengo.LIFRate()),
        **ens_params
    )
    nengo.Connection(stim, inp.neurons, transform=1, synapse=None)
