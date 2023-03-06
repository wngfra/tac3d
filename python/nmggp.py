import os
import nengo
import numpy as np
from TouchDataset import TouchDataset

# Prepare dataset
datapath = os.path.join(os.path.dirname(__file__), "../data/touch.pkl")
dataset = TouchDataset(datapath, noise_scale=0.2, scope=(-1.0, 1.0))
X_train, y_train, X_test, y_test = dataset.split_set(ratio=0.5)
height, width = X_train[0].shape
n_dims = height * width

# Simulation parameters
dt = 1e-3
duration = (len(X_train) - 1) * dt
frames = int(duration // dt)
max_rates = 200
n_steps = 200

ens_params = dict(radius=1, intercepts=nengo.dists.Gaussian(0, 0.1))

conn_config = dict(
    learning_rule_type=nengo.PES(learning_rate=1e-4),
    synapse=dt,
)


def input_func(t):
    index = int(t / dt) // n_steps
    sample = X_train[index].flatten()
    return sample


# Create the Nengo model
with nengo.Network(label="MGGD") as model:
    stim = nengo.Node(input_func)
    inp = nengo.Ensemble(
        n_neurons=stim.size_out,
        dimensions=1,
        max_rates=np.ones(stim.size_out) * max_rates,
        neuron_type=nengo.PoissonSpiking(nengo.LIFRate()),
        **ens_params
    )
    hidden = nengo.Ensemble(n_neurons=100, dimensions=1, **ens_params)

    # Output layer
    sigma = nengo.Ensemble(n_neurons=16, dimensions=4, **ens_params)
    tau = nengo.Ensemble(n_neurons=16, dimensions=1, **ens_params)

    nengo.Connection(stim, inp.neurons, transform=1, synapse=None)
    conn_inp_hidden = nengo.Connection(
        inp.neurons,
        hidden.neurons,
        transform=np.random.normal(0, 0.25, size=(hidden.n_neurons, inp.n_neurons)),
        **conn_config
    )

    nengo.Connection(
        hidden.neurons,
        sigma,
        transform=np.random.normal(0, 0.25, size=(sigma.dimensions, hidden.n_neurons)),
        **conn_config
    )
    nengo.Connection(
        hidden.neurons,
        tau,
        transform=np.random.normal(0, 0.25, size=(tau.dimensions, hidden.n_neurons)),
        **conn_config
    )