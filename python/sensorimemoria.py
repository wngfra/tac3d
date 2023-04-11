# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import nengo
import numpy as np
from sklearn.datasets import make_spd_matrix

n_space = 5 * 5
stim_dim = (10, 10)
stim_size = np.prod(stim_dim)
n_samples = 200
sensor_dim = 10


def stim_func(t):
    cov = make_spd_matrix(2)
    x, y = np.random.multivariate_normal(np.random.rand(2), cov, size=n_samples).T
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    img = np.zeros(stim_dim)
    iy = np.floor(y * stim_dim[0]).astype(int)
    ix = np.floor(x * stim_dim[1]).astype(int)
    ix = np.clip(ix, 0, stim_dim[1] - 1)
    iy = np.clip(iy, 0, stim_dim[0] - 1)
    for i in range(n_samples):
        img[iy[i], ix[i]] += 1

    return img.ravel()


def motion_func(t):
    return np.sin(t * np.pi * 2), np.cos(t * np.pi * 2)


with nengo.Network("sensori-memoria") as model:
    sin = nengo.Node(lambda t: np.sin(t * np.pi * 20))
    oscillator = nengo.Ensemble(2, dimensions=1, radius=1)
    memory_cells = nengo.Ensemble(n_space * 8, dimensions=n_space * 2, radius=1)

    nengo.Connection(sin, oscillator, synapse=None)
    for i in range(n_space):
        nengo.Connection(oscillator, memory_cells[i], transform=1, synapse=0.01)

    stim = nengo.Node(output=stim_func)
    inp = nengo.Ensemble(stim_size, dimensions=sensor_dim, radius=1)
    nengo.Connection(stim, inp.neurons, synapse=None)
    nengo.Connection(inp, memory_cells, transform=np.random.rand(n_space*2, inp.size_out), synapse=0.01)

    state_inp = nengo.Node(output=motion_func)
    state = nengo.Ensemble(16, dimensions=2, radius=1)
    nengo.Connection(state_inp, state, synapse=None)