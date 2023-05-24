# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
import matplotlib.pyplot as plt
from custom_learning_rules import SynapticSampling
from nengo_extras.plot_spikes import plot_spikes

N = 10
M = np.log2(N).astype(int) + 1

dt = 1e-3
presentation_time = 0.3
n_neurons = 100
default_rate = nengo.dists.Choice([100])
duration = 5


def dec2bin(x: int):
    """Convert decimal to binary array..

    Args:
        x (int): Decimal number.

    Returns:
        list: Binary array.
    """
    bin_arr = [int(i) for i in bin(x)[2:]]
    bin_arr = [0] * (M - len(bin_arr)) + bin_arr
    return np.array(bin_arr)


def inp_func(t):
    x = int(t / presentation_time) % N
    x_arr = dec2bin(x)
    f = x_arr * np.random.normal(np.sqrt(x_arr), x_arr, M)
    return f


with nengo.Network(label="max-entropy", seed=1) as model:
    decimal = nengo.Node(lambda t: int(t / presentation_time) % N)
    inp = nengo.Node(output=inp_func)
    ens = nengo.Ensemble(n_neurons, M, max_rates=default_rate)
    nengo.Connection(inp, ens)

    out = nengo.networks.EnsembleArray(
        16,
        n_ensembles=N,
        max_rates=default_rate,
    )
    conn = nengo.Connection(ens.neurons, out.input, transform=np.zeros((N, n_neurons)))
    # conn.learning_rule_type = SynapticSampling()
    nengo.Connection(
        out.output,
        out.input,
        transform=3 * (-np.ones((N, N)) + np.eye(N)),
        synapse=0.01,
    )

    probe_conn = nengo.Probe(
        conn,
        "weights",
        synapse=0.01,
    )
    probe_output = nengo.Probe(out.output, synapse=0.01)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(duration)

plt.figure(figsize=(5, 10))
# Find weight row with max variance
neuron = np.argmax(np.mean(np.var(sim.data[probe_conn], axis=0), axis=1))
plt.plot(sim.trange(), sim.data[probe_conn][:, neuron, :])
plt.xlabel("time (s)")
plt.ylabel("weights")

fig, ax = plt.subplots(1, 1)
plot_spikes(sim.trange(), sim.data[probe_output], ax=ax)
ax.set_title("Output Spikes")
plt.show()
