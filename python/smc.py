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
max_rate = 150
amp = 1.0 / max_rate
rate_target = max_rate * amp  # must be in amplitude scaled units

n_hidden = 16
n_output = 36
<<<<<<< HEAD
presentation_time = 0.2
=======
n_steps = 500
>>>>>>> fa858d315186496ba21b71d20b35e6a5f38bba54
sigma = 10

default_neuron = nengo.AdaptiveLIF(amplitude=amp)
default_intercepts = nengo.dists.Gaussian(0, 0.1)

layer_confs = [
    dict(
        name="input_layer",
        n_neurons=_IMAGE_DIM * _IMAGE_DIM,
        max_rates=nengo.dists.Choice([rate_target]),
        on_chip=False,
    ),
    dict(
        name="hidden_layer",
        n_neurons=n_hidden,
    ),
    dict(
        name="output_layer",
        n_neuron=_IMAGE_DIM * _IMAGE_DIM,
    ),
]

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
        size * size, p=pdf.ravel(), size=size * size // (n_hidden * n_hidden) + 1
    )
    M = np.zeros(size * size)
    M[sample] = 1
    norm_coeff = np.sum(M[M > 0])
    return M / norm_coeff


def motion_func(t):
    index = np.random.randint(0, n_output, size=1)
    M = np.zeros(n_output, dtype=int)
    M[index] = 1
    return M


# Create the Nengo model
with nengo.Network(label="smc") as net:
    truth = nengo.Node(
        lambda t: y_train[int(t / presentation_time)]
        - np.floor(y_train[int(t / presentation_time)] / np.pi) * np.pi
    )
    stim = nengo.Node(lambda t: X_train[int(t / presentation_time)].ravel())

    connections = []
    transforms = []
    layer_probes = []
    shape_in = nengo.transforms.ChannelShape((_IMAGE_DIM * _IMAGE_DIM,))
    x = stim

    # Create layers
    for k, layer_conf in enumerate(layer_confs):
        layer_conf = dict(layer_conf)  # copy, so we don't modify the original
        name = layer_conf.pop("name")
        intercepts = layer_conf.pop("intercepts", default_intercepts)
        max_rates = layer_conf.pop("max_rates", nengo.dists.Choice([rate_target]))
        neuron_type = layer_conf.pop("neuron", default_neuron)
        on_chip = layer_conf.pop("on_chip", True)
        block = layer_conf.pop("block", None)

        # Create layer transform
        if "filters" in layer_confs:
            # Convolutional layer
            pass
        else:
            # Dense layer
            n_neurons = layer_conf.pop("n_neurons")
            shape_out = nengo.transforms.ChannelShape((n_neurons,))
            transform = nengo.Dense(
                (shape_out.size, shape_in.size),
                init=nengo.dists.Gaussian(0.5, 0.1),
            )

            loc = "chip" if on_chip else "host"
            n_weights = np.prod(transform.shape)

        assert len(layer_conf) == 0, "Unused fields in {}: {}".format(
            [name], list(layer_conf)
        )

        if neuron_type is None:
            assert not on_chip, "Nodes can only be run off-chip"
            y = nengo.Node(size_in=shape_out.size, label=name)
        else:
            ens = nengo.Ensemble(shape_out.size, 1, neuron_type=neuron_type, label=name)
            y = ens.neurons

            # Add a probe so we can measure individual layer rates
            probe = nengo.Probe(y, synapse=None, label="%s_p" % name)
            layer_probes.append(probe)

        conn = nengo.Connection(x, y, transform=transform)

        transforms.append(transform)
        connections.append(conn)
        x = y
        shape_in = shape_out

    # Create theta layer
    theta = nengo.Ensemble(
        n_neurons=10,
        dimensions=1,
        radius=10,
        intercepts=nengo.dists.Choice([0.1, 0.1]),
        encoders=nengo.dists.Uniform(-1, 1),
        neuron_type=nengo.SpikingRectifiedLinear(),
        label="theta",
    )
    """
    conn_encoding2theta = nengo.Connection(
        encoding,
        theta.neurons,
        transform=nengo.dists.Gaussian(0.1, 0.5),
        synapse=0.01,
        learning_rule_type=MirroredSTDP(),
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
        encoding,
        out,
        function=lambda x: np.random.random(out.dimensions),
        learning_rule_type=nengo.PES(1e-3),
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
with nengo.Simulator(net) as sim:
    sim.run(5.0)

plt.plot(sim.trange(), sim.data[weights_p][..., n_hidden])
plt.show()
