from typing import Optional

import nengo
import numpy as np

image_height, image_width = 15, 15
image_size = image_height * image_width

# Constants
_MAX_RATE = 100
_AMP = 1.0 / _MAX_RATE
_RATE_TARGET = _MAX_RATE * _AMP

# Network params
dt = 1e-3
dim_states = 7
n_hidden_neurons = 100
n_coding_neurons = 36
default_neuron = nengo.AdaptiveLIF()
default_rates = nengo.dists.Choice([_MAX_RATE])
default_intercepts = nengo.dists.Choice([0, 0.1])
learning_rate = 1e-3


def normalize(x, dtype=np.uint8):
    iinfo = np.iinfo(dtype)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min()) * (iinfo.max - 1)
    return x.astype(dtype)


def log(node, x):
    node.get_logger().warn("DEBUG LOG: {}".format(x))


def gen_transform(pattern="random", weights=None):
    W: Optional[np.ndarray] = None

    def inner(shape, weights=weights):
        """_summary_

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            _type_: _description_
        """
        match pattern:
            case "identity_excitation":
                W = 1
            case "identity_inhibition":
                W = -1
            case "exclusive_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = -np.ones((shape[0], shape[0])) + 2 * np.eye(shape[0])
            case "cyclic_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                xmax = shape[1] // 2
                x = np.abs(np.arange(shape[0]) - xmax)
                W = np.empty((shape[0], shape[0]))
                for i in range(shape[0]):
                    W[i, :] = np.roll(x, i)
                W = -W
                W[W == 0] = 1
                W *= 0.2
            case "custom":
                if weights is None:
                    raise ValueError("No weights provided!")
                W = weights
            case "zero":
                W = 0
            case _:
                W = nengo.Dense(
                    shape,
                    init=nengo.dists.Gaussian(0, 0.1),
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

layer_confs = [
    dict(
        name="inhibit",
        neuron=None,
        output=inhib,
    ),
    dict(
        name="state",
        neuron=None,
        size_out=dim_states,
    ),
    dict(
        name="state_ens",
        n_neurons=n_coding_neurons,
        dimensions=dim_states,
    ),
    dict(
        name="delayed_state",
        neuron=None,
        output=delay.step,
        size_in=dim_states,
        size_out=dim_states,
    ),
    dict(
        name="delayed_state_ens",
        n_neurons=n_coding_neurons,
        dimensions=dim_states,
        neuron=nengo.LIF(),
        radius=np.pi,
    ),
    dict(
        name="delta_state_ens",
        n_neurons=n_coding_neurons,
        dimensions=dim_states,
        neuron=nengo.LIF(),
        radius=np.pi,
    ),
    dict(
        name="input",
        neuron=None,
    ),
    dict(
        name="stim_ens",
        n_neurons=image_size,
        dimensions=15,
        radius=1,
        encoders=nengo.dists.Gaussian(0, 0.5),
        on_chip=False,
    ),
    dict(
        name="hidden_ens",
        n_neurons=n_hidden_neurons,
        dimensions=15,
        radius=1,
    ),
    dict(
        name="output_ens",
        n_neurons=image_size,
        dimensions=2,
        radius=1,
    ),
    dict(
        name="coding_ens",
        n_neurons=n_coding_neurons,
        dimensions=dim_states,
        radius=np.pi,
    ),
    dict(
        name="error_ens",
        n_neurons=n_coding_neurons,
        dimensions=dim_states,
        radius=np.pi,
    ),
    dict(
        name="reconstruction_error_ens",
        n_neurons=image_size,
        dimensions=2,
        radius=1,
    ),
]

conn_confs = [
    dict(
        pre="state",
        post="state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state",
        post="delayed_state",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delayed_state",
        post="delayed_state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_ens",
        post="delta_state_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="delayed_state_ens",
        post="delta_state_ens",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="input",
        post="stim_ens_neurons",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="stim_ens",
        post="hidden_ens",
        synapse=0.01,
        solver=nengo.solvers.LstsqL2(weights=True),
    ),
    dict(
        pre="hidden_ens_neurons",
        post="output_ens_neurons",
        synapse=0.01,
        learning_rule=nengo.PES(learning_rate=learning_rate),
    ),
    dict(
        pre="hidden_ens_neurons",
        post="coding_ens_neurons",
        learning_rule=nengo.PES(learning_rate=learning_rate),
    ),
    dict(
        pre="coding_ens",
        post="error_ens",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="state_ens",
        post="error_ens",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="stim_ens_neurons",
        post="reconstruction_error_ens_neurons",
        transform=gen_transform("identity_excitation"),
    ),
    dict(
        pre="output_ens_neurons",
        post="reconstruction_error_ens_neurons",
        transform=gen_transform("identity_inhibition"),
    ),
    dict(
        pre="inhibit",
        post="error_ens_neurons",
        transform=gen_transform("custom", weights=np.ones((n_coding_neurons, 1)) * -3),
        synapse=0.01,
    ),
]

learning_confs = [
    # dict(name="learning_coding",pre="error_ens",post="hidden_ens_neurons2coding_ens_neurons",),
]
