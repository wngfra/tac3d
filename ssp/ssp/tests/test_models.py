import numpy as np

from ssp.models import MLP, Linear


x_dim = 128
h_dim = 256
y_dim = 128

n_steps = 100
n_samples = 10
threshold = 1e-1

xs = np.random.random((n_samples, x_dim))
ys = np.random.random((n_samples, y_dim))


def test_mlp():
    model = MLP(x_dim=x_dim, h_dims=h_dim, y_dim=y_dim)
    model.train(xs, ys, n_steps)

    # check that loss during training decreases monotonically
    assert np.all(np.diff(model.costs) <= 0)
    # check that final loss is small
    assert model.costs[-1] < threshold


def test_linear():
    model = Linear(x_dim=x_dim, y_dim=y_dim)
    model.train(xs, ys, n_steps)

    # check that loss during training decreases monotonically
    assert np.all(np.diff(model.costs) <= 0)
    # check that final loss is small
    assert model.costs[-1] < threshold


def test_eigvals():
    model = Linear(x_dim=x_dim, y_dim=y_dim)
    model.train(xs, ys, n_steps)

    eigvals, eigvecs = model.eigdata
    W = model.weights

    # check that matrix has expected effect given eigenvalues
    for i in range(x_dim):
        val_times_vec = eigvals[i] * eigvecs[:, i]
        mod_times_vec = np.dot(W, eigvecs[:, i])
        assert np.allclose(val_times_vec, mod_times_vec)

        # check that sorting of eigenvalues is based on magnitude
        if i < x_dim - 1:
            assert np.abs(eigvals[i]) >= np.abs(eigvals[i + 1])
