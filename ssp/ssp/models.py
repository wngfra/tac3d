from nengo.utils.numpy import array, is_array
from nengo.utils.progress import Progress, ProgressTracker
import nengo_spa as spa

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")


class Model:
    """Implements features used by all model types, assuming by default that
    the goal of training is to minimize mean squared error when mapping from
    xs to ys.
    """

    def __init__(self):
        self.opt = tf.keras.optimizers.Adam()
        self.cost = tf.keras.losses.MeanSquaredError()
        self.costs = []

    def __call__(self, xs):
        if isinstance(xs, spa.SemanticPointer):
            return spa.SemanticPointer(
                data=self(xs.v),
                vocab=xs.vocab,
                algebra=xs.algebra,
                name="%s(%s)" % (type(self).__name__, xs.name),
            )
        if is_array(xs):
            xs = np.asarray(xs)
            if xs.ndim == 1:
                # make and return a single inference
                return np.asarray(self(np.expand_dims(xs, axis=0))).squeeze(axis=0)
        return self.model(xs)

    def train(self, xs, ys, n_steps=1, self_train=False):
        """Train a model to minimize MSE on x, y data"""
        inps = xs

        with ProgressTracker(
            True, Progress("Training", "Trained", n_steps)
        ) as progress_bar:
            for step in range(n_steps):
                with tf.GradientTape() as tape:
                    pred = self(inps)
                    cost = self.cost(pred, ys)

                    # use predictions at t as input at t+1 (needs sequential data)
                    if self_train and step % 10 == 0:
                        inps = np.roll(pred, 1, axis=1)
                        inps[0, :] = xs[0, :]  # since init is never predicted
                    else:
                        inps = xs

                grad = tape.gradient(cost, self.model.trainable_weights)
                self.opt.apply_gradients(zip(grad, self.model.trainable_weights))
                self.costs.append(cost)

                progress_str = "(cost=%f)" % cost
                progress_bar.total_progress.name_during = "Training %s" % progress_str
                if step == n_steps - 1:
                    progress_bar.total_progress.name_after = "Trained %s" % progress_str
                progress_bar.total_progress.step()


class Linear(Model):
    """Learns a linear transformation via gradient descent to minimize mean
    squared error on example target transformation outputs.

    Parameters:
    ----------
    x_dim: int
        The input dimensionality of the model.
    y_dim: int
        The output dimensionality of the model.
    """

    def __init__(self, x_dim, y_dim):

        super().__init__()

        self.model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(y_dim, use_bias=False, input_shape=(x_dim,))]
        )

    @property
    def weights(self):
        return self.model.layers[0].get_weights()[0]

    @property
    def eigdata(self):
        eigvals, eigvecs = np.linalg.eig(self.weights)
        indices = np.abs(eigvals).argsort()[::-1]
        eigvals = eigvals[indices]
        eigvecs = eigvecs[:, indices]

        return eigvals, eigvecs


class MLP(Model):
    """A multilayer perception with one (or more) hidden layer(s) that is
    trained via gradient descent to minimize mean squared error on target
    model outputs.

    Parameters:
    ----------
    x_dim: int
        The input dimensionality of the model.
    h_dims: int or array_like
        The hidden layer dimensionality of the model.
        Can alternatively be an array of dimensionalities, one
        per layer.
    y_dim: int
        The output dimensionality of the model.
    """

    def __init__(self, x_dim, h_dims, y_dim):

        super().__init__()
        h_dims = array(h_dims, min_dims=1)

        self.model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(h_dims[0], "relu", input_shape=(x_dim,)),]
            + [tf.keras.layers.Dense(h_dims[i], "relu") for i in range(1, len(h_dims))]
            + [tf.keras.layers.Dense(y_dim),]
        )
