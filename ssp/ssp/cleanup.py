import nengo_spa as spa
import numpy as np
import tensorflow as tf

from ssp.models import MLP, Model


def generate_cleanup_data(
    vocab, objs, low, high, sigma=0, return_coords=False, n_samples=10240
):
    """Generate the data for training or testing a cleanup.

    If ``return_coords`` is False (default), then the cleanup takes the form:

        f(X^x Y^y + \eta) = X^x Y^y

    Otherwise, it takes the form:

        f(X^x Y^y + \eta) = [x, y]

    In either case, \eta, is the superposition of other objects, unbound with
    the object being cleaned up. 

    Notes
    -----
    The random number generator for the vocab's pointer_gen is used.

    Parameters
    ----------
    vocab : nengo_spa.Vocabulary
        The SPA vocabulary to use for objects and base vectors.
        Assumed to be consistent with the maps vocabulary.
    objs : array of str
        The list of objects that the cleanup should be trained on.
        The cleanup should still generalize beyond these objects.
    low : float or array of float
        Lower-bound(s) defining the grid region of each coordinate.
    high : float or array of float
        Upper-bound(s) defining the grid region of each coordinate.
    sigma : float
        Standard deviation of noise added to the inputs.
        Automatically scaled by the inverse square root of dimensionality.
    return_coords : bool
        Whether the outputs should be two-dimensional [x, y] (True) or a
        semantic pointer of the form X^x Y^y (False).
    n_samples : int 
        Number of samples to generate.
    """

    rng = vocab.pointer_gen.rng
    pos = rng.uniform(low=low, high=high, size=(n_samples, len(objs), 2))

    # TODO: probably overkill to be using particular objects and
    # random targets. probably sufficient to use arbitrary objects
    # and the same target each time, by symmetry, and because the
    # inverse for unitary vectors is exact
    targets = rng.randint(0, len(objs), size=n_samples)

    sigma /= np.sqrt(vocab.dimensions)
    noise = rng.normal(loc=0, scale=sigma, size=(n_samples, vocab.dimensions))

    x = np.empty((n_samples, vocab.dimensions), dtype=np.float64)
    if return_coords:
        y = np.empty((n_samples, 2), dtype=np.float64)
    else:
        y = np.empty_like(x)
    for i in range(n_samples):
        # TODO: this is redundant with maps.py and assumes it is using the
        # same encoding with the same base vectors and vocabulary
        x[i, :] = (
            noise[i]
            + (
                np.sum(
                    [
                        vocab[objs[j]]
                        * vocab["X"] ** pos[i, j, 0]
                        * vocab["Y"] ** pos[i, j, 1]
                        for j in range(len(objs))
                    ]
                )
                * ~vocab[objs[targets[i]]]
            ).v
        )
        if return_coords:
            y[i, :] = pos[i, targets[i], :]
        else:
            y[i, :] = (
                vocab["X"] ** pos[i, targets[i], 0]
                * vocab["Y"] ** pos[i, targets[i], 1]
            ).v
    return x, y


class XYLoss:
    """Penalize norm of error projected onto the gradient of X^x Y^y.
    
    This leverages the fact that:
        
        (ssp * (I + eps*X)).unitary()
        (ssp * (I + eps*Y)).unitary()

    approximates eps-small displacements of X in ssp. And similarly for Y.
    This is because the gradient of X^x Y^y is (X, Y) with respect to (x, y)
    in the Fourier domain.

    We project onto this gradient to find the eps for X and the eps for Y,
    which therefore penalizes the magnitude of the error with respect to the
    displacement in (x, y) space along the surface of X^x Y^y.
    """

    def __init__(self, X, Y):
        dim = len(X)
        self.rfftscale = 1 / (dim // 2 + 1)  # to make rfft unitary
        self.dftX = tf.signal.rfft(X) * rfftscale
        self.dftY = tf.signal.rfft(Y) * rfftscale

    def __call__(self, pred, ys):
        pred = tf.math.l2_normalize(pred, axis=1)
        dftA = tf.signal.rfft(pred) * self.rfftscale
        dftBinv = tf.math.conj(
            tf.signal.rfft(ys) * self.rfftscale
        )  # since ys is unitary
        dftE = (
            tf.math.multiply(
                tf.cast(dftA, tf.complex64), tf.cast(dftBinv, tf.complex64)
            )
            - 1
        )
        # https://github.com/tensorflow/tensorflow/pull/39529
        return -tf.cast(
            tf.reduce_mean(
                tf.abs(tf.tensordot(dftE, self.dftX, axes=1))
                + tf.abs(tf.tensordot(dftE, self.dftY, axes=1))
            ),
            tf.float64,
        )


class Cleanup:
    """Clean up the position for an object in a spatial semantic pointer."""

    def __init__(self, model, vocab):
        if not isinstance(model, Model):
            # assume model is being specified as the middle argument to MLP
            model = MLP(vocab.dimensions, model, vocab.dimensions)

        self.model = model
        self.vocab = vocab

    def train(self, objs, low, high, sigma=0.1, n_steps=5000):
        # note: assumes scale=1, otherwise can be multiplied with low/high
        x, y = generate_cleanup_data(self.vocab, objs, low, high, sigma=sigma)

        self.model.train(x, y, n_steps=n_steps)

    def __call__(self, sp):
        output = self.model(sp.v)
        # if trained with return_coords=True can instead do:
        # return ssp_map.encode_point(*output, name=None)
        return spa.SemanticPointer(output).unitary()
