import numpy as np

from nengo_spa.algebras.hrr_algebra import HrrAlgebra, HrrProperties
from nengo_spa.vector_generation import UnitaryVectors

from functools import reduce


class BaseVectors(UnitaryVectors):
    """Generator for unitary and nondegenerate SSPs.

    Avoids Fourier coefficients that are directly on the +/- pi boundary
    since these are degenerate when raised to fractional exponents.

    Parameters
    ----------
    d : int
        Dimensionality of returned vectors.
    algebra : AbstractAlgebra
        Algebra that defines what vectors are unitary and nondegenerate.
        Defaults to `.HrrAlgebra`.
    rng : numpy.random.RandomState, optional
        The random number generator to use to create new vectors.
    """

    def __init__(self, d, algebra=None, rng=None):
        if algebra is None:
            algebra = HrrAlgebra()
        super().__init__(d=d, algebra=algebra, rng=rng)

    def __next__(self):
        return self.algebra.create_vector(self.d, [HrrProperties.POSITIVE, HrrProperties.UNITARY])


def to_sum(sp_list):
    """Compute new semantic pointer that sums a list of semantic pointers."""
    return reduce(lambda sp1, sp2: sp1 + sp2, sp_list)


def normalize(v):
    """Normalize the vector to unit length"""
    norm = np.linalg.norm(v)
    return (v * float(1 / norm)) if norm > 0 else v
