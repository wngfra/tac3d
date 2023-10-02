import pytest

from nengo_spa import SemanticPointer
import numpy as np

from ssp.pointers import BaseVectors


@pytest.mark.parametrize("d", [1, 4, 9, 16, 129, 512])
def test_base_vectors(d, rng):
    """Tests generation of BaseVectors."""

    gen = BaseVectors(d, rng=rng)
    for _ in range(100):
        x = SemanticPointer(next(gen))
        assert len(x) == d
        assert np.allclose(np.linalg.norm(x.v), 1)

        sqrt_x = x ** 0.5
        x_check = sqrt_x * sqrt_x
        assert np.allclose(np.linalg.norm(sqrt_x.v), 1)

        assert np.allclose(x_check.v, x.v)
        assert np.allclose((sqrt_x ** 2).v, x.v)
        assert np.allclose((x ** (-1)).v, (~x).v)
        assert np.allclose((x ** 3.4 * x ** (-0.4)).v, (x * x * x).v)
