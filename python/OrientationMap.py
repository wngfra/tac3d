# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import multivariate_normal

SQRT3_2 = np.sqrt(3) / 2


def generate_hexlattice(shape, d, theta):
    v = SQRT3_2 * d
    nrows, ncols = shape
    xs = np.linspace(0, ncols, int(ncols / d))
    ys = np.linspace(v, nrows, int(nrows / v) - 1)
    lattice = [(x, y) for x in xs for y in ys]
    lattice = np.asarray(lattice)
    if theta != 0:
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lattice = np.dot(lattice, R)
    return lattice


class OrientationMap:
    def __init__(self, shape=20, d=2, alpha=0, theta=5) -> None:
        if isinstance(shape, int):
            shape = (shape, shape)
        self._shape = shape

        dm = (
            (1 + alpha)
            / np.sqrt(alpha * alpha + 2 * (1 - np.cos(theta)) * (1 + alpha))
            * d
        )
        self.lattice_off = generate_hexlattice(shape, d, 0)
        self.lattice_on = generate_hexlattice(shape, dm, theta)

    @property
    def shape(self):
        return self._shape


def main():
    dipole_gaussian(shape=(20, 20))


def dipole_gaussian(center=(0, 0), theta=np.pi/4, scale=(10, 2), shape=(20, 20)):
    nrows, ncols = shape
    mu_on = np.asarray(center) + scale[1] * np.asarray([np.sin(theta), np.cos(theta)])
    mu_off = np.asarray(center) - scale[1] * np.asarray([np.sin(theta), np.cos(theta)]) 
    
    # Data
    x = np.linspace(-nrows//2, ncols//2, ncols)
    y = np.linspace(-nrows//2, nrows//2, nrows)
    X, Y = np.meshgrid(x, y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Multivariate Gaussian ON
    V = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    D = [[scale[1], 0], [0, scale[0]]]
    sigma = np.matmul(np.matmul(V, D), np.linalg.inv(V))
    rv = multivariate_normal(mu_on, sigma)
    pd_on = rv.pdf(pos)

    # Multivariate Gaussian OFF
    rv = multivariate_normal(mu_off, sigma)
    pd_off = -rv.pdf(pos)

    img = pd_on + pd_off

    # Plot
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
