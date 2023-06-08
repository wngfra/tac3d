# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def sample_bipole_gaussian(shape, center, eigenvalues, phi, binary=False):
    """Sample from two 2D Gaussian with two peaks at the center of the bipole.
    Args:
        shape (tuple): Shape of the output array.
        center (tuple): Center of the bipole.
        eigenvalues (list like): Eigenvalues of the covariance matrix.
        phi (float): Rotation angle in radians.
    Returns:
        np.ndarray: 2D Gaussian array.
    """
    eigenvalues = np.asarray(eigenvalues)
    w0, w1 = eigenvalues.max(), eigenvalues.min()
    # Compute the mean (centre) of the two Gaussians
    mu = (
        np.tile(center, [2, 1])
        + np.asarray([[np.sin(phi), -np.cos(phi)], [-np.sin(phi), np.cos(phi)]]) * w1
    )

    # Generate the meshgrid
    assert shape[0] == shape[1], "shape must be square"
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    pos = np.zeros(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Multivariate Gaussian ON cell
    V = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]).T
    D = [[w0, 0], [0, w1]]
    sigma = np.matmul(np.matmul(V, D), np.linalg.inv(V))
    rv_on = multivariate_normal(mu[0], sigma)
    rv_off = multivariate_normal(mu[1], sigma)
    pdf = rv_on.pdf(pos) - rv_off.pdf(pos)
    if binary:
        pdf[pdf < 0] = -1
        pdf[pdf > 0] = 1
    return pdf


def main():
    im = sample_bipole_gaussian((9, 9), (4, 4), [3.0, 1.0], np.pi / 6)
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    main()
