# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import multivariate_normal


def check_shape(shape):
    shape = np.asarray(shape)
    if shape.size != 2:
        match shape.size:
            case 1:
                shape = np.asarray([shape, shape])
            case _:
                shape = np.asarray([shape[0], shape[0]])
    return shape


def generate_hexlattice(shape, d, theta: float, lattice_type="on"):
    """_summary_

    Args:
        shape (tuple or int): _description_
        d (_type_): _description_
        theta (float): Rotation angle in degrees.
        lattice_type (str, optional): _description_. Defaults to "on".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """ """_summary_

    Args:
        shape (_type_): _description_
        d (_type_): _description_
        theta (_type_): Rotation angle in degrees.
        lattice_type (str, optional): _description_. Defaults to "on".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    match lattice_type:
        case "on":
            fillin = 1
        case "off":
            fillin = -1
        case _:
            raise ValueError("lattice_type must be either 'on' or 'off'")
    shape = check_shape(shape)
    v = int(0.5 * np.sqrt(3) * d)  # y-axis distance between two adjacent hexagons
    d = int(d)
    phi = theta * np.pi / 180
    height_offset = np.ceil(np.abs(shape[0] * np.sin(phi) * np.cos(phi))).astype(int)
    width_offset = np.ceil(np.abs(shape[1] * np.sin(phi) * np.cos(phi))).astype(int)
    height = np.ceil(shape[0]).astype(int) + height_offset * 2
    width = np.ceil(shape[1]).astype(int) + width_offset * 2
    lattice = np.zeros((height, width))

    lattice[0:height:v, d:width:d] = fillin

    if theta != 0:
        lattice = ndimage.rotate(
            lattice, theta, reshape=False, order=0, mode="constant", cval=0
        )

    return lattice[
        height_offset : height_offset + shape[0], width_offset : width_offset + shape[1]
    ]


def sample_bipole_gaussian(shape, center, scale, phi):
    """Sample a 2D Gaussian with two peaks at center + scale * [sin(phi), cos(phi)] and center - scale * [sin(phi), cos(phi)].
    Args:
        shape (tuple): Shape of the output array.
        center (tuple): Center of the Gaussian.
        scale (tuple): Scale of the eigenvalues of the covariance matrix.
        phi (float): Rotation angle in radians.
    Returns:
        np.ndarray: 2D Gaussian array.
    """
    mu_on = np.asarray(center) + scale[1] * np.asarray([np.sin(phi), np.cos(phi)])
    mu_off = np.asarray(center) - scale[1] * np.asarray([np.sin(phi), np.cos(phi)])
    # Data
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # Multivariate Gaussian ON
    V = [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
    D = [[scale[1], 0], [0, scale[0]]]
    sigma = np.matmul(np.matmul(V, D), np.linalg.inv(V))
    rv = multivariate_normal(mu_on, sigma)
    pd_on = rv.pdf(pos)
    # Multivariate Gaussian OFF
    rv = multivariate_normal(mu_off, sigma)
    pd_off = -rv.pdf(pos)
    return pd_on + pd_off


class OrientationMap:
    def __init__(self, shape=120, d=4, alpha=0.5, theta=5) -> None:
        shape = check_shape(shape)
        self._shape = shape
        self._params = {"d": d, "alpha": alpha, "phi": theta / 180 * np.pi}

        self.lattice = {
            "on": generate_hexlattice(shape, d, 0, "on"),
            "off": generate_hexlattice(shape, (1 + alpha) * d, theta, "off"),
        }
        self.construct_map()

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._shape[0] * self._shape[1]

    @property
    def scaling_factor(self):
        alpha = self._params["alpha"]
        phi = self._params["phi"]
        return (1 + alpha) / np.sqrt(
            alpha * alpha + 2 * (1 - np.cos(phi)) * (1 + alpha)
        )

    def __call__(self):
        return self.map

    def construct_map(self):
        superposition = self.lattice["on"] + self.lattice["off"]
        self.map = np.zeros(self.shape)
        nonzero_indices = np.transpose(np.nonzero(superposition))
        for x, y in nonzero_indices:
            if superposition[x, y] == 1:
                query_indices = np.argwhere(superposition == -1)
            elif superposition[x, y] == -1:
                query_indices = np.argwhere(superposition == 1)
            distances = np.linalg.norm(query_indices - np.array([x, y]), axis=1)
            neartest_index = query_indices[np.argmin(distances)]
            self.map[x, y] = np.arctan2(neartest_index[0] - x, neartest_index[1] - y)

        zero_indices = np.argwhere(superposition == 0)
        for x, y in zero_indices:
            query_indices = nonzero_indices
            distances = np.linalg.norm(query_indices - np.array([x, y]), axis=1)
            neartest_index = query_indices[np.argmin(distances)]
            self.map[x, y] = self.map[neartest_index[0], neartest_index[1]]

    def gen_transform(self, shape_in):
        shape_in = check_shape(shape_in)
        nrows, ncols = shape_in
        assert (
            len(shape_in) == 2 and nrows > self.shape[0] and ncols > self.shape[1]
        ), "Expect larger input field than the oritentation map."
        eigenvalues = (4, 2)
        stride_x = ncols // self.shape[1]
        stride_y = nrows // self.shape[0]
        transform = np.empty((self.size, nrows * ncols))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                phi = self.map[i, j]  # Obtain angle selectivity
                # Obtain bipole receptive field
                bipole_rf = sample_bipole_gaussian(
                    (nrows, ncols),
                    (
                        i * stride_y + eigenvalues[0] // 2,
                        j * stride_x + eigenvalues[1] // 2,
                    ),
                    eigenvalues,
                    phi,
                )
                transform[i * self.shape[1] + j] = bipole_rf.ravel()

        return transform


def main():
    orimap = OrientationMap(14, 4, 1, 22.5)
    OM = orimap()
    T = orimap.gen_transform(15)
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(OM)
    axs[1].imshow(T)
    plt.show()


if __name__ == "__main__":
    main()