# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import multivariate_normal


def check_shape(shape):
    """Check the shape of the input array. If the shape is not a 2D array, convert it to a 2D array.

    Args:
        shape (Any): A scalar or a 1D array.

    Returns:
        np.ndarray: Shape as a 2D array.
    """
    shape = np.asarray(shape)
    if shape.size != 2:
        match shape.size:
            case 1:
                shape = np.asarray([shape, shape])
            case _:
                shape = np.asarray([shape[0], shape[1]])
    return shape


def generate_hexlattice(shape, d, theta: float, lattice_type="on", noise=True):
    """Generate a hexagonal lattice with a given shape and lattice constant.

    Args:
        shape (tuple or int): Shape of the lattice.
        d (float or int): The lattice constant (spacing).
        theta (float): Rotation angle in degrees.
        lattice_type (str, optional): Type parameter that decides the fillin. Defaults to "on".

    Raises:
        ValueError: lattice_type must be either 'on' or 'off'

    Returns:
        np.ndarray: Generated hexagonal lattice as a 2D array.
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
    d = int(d)  # x-axis distance between two adjacent hexagons
    phi = np.deg2rad(theta)
    height_offset = np.ceil(np.abs(shape[0] * np.sin(phi) * np.cos(phi))).astype(int)
    width_offset = np.ceil(np.abs(shape[1] * np.sin(phi) * np.cos(phi))).astype(int)
    height = np.ceil(shape[0]).astype(int) + height_offset * 2
    width = np.ceil(shape[1]).astype(int) + width_offset * 2
    lattice = np.zeros((height, width))

    # Fill odd rows
    ys, xs = np.arange(0, height, 2 * v), np.arange(0, width, d)
    if noise:
        ys += np.random.normal(0, 2, ys.size).astype(int)
        xs += np.random.normal(0, 2, xs.size).astype(int)
    ys, xs = np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)
    lattice[np.ix_(ys, xs)] = fillin

    # Fill even rows
    ys, xs = np.arange(v, height, 2 * v), np.arange(d // 2, width, d)
    if noise:
        ys += np.random.normal(0, 2, ys.size).astype(int)
        xs += np.random.normal(0, 2, xs.size).astype(int)
    ys, xs = np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)
    lattice[np.ix_(ys, xs)] = fillin

    # Rotate the lattice
    if theta != 0:
        lattice = ndimage.rotate(
            lattice, theta, reshape=False, order=0, mode="constant", cval=0
        )

    # Crop the lattice to shape
    return lattice[
        height_offset : height_offset + shape[0], width_offset : width_offset + shape[1]
    ]


def sample_bipole_gaussian(shape, center, eigenvalues, phi):
    """Sample from two 2D Gaussian with two peaks at the center of the bipole.
    Args:
        shape (tuple): Shape of the output array.
        center (tuple): Center of the bipole without shift.
        eigenvalues (list like): Eigenvalues of the covariance matrix.
        phi (float): Rotation angle in radians.
    Returns:
        np.ndarray: 2D Gaussian array.
    """
    # FIXME: The receptive field is not symmetric in the input space. Deal with the boundary issues.
    eigenvalues = np.asarray(eigenvalues)
    w0, w1 = eigenvalues.max(), eigenvalues.min()
    center += np.array([w0 // 2 * np.sin(phi), w0 // 2 * np.cos(phi)]).astype(int)

    # Compute mean of the Gaussian for ON and OFF subfields
    mu_on = np.asarray(center) + w1 * np.asarray([-np.cos(phi), np.sin(phi)])
    mu_off = np.asarray(center) + w1 * np.asarray([np.cos(phi), -np.sin(phi)])

    # Generate the meshgrid
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Multivariate Gaussian ON cell
    V = [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
    D = [[w0, 0], [0, w1]]
    sigma = np.matmul(np.matmul(V, D), np.linalg.inv(V))
    rv = multivariate_normal(mu_on, sigma)
    pd_on = rv.pdf(pos)
    # Multivariate Gaussian OFF cell
    rv = multivariate_normal(mu_off, sigma)
    pd_off = -rv.pdf(pos)
    return pd_on + pd_off


class OrientationMap:
    """Orientation map generator. The orientation map is the superposition of two hexagonal lattices with different spacing and orientation. One lattice represents the subfield of ON cell and the other represents the subfield of OFF cell. 
    The orientation selectivity is computed as the perpendicular direction to the ON-OFF dipoles.
    """

    def __init__(self, shape=120, d=2, alpha=0.5, theta=5, zoom=1) -> None:
        shape = check_shape(shape)
        self._shape = shape
        self._params = {"d": d, "alpha": alpha, "phi": theta / 180 * np.pi}

        self.lattice = {
            "on": generate_hexlattice(shape, d, 0, "on"),
            "off": generate_hexlattice(shape, (1 + alpha) * d, theta, "off"),
        }
        self.construct_map()
        self.zoom(zoom)

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

    @property
    def unique(self):
        return np.unique(self.map)

    def __call__(self):
        return self.map

    def construct_map(self):
        self.map = np.zeros(self.shape)
        nonzero_indices = np.argwhere(self.lattice["on"] != 0)
        for y, x in nonzero_indices:
            query_indices = np.argwhere(self.lattice["off"] != 0)
            distances = np.linalg.norm(query_indices - np.array([y, x]), axis=1)
            nearest_index = query_indices[np.argmin(distances)]
            orientation = (
                np.arctan2(nearest_index[0] - y, nearest_index[1] - x) + np.pi / 2
            )
            if orientation < 0:
                orientation += np.pi
            if orientation > np.pi:
                orientation -= np.pi
            self.map[y, x] = orientation

        zero_indices = np.argwhere(self.map == 0)
        for y, x in zero_indices:
            query_indices = nonzero_indices
            distances = np.linalg.norm(query_indices - np.array([y, x]), axis=1)
            nearest_index = query_indices[np.argmin(distances)]
            self.map[y, x] = self.map[nearest_index[0], nearest_index[1]]

    def gen_transform(self, shape_in, eigenvalues=(6, 3)):
        shape_in = check_shape(shape_in)
        nrows, ncols = shape_in
        assert (
            len(shape_in) == 2 and nrows > self.shape[0] and ncols > self.shape[1]
        ), "Expect larger input field than the oritentation map."
        stride_x = ncols // self.shape[1]
        stride_y = nrows // self.shape[0]
        transform = np.empty((self.size, nrows * ncols))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Obtain angle selectivity
                phi = self.map[i, j]
                # Obtain bipole receptive field
                bipole_rf = sample_bipole_gaussian(
                    (nrows, ncols),
                    (
                        i * stride_y,
                        j * stride_x,
                    ),
                    eigenvalues,
                    phi,
                )
                transform[i * self.shape[1] + j] = bipole_rf.ravel()
        plt.imshow(transform)
        plt.tight_layout()
        plt.show()
        return transform

    def zoom(self, zoom):
        assert zoom > 0, "Zoom factor must be positive."
        if zoom != 1:
            self.map = ndimage.zoom(self.map, zoom=zoom, order=0)
            self._shape = self.map.shape


def main():
    orimap = OrientationMap(30, 4, 1, 10)
    orimap.zoom(0.5)
    OM = orimap()
    T = orimap.gen_transform(36, (10, 3))
    _, axs = plt.subplots(1, 2)
    axs[0].imshow(OM)
    axs[1].imshow(T)
    plt.suptitle("Scale: {:.1f}".format(orimap.scaling_factor))
    plt.show()


if __name__ == "__main__":
    main()
