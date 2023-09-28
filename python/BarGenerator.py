# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy import linalg
from scipy import ndimage


class BarGenerator:
    def __init__(self, shape):
        self._shape = np.asarray(shape)

    def __call__(self, shift=None, angle=0, dim=(1, 1)):
        """Generate a bar with given shape.

        Args:
            shift (tuple, optional): Shift of the bar centre. Defaults to image centre.
            angle (float, optional): Angle of the bar. Defaults to 0.
            dim (tuple, optional): Dimension of the bar. Defaults to (1, 1).

        Returns:
            np.ndarray: Bar with shape (width, shape[1]).
        """
        # Compute the new side length for the padded image
        L = linalg.norm(self.shape).astype(int)
        if L % 2 == 0:
            L += 1
        padded = np.zeros((L, L))
        if dim[0] > L or dim[1] > L:
            raise ValueError("Bar dimension is larger than image size.")
        padded[
            L // 2 - dim[0] // 2 : L // 2 - dim[0] // 2 + dim[0],
            L // 2 - dim[1] // 2 : L // 2 - dim[1] // 2 + dim[1],
        ] = 1
        padded = ndimage.rotate(padded, angle)
        if shift:
            padded = ndimage.shift(padded, shift)
        padbottom, padleft = (padded.shape - self.shape) // 2

        cropped = padded[
            padbottom : padbottom + self.shape[0], padleft : padleft + self.shape[1]
        ]

        cropped[cropped < 0.1] = 0
        cropped /= cropped.max() if cropped.max() > 0 else 1

        return cropped

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def generate_samples(
        self,
        num_samples,
        dim,
        shift=None,
        start_angle=0,
        step=1,
        repeats=2,
        add_test=False,
    ):
        bar = [self(shift, start_angle + i * step, dim) for i in range(num_samples)]
        inf = np.arange(start_angle, start_angle + num_samples * step, step)
        bar = np.asarray(bar)
        bars = np.repeat(bar, axis=0, repeats=repeats)
        infs = np.repeat(inf, axis=0, repeats=repeats)
        rng = np.random.default_rng(seed=0)
        arr = np.arange(infs.size, dtype=int)
        np.random.shuffle(arr)
        bars = bars[arr]
        infs = infs[arr]
        if add_test:
            bars = np.concatenate((bars, bar))
            infs = np.concatenate((infs, inf))
        return bars, infs


def gen_transform(pattern=None, **kwargs):
    def inner(shape):
        """Closure of the transform matrix generator.

        Args:
            shape (array_like): Linear transform mapping of shape (size_out, size_mid).
        Returns:
            inner: Function that returns the transform matrix.
        """
        W = np.zeros(shape)

        match pattern:
            case "uniform_inhibition":
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                W = -np.ones(shape) + 2 * np.eye(shape[0])
            case "circular_inhibition":
                # For self-connections
                assert shape[0] == shape[1], "Transform matrix is not symmetric!"
                weight = np.abs(np.arange(shape[0]) - shape[0] // 2)
                for i in range(shape[0]):
                    W[i, :] = np.roll(weight, i + shape[0] // 2 + 1)
                W = -W + np.eye(shape[0])
            case 0:
                pass
            case 1:
                W = np.ones(W.shape)
            case _:
                W = np.random.randint(0, 2, W.shape)
        return W

    return inner