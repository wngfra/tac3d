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

    def generate_samples(
        self, num_samples, dim, shift=None, start_angle=0, step=1, add_test=False
    ):
        bars = [self(shift, start_angle + i * step, dim) for i in range(num_samples)]
        info = np.arange(start_angle, start_angle + num_samples * step, step)
        bars = np.asarray(bars)
        info = np.asarray(info)
        if add_test:
            rng = np.random.default_rng(seed=0)
            arr = np.arange(num_samples, dtype=int)
            np.random.shuffle(arr)
            bars = np.concatenate((bars, bars[arr]))
            info = np.concatenate((info, info[arr]))
        return bars, info

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


if __name__ == "__main__":
    bg = BarGenerator((15, 15))
    bar = bg((0, 0), 60, (2, 21))
    import matplotlib.pyplot as plt

    plt.imshow(bar)
    plt.grid()
    plt.show()
