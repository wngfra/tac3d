# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from scipy import ndimage


class BarGenerator:
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, offset=(0, 0), angle=0, dim=(1, 1)):
        """Generate a bar with given shape.

        Args:
            offset (tuple, optional): Offset of the bar. Defaults to (0, 0).
            angle (float, optional): Angle of the bar. Defaults to 0.
            dim (tuple, optional): Dimension of the bar. Defaults to (1, 1).

        Returns:
            np.ndarray: Bar with shape (width, shape[1]).
        """
        bar = np.zeros(self._shape)
        x, y = int(offset[0]), int(offset[1])
        dx, dy = int(dim[0]), int(dim[1])
        if x + dx > self._shape[0]:
            dx = self._shape[0] - x
        if y + dy > self._shape[1]:
            dy = self._shape[1] - y
        bar[x : x + dx, y : y + dy] = 1

        rotated_bar = ndimage.rotate(bar, angle, reshape=False)

        return rotated_bar

    def generate_bars(
        self,
        num_samples,
        min_offset=(0, 0),
        max_offset=(1, 1),
        min_angle=0,
        max_angle=180,
        min_dim=(1, 1),
        max_dim=(1, 1),
    ):
        """Generate a set of random bars.

        Args:
            num_samples (int): Number of samples to generate.
            min_offset (tuple, optional): Minimum offset of the bar. Defaults to (0, 0).
            max_offset (tuple, optional): Maximum offset of the bar. Defaults to (0, 0).
            min_angle (float, optional): Minimum angle of the bar. Defaults to 0.
            max_angle (float, optional): Maximum angle of the bar. Defaults to 180.
            min_dim (tuple, optional): Minimum dimension of the bar. Defaults to (1, 1).
            max_dim (tuple, optional): Maximum dimension of the bar. Defaults to (1, 1).

        Returns:
            np.ndarray: Bars with shape (num_samples, width, shape[1]).
            np.ndarray: Info with shape (num_samples, 1).
        """
        bars = np.empty((num_samples, *self._shape))
        info = np.empty((num_samples, 1))
        for i in range(num_samples):
            dx = np.random.randint(min_offset[0], max_offset[0])
            dy = np.random.randint(min_offset[1], max_offset[1])
            angle = np.random.randint(min_angle, max_angle)
            dimx = np.random.randint(min_dim[0], max_dim[0])
            dimy = np.random.randint(min_dim[1], max_dim[1])
            bars[i] = self((dx, dy), angle, (dimx, dimy))
            info[i] = angle
        return bars, info

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape


if __name__ == "__main__":
    bg = BarGenerator((15, 15))
    bars, info = bg.generate_bars(
        100,
        max_offset=(7, 7),
        min_angle=0,
        max_angle=180,
        min_dim=(2, 10),
        max_dim=(5, 15),
    )
    import matplotlib.pyplot as plt

    plt.imshow(bars[50])
    plt.show()
