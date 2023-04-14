# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numpy import linalg
from scipy import ndimage


class BarGenerator:
    def __init__(self, shape):
        self._shape = np.asarray(shape)

    def __call__(self, centre=None, angle=0, dim=(1, 1)):
        """Generate a bar with given shape.

        Args:
            centre (tuple, optional): Centre of the bar. Defaults to image centre.
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
        
        # Compute left and bottom pads
        padleft, padbottom = (L - self.shape[0]) // 2, (L - self.shape[1]) // 2
        if centre is None:
            x, y = L // 2, L // 2
        else:
            x, y = int(centre[0] + padleft), int(centre[1] + padbottom)
        
        # Compute the start position for the bar, from left and bottom
        # FIXME: This is not correct, should be from centre, problems with odd dimensions
        startx = x - dim[0] // 2
        starty = y - dim[1] // 2
        padded[startx : startx + dim[0], starty : starty + dim[1]] = 1
        
        padded = ndimage.rotate(padded, angle)
        bar = padded[
            padleft : padleft + self.shape[0], padbottom : padbottom + self.shape[1]
        ]
        bar[bar < 0] = 0
        bar /= bar.max() if bar.max() > 0 else 1
        
        return bar

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
            np.ndarray: Info with shape (num_samples, 1). Defaults to degrees in [0, 180].
        """
        bars = np.empty((num_samples, *self._shape))
        info = np.empty((num_samples, 1))
        for i in range(num_samples):
            x = np.random.randint(min_offset[0], max_offset[0])
            y = np.random.randint(min_offset[1], max_offset[1])
            angle = np.random.randint(min_angle, max_angle)
            dimx = np.random.randint(min_dim[0], max_dim[0])
            dimy = np.random.randint(min_dim[1], max_dim[1])
            bars[i] = self((x, y), angle, (dimx, dimy))
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
    bar = bg((7, 5), 40, (2, 15))
    import matplotlib.pyplot as plt

    plt.imshow(bar)
    plt.grid()
    plt.show()
