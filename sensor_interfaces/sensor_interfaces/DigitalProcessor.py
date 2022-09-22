# Copyright (C) 2022 wngfra/captjulian
#
# tac3d is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tac3d is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tac3d. If not, see <http://www.gnu.org/licenses/>.

import numpy as np


class DigitalProcessor:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, xs, **kwargs):
        if isinstance(xs, np.ndarray):
            xs = np.asarray(xs)
        if hasattr(self, 'axis'):
            ys = np.apply_along_axis(self.apply, axis=self.axis, arr=xs)
        else:
            ys = self.apply(xs)
        return ys

    def __repr__(self) -> str:
        return self.__class__.__name__

    def apply(self, xs):
        raise NotImplementedError()


class Grayscale(DigitalProcessor):
    def __init__(self, keep_dim=False):
        super().__init__(keep_dim=keep_dim)

    def apply(self, xs):
        return self.equalize_histogram(xs)

    def equalize_histogram(self, img):

        # Get image histogram
        histogram, bins = np.histogram(img.flatten(), 256, density=True)
        cdf = histogram.cumsum()
        cdf = 255 * cdf / cdf[-1]

        # Use linear interpolation of cdf to find new pixel values
        img_equalized = np.interp(
            img.flatten(), bins[:-1], cdf)

        if self.keep_dim:
            img_equalized = img_equalized.reshape(img.shape)

        return img_equalized.astype(np.uint8)