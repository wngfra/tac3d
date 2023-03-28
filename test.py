# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

x, y = np.meshgrid(np.arange(15), np.arange(15))
ind = np.arange(225)
np.random.shuffle(ind)
print(x.ravel()[ind], y.ravel()[ind])