# Copyright (C) 2022 wngfra
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
import matplotlib.pyplot as plt

from brian2 import *
from brian2tools import *

from TacNet import TacNet


set_device('cpp_standalone')

# Load dataset
streams = np.load("touch_stream.npz")
[sCurvy, sRound, sPlateau, oCurvy, oRound, oPlateau] = [
    np.squeeze(streams[file]) for file in streams.files]
frames = np.min([len(sCurvy), len(sRound), len(sPlateau)])

dt = 1*ms
inputs = sPlateau/sPlateau.max()*200*pA
inputs = inputs.reshape(inputs.shape[0], -1)
I = TimedArray(inputs, dt=dt)
duration = I.values.shape[0]*dt


model = TacNet([400, 20, 36])
mons = model.run(I, duration)

plt.plot(inputs)
plt.show()