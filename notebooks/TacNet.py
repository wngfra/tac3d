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

import json
import numpy as np

from brian2 import *
from brian2tools import brian_plot


class TacNet(objet):
    def __init__(self, input_dim, confg_dir) -> None:
        """Constructor of the Tactile Encoding Network.

        Args:
            input_dim (_type_): _description_
            confg_dir (_type_): _description_
        """        
        self.config = json.load(open(confg_dir))
        
        self.params = dict()
        for k, v in self.config['net_params']:
            self.net_params[k] = v
        for k, v in self.config['sim_params']:
            self.sim_params[k] = v