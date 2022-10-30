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


class TacSet(object):
    def __init__(self, streams, labels) -> None:
        """Construct the TacSet with streams and labels.

        Args:
            streams (list): A list of data streams grouped by labels.
            labels (list): A list of labels (str) in the same order as the streams.
        """
        label2id = dict()
        for i, label in enumerate(labels):
            label2id[label] = i
        labels = [np.repeat(label2id[label], len(streams[i])) for i, label in enumerate(labels)]
        
        self.data = np.squeeze(np.vstack(streams))
        self.labels = np.hstack(labels)        
    
    
    def __getitem__(self, key):
        """Get a sample from the TacSet

        Args:
            key (int): The index of the sample.

        Returns:
            (nd.array, int): A sample of the data and its label
        """        
        return self.data[key], self.labels[key]