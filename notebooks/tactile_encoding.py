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

from scipy.special import gamma


class TacSet(object):
    def __init__(self, streams, labels, normalized=False) -> None:
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
        self.normalized = normalized
    
    
    def __getitem__(self, key):
        """Get a sample from the TacSet

        Args:
            key (int): The index of the sample.

        Returns:
            (nd.array, int): A sample of the data and its label
        """
        return self.data[key], self.labels[key]

    
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


class MGGD:
    def __init__(self, m, beta, mu, M):
        self.m = m
        self.beta = beta
        self.mu = mu
        if not check_symmetric(M):
            raise ValueError("M is not symmetric!")
        self.M = M
        k = np.linalg.matrix_rank(self.M)
        
        self.coeffp = 1/np.sqrt(np.linalg.det(M))
        self.coeffh = beta*gamma(k/2)*(-np.power(np.pi, k/2)*gamma(0.5*k/beta)*np.power(2, 0.5*k/beta))*np.power(m, -0.5*k)
    
    def __call__(self, x):
        y = x.T*np.linalg.inv(M)*x
        h = self.coeffh*np.exp(-0.5*np.power(y, beta)*np.power(m, -beta))
        return self.coeffp*h