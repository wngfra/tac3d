# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np

from nengo.dists import Choice, Uniform
from nengo.neurons import AdaptiveLIF
from nengo.params import NumberParam


class AdaptiveExpLIF(AdaptiveLIF):
    def __init__(
        self,
        tau_n=1,
        inc_n=0.01,
        tau_rc=0.02,
        tau_ref=0.002,
        min_voltage=0,
        amplitude=1,
        initial_state=None,
    ):
        super().__init__(
            tau_n=tau_n,
            inc_n=inc_n,
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            min_voltage=min_voltage,
            amplitude=amplitude,
            initial_state=initial_state,
        )

    def step(self, dt, J, output, voltage, refractory_time, adaptation):
        n = adaptation
        super().step(dt, J - n, output, voltage, refractory_time, adaptation)
        
        n += (dt / self.tau_n) * (self.inc_n * output - n)
