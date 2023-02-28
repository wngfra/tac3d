# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from base import NeuronType


class Neuron(NeuronType):
    def __init__(self):
        eqs = [
            "dv/dt = (g_l*(V_res - v) - g_e*v + g_i*(V_ir - v))/C_mem",
            "dg_e/dt = -g_e/tau_e",
            "dg_i/dt = -g_i/tau_i",
            "dv_th/dt = (V_theta - v_th)/tau_theta",
        ]
        super().__init__(eqs)

        self.params = {
            "g_l": 10e-9,
            "V_res": -65e-3,
            "V_ir": -80e-3,
            "C_mem": 200e-12,
            "tau_e": 10e-3,
            "tau_i": 10e-3,
            "V_theta": -55e-3,
            "tau_theta": 5e-3,
        }


if __name__ == "__main__":
    neuron = Neuron()
    print(neuron._func)