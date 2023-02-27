# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import numpy as np

from sympy import diff, dsolve, symbols, Function
from sympy.utilities.lambdify import lambdify


class NeuronType:
    def __init__(self, eqs: str, vars: str, params: dict[str, float]):
        self._vars: Optional[dict[str, any]] = None
        self._params = params

        self.declare_variables(vars)

    def declare_variables(self, vars: str):
        exec(vars + " = symbols('{}')".format(vars))
        if not locals("v"):
            v = symbols("v")

    def __call__(self):
        return lambdify(self._var, self._var, "numpy")


if __name__ == "__main__":
    params = {"V_THETA": -55e-3, "C_MEM": 200e-12, "V_REST": -65e-3, "G_L": 10e-9}

    a, b, t, i, C1 = symbols("a b t i C1")
    v = symbols("v", cls=Function)
    eq = diff(v(t), (t, 1)) - (G_L * (V_REST - v(t)) + i) / C_MEM
    sol = dsolve(eq, v(t))

    func = lambdify([t, i, C1], sol.rhs, "numpy")

    print(func(np.arange(100), np.arange(100), np.random.rand(100)))
    LIF = NeuronType()
