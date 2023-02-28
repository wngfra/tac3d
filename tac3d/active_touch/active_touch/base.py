# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Optional

import numpy as np
import sympy
from sympy import diff, dsolve, symbols
from sympy.solvers.ode.systems import dsolve_system
from sympy.utilities.lambdify import lambdify


_DEPENDENT_VAR_PATTERN = r"d(.*?)/dt"


class NeuronType:
    def __init__(self, eqs: list[str]):
        """Constructor of the NeuronType class.

        Args:
            eqs (list[str]): list of ODEs.

        Raises:
            NameError: Raise exception if ODEs are not well defined.

        Example:
            eqs = ["dv/dt = (g_l * (V_res - v) + I_ext) / C_mem"]
            neuron = NeuronType(eqs)
        """
        self._initial_states: Optional[dict[str, float]] = None
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        if not isinstance(eqs, list):
            eqs = [eqs]
        self._exprs = []
        self._params = dict()
        self._states = dict()
        self._linearized = False

        # Construct neuron ODEs
        t = symbols("t")

        # Define functions
        for eq in eqs:
            m = re.search(_DEPENDENT_VAR_PATTERN, eq)
            if m:
                var_name = m.group(1)
                locals()[var_name] = sympy.Function(var_name)(t)
                self._states[var_name] = 0.0
            else:
                raise NameError("{} needs to define an ODE.".format(eq))

        # Define parameters
        for var_name, eq in zip(self._states.keys(), eqs):
            # Find all undefined names in the string
            undefined_names = (
                set(re.findall(r"\b([a-zA-Z_]\w*)\b", eq))
                - set(dir(sympy))
                - set(self._params.keys())
                - set(self._states.keys())
            )
            for name in undefined_names:
                locals()[name] = symbols(name)
                self._params[name] = 0.0

            expr = eval(eq.split("=")[1]).expand()
            self._exprs.append(expr)

        # Backup the initial expressions
        self._exprs_undefined = self._exprs.copy()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: dict[str, float]):
        """Set the parameters.

        Args:
            params (dict[str, float]): Parameter dict. If None, clear all parameters.
        """
        if params:
            self._params = params
            for index, expr in enumerate(self._exprs):
                for k, v in params.items():
                    expr = expr.subs(k, v)
                self._exprs[index] = expr
            self._linearize()
        else:
            self.exprs = self._exprs_undefined
            self._params = None
            self._linearized = False

    @property
    def states(self) -> dict[str, float]:
        return self._states

    def _linearize(self):
        """Organize the ODEs into dX/dt = AX + B for fast evaluation.
        A : self._A
        B : self._B
        """
        # FIXME not able to linearize the exprs
        self._linearized = True
        A = []
        B = []
        for var_name, expr in zip(self._states.keys(), self._exprs):
            coeff = expr.coeff(var)
            A.append(coeff)
        self._A = A
        self._B = B

    def step(self, dt: float):
        """Compute the states at the next time step.

        Args:
            dt (float): _description_
        """
        if not self._linearized:
            self._linearize()
        for expr in self.exprs:
            args = list(expr.atoms(sympy.Symbol))
            func = lambdify(args, expr, "numpy")
