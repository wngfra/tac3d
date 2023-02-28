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
        """
        self._initial_states: Optional[dict[str, float]] = None
        self._func: Optional[list] = None

        if not isinstance(eqs, list):
            eqs = [eqs]
        self._exprs = []
        self._params = dict()
        self._states = dict()

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

        # Backup the initial expressions and symbols
        self._exprs_undefined = self._exprs.copy()
        self._gen_func()

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
        else:
            self.exprs = self._exprs_undefined
            self._params = None

    @property
    def states(self) -> dict[str, float]:
        return self._states

    def _gen_func(self):
        t = symbols("t")
        vars_list = [t]
        for var_name in self.states.keys():
            locals()[var_name] = sympy.Function(var_name)(t)
            vars_list.append(locals()[var_name])
        self._func = [
            lambdify(vars_list, expr, modules="numpy") for expr in self._exprs
        ]

    def step(self, dt: float):
        """Compute the states at the next time step.

        Args:
            dt (float): _description_
        """
        for expr in self.exprs:
            args = list(expr.atoms(sympy.Symbol))
            func = lambdify(args, expr, "numpy")
