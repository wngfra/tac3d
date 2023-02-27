# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Optional

import numpy as np
import sympy
from sympy import dsolve, symbols
from sympy.utilities.lambdify import lambdify


class NeuronType:
    def __init__(self, eqs: list[str]):
        """_summary_

        Args:
            eqs (list[str]): _description_

        Raises:
            NameError: _description_
        """           
        if not isinstance(eqs, list):
            eqs = [eqs]
        
        self._eqs = dict()
        self._sol = dict()
        self._params: Optional[dict[str, dict[str, float]]] = None

        # Construct neuron ODEs
        var: Optional[str] = None
        pattern = r'd(.*?)/dt'
        t = symbols('t')
        
        for eq in eqs:
            m = re.search(pattern, eq)
            if m:
                var = m.group(1)
                locals()[var] = sympy.Function(var)(t)
            else:
                raise NameError("{} needs to define an ODE.".format(eq))
            # Find all undefined names in the string
            undefined_names = set(re.findall(r'\b([a-zA-Z_]\w*)\b', eq)) - set(dir(sympy))- set(var)
            for name in undefined_names:
                locals()[name] = symbols(name)

            eq_ = re.sub(pattern, lambda match: "sympy.diff({}, (t, 1))".format(match.group(1)), eq)
            lhs, rhs = eq_.split('=')
            lhs, rhs = eval(lhs), eval(rhs)

            self._eqs[var] = sympy.Eq(lhs, rhs)
            self._sol[var] = dsolve(lhs - rhs, locals()[var]).rhs
            self._sol_undefined = self._sol.copy()

    @property
    def eqs(self) -> dict[str, sympy.Eq]:
        """Get the ODEs representing neural dynamics.

        Returns:
            dict[str, sympy.Eq]: _description_
        """          
        return self._eqs
    
    @property
    def sol(self) -> dict[str, sympy.Expr]:
        """Get the analytical solutions of the ODEs.

        Returns:
            dict[str, sympy.Expr]: _description_
        """        
        return self._sol
    
    @property
    def params(self):
        return self._params

    def clear_params(self):
        self._sol = self._sol_undefined
        self._params: Optional[dict[str, dict[str, float]]] = None

    def set_params(self, params: dict[str, dict]):
        if params:
            self._params = dict()
            for k, v in params.items():
                for var, value in v.items():
                    self._sol[k] = self._sol[k].subs(var, value)
                    if len(self._params) == 0:
                        self._params[k] = {var: value}

    def __call__(self):
        func = []
        for expr in self._sol.values():
            args = list(expr.atoms(sympy.Symbol))
            func.append(lambdify(args, expr, "numpy"))
        return func


if __name__ == "__main__":
    eqs = ["dv/dt = - (g_l * (v_rest- v) + i) / c_mem"]
    neuron = NeuronType(eqs)
    neuron.set_params({'v': {'g_l': 1e-9, 'v_rest': -65e-3}})
    func = neuron()
    print(
        func[0](
            np.arange(100) + 1,
            200e-9,
            np.arange(100) + 1,
            np.random.rand(100) + 1,
        )
    )
    neuron.clear_params()
