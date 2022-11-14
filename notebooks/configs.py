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
from brian2 import *


# autopep8: off
events = {
    'event3': ['increase_threshold', 'v > v_th', 'v_th += delta_theta*rand()']
}

params = {
    # Model constants
    'C_mem'           : 200*pF,       # Membrane capacitance
    'delta_theta'     : 5*mV,         # Adaptive threshold step-size
    'g_l'             : 10*nS,        # Leak conductance
    'J_C'             : 1,            # Scale of the calcium variable
    'tau_c'           : 60*ms,        # Calcium variable time constant
    'tau_e'           : 5*ms,         # Excitatory synaptic time constant
    'tau_i'           : 5*ms,         # Inhibitory synaptic time constant
    'tau_r'           : 5*ms,         # Refractory period
    'tau_theta'       : 5*ms,         # Adaptive spiking threshold
    'V_ir'            : -80*mV,       # Inhibitory reverse potential
    'V_res'           : -60*mV,       # Resting potential
    'V_theta'         : -50*mV,       # Spiking threshold
    'w_e'             : 30*nS,        # Excitatory conductance increment
    'w_i'             : 30*nS,        # Inhibitory conductance increment
    'X_max'           : 1,            # Synaptic variable maximum
    'X_min'           : 0,            # Synaptic variable minimum

    # Initial values
    'c_initial'       : 2,
    'v_initial'       : 'V_res + rand()*(V_theta - V_res)',
    'X_initial'       : 'rand()*X_max',

    # Simulation parameters
    'defaultclock.dt' : 0.1*ms,      # Time step
}
# Thresholds and plasticity parameters
params['a']           = 0.1*params['X_max']
params['b']           = 0.1*params['X_max']
params['alpha']       = 3.5*params['X_max']*Hz
params['beta']        = 3.5*params['X_max']*Hz
params['theta_hup']   = 12*params['J_C']
params['theta_lup']   = 3*params['J_C']
params['theta_hdown'] = 4*params['J_C']
params['theta_ldown'] = 3*params['J_C']
params['theta_v']     = 0.8*params['V_theta']
params['theta_X']     = 0.5*params['X_max']

equations = {
    # Neuronal models
    'L1': '''
        dv/dt = (g_l*(V_res - v) + I(t,i))/C_mem : volt (unless refractory)
        ''',
    'L2': '''
        dv/dt = (g_l*(V_res - v) - g_e*v)/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        sum_w : 1
        ''',
    'L3': '''
        dv/dt = (g_l*(V_res - v) - g_e*v + g_i*(V_ir - v))/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        dg_i/dt = -g_i/tau_i : siemens
        dv_th/dt = (V_theta - v_th)/tau_theta : volt
        is_winner : boolean
        sum_w : 1
        ''',
    # Synaptic models
    'Syn12': '''
        w : 1
        sum_w_post = w : 1 (summed)
        ''',
    'Syn23': '''
        count : 1
        X_condition : 1
        dc/dt = -c/tau_c + J_C*count*Hz: 1 (clock-driven)
        dX/dt = (alpha*int(X > theta_X)*int(X < X_max) - beta*int(X <= theta_X)*int(X > X_min))*(1 - X_condition) : 1 (clock-driven)
        w = int(X > 0.5) : 1
        sum_w_post = w : 1 (summed)
        ''',
    'Syn33': '''
        w : 1
        ''',
    # Synaptic events
    'Pre12': '''
        g_e_post += int(sum_w_post >= 1)*w_e/(sum_w_post + 1e-6)
        ''',
    'Pre23': '''
        g_e_post += int(sum_w_post >= 1)*w_e/(sum_w_post + 1e-6)*X
        X += a*int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) - b*int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        X = clip(X, X_min, X_max)
        X_condition = int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) + int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        ''',
    'Pre33': '''
        g_i_post += w_i*abs(i - j)
        ''',
    'Post23': '''
        count += 1
        X_condition = 0
        ''',
}

connections = {
    'Syn12': {'mode': 'random'},
    'Syn23': {},
    'Syn33': {'condition': 'i != j'}
}

monitors = {
    'L1': ['v'],
    'L2': ['v', 'g_e'],
    'L3': ['v', 'v_th', 'g_e', 'g_i', 'sum_w'],
    'Syn23': ['X', 'w', 'c']
}
# autopep8: on