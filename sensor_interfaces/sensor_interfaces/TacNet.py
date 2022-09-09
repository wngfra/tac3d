import argparse
import multiprocessing
import numpy as np

from brian2 import *
from brian2tools import brian_plot

# autopep8: off
params = {
    # Model constants
    'C_mem'           : 200*pF,       # Membrane capacitance
    'delta_theta'     : 5*mV,         # Adaptive threshold step-size
    'g_l'             : 10*nS,        # Leak conductance
    'J_C'             : 1,            # Scale of the calcium variable
    'tau_c'           : 60*ms,        # Calcium variable time constant
    'tau_e'           : 5*ms,         # Excitatory synaptic time constant
    'tau_i'           : 10*ms,        # Inhibitory synaptic time constant
    'tau_r'           : 5*ms,         # Refractory period
    'tau_theta'       : 10*ms,        # Adaptive spiking threshold
    'V_ir'            : -80*mV,       # Inhibitory reverse potential
    'V_res'           : -60*mV,       # Resting potential
    'V_theta'         : -50*mV,       # Spiking threshold
    'w_e'             : 30*nS,        # Excitatory conductance increment
    'w_i'             : 50*nS,        # Inhibitory conductance increment
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

Eqs = {
    # Neuronal models
    'lif1': '''
        dv/dt = (g_l*(V_res - v) + I(t,i))/C_mem : volt (unless refractory)
        ''',
    'lif2': '''
        dv/dt = (g_l*(V_res - v) - g_e*v)/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        sum_w : 1
        ''',
    'lif3': '''
        dv/dt = (g_l*(V_res - v) - g_e*v + g_i*(V_ir - v))/C_mem : volt (unless refractory)
        dg_e/dt = -g_e/tau_e : siemens
        dg_i/dt = -g_i/tau_i : siemens
        dv_th/dt = (V_theta - v_th)/tau_theta : volt
        sum_w : 1
        ''',
    # Synaptic models
    'syn12': '''
        w : 1
        sum_w_post = w : 1 (summed)
        ''',
    'syn23': '''
        delta : 1
        X_condition : 1
        dc/dt = -c/tau_c + J_C*delta*Hz: 1 (clock-driven)
        dX/dt = (alpha*int(X > theta_X)*int(X < X_max) - beta*int(X <= theta_X)*int(X > X_min))*(1 - X_condition) : 1 (clock-driven)
        w = int(X > 0.5) : 1
        sum_w_post = w : 1 (summed)
        ''',
    # Synaptic events
    'pre12': '''
        g_e_post += w_e/sum_w_post
        ''',
    'pre23': '''
        g_e_post += w*w_e/sum_w_post
        X += a*int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) - b*int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        X = clip(X, X_min, X_max)
        X_condition = int(v_pre > theta_v)*int(theta_lup < c)*int(c < theta_hup) + int(v_pre <= theta_v)*int(theta_ldown < c)*int(c < theta_hdown)
        ''',
    'pre33': '''
        g_i_post += w_i
        ''',
    'post23': '''
        delta += 1
        X_condition = 0
        ''',
}
# autopep8: on


class WTA_SNN(object):
    def __init__(self, num_input, num_hidden, num_output, wta=True, monitor=False):
        """Construct a 3-layer SNN network with an optional WTA neuron.

        Args:
            num_input (int): The number of neurons in the input layer.
            num_hidden (int): The number of neurons in the hidden layer.
            num_output (int): The number of neurons in the output layer.
            wta (bool, optional): Option to the extra WTA neuron after the output layer. Defaults to True.
            monitor (bool, optional): Option to the spike/state monitors. Defaults to False.
        """

        # autopep8: off
        # Store network parameters
        self.num_input  = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.wta        = wta
        self.monitor    = monitor

        # Define neurons
        self.lif1 = NeuronGroup(num_input,
            model=Eqs['lif1'],
            threshold='v > V_theta',
            refractory='tau_r',
            reset='v = V_res',
            method='euler',
            name='input_layer',
            namespace=params)
        self.lif2 = NeuronGroup(num_hidden,
            model=Eqs['lif2'],
            threshold='v > V_theta',
            refractory='tau_r',
            reset='v = V_res',
            method='euler',
            name='hidden_layer',
            namespace=params)
        self.lif3 = NeuronGroup(num_output,
            model=Eqs['lif3'],
            threshold='v > v_th',
            refractory='tau_r + rand()*tau_r',
            reset='v = V_res; v_th = V_theta',
            events={'increase_threshold': 'v > v_th'},
            method='euler',
            name='output_layer',
            namespace=params)

        self.lif3.run_on_event('increase_threshold', 'v_th += delta_theta*rand()')

        # Define synapses
        self.syn12 = Synapses(self.lif1,
            self.lif2,
            model=Eqs['syn12'],
            on_pre=Eqs['pre12'],
            method='euler',
            name='synapse_12',
            namespace=params)
        self.syn23 = Synapses(self.lif2,
            self.lif3,
            model=Eqs['syn23'],
            on_pre=Eqs['pre23'],
            on_post=Eqs['post23'],
            method='euler',
            name='synapse_23',
            namespace=params)
        self.syn33 = Synapses(self.lif3,
            self.lif3,
            model='w : 1',
            on_pre=Eqs['pre33'],
            name='synapse_33_inh',
            namespace=params)

        # Define monitors
        self.mons = {
            's_l1':  SpikeMonitor(self.lif1, record=True, name='spike_l1'),
            's_l2':  SpikeMonitor(self.lif2, record=True, name='spike_l2'),
            's_l3':  SpikeMonitor(self.lif3, record=True, name='spike_l3'),
            'lif1':  StateMonitor(self.lif1, 'v', record=True, name='lif1'),
            'lif2':  StateMonitor(self.lif2, ['v', 'g_e'], record=True, name='lif2'),
            'lif3':  StateMonitor(self.lif3, ['v', 'v_th', 'g_e', 'g_i', 'sum_w'], record=True, name='lif3')
        }
        # autopep8: on

        self.initialized = False

    def init_network(self, p=1):
        """Connect the layers with synapses and initialize parameters.

        Args:
            p (int, optional): Determines the probability in [0, 1] of synaptic connections. Defaults to 1.
        """

        self.initialized = True

        # Configures receptive field
        np.random.seed(10240)
        source = np.arange(self.num_input)
        target = np.random.randint(0, self.num_hidden, self.num_input)

        self.syn12.connect(i=source, j=target)
        self.syn23.connect()

        self.mons['syn23'] = StateMonitor(self.syn23, ['X', 'w', 'c'], record=np.arange(
            self.num_hidden*self.num_output), name='syn23')

        self.net = Network(self.lif1,
                           self.lif2,
                           self.lif3,
                           self.syn12,
                           self.syn23)

        if self.wta:
            self.syn33.connect(condition='i != j')

        if self.monitor:
            self.net.add(self.mons)

        # Initial values
        self.lif1.v = params['v_initial']
        self.lif2.v = params['v_initial']
        self.lif3.v = params['v_initial']
        self.lif3.v_th = params['V_theta']
        self.lif3.g_e = 0*nS
        self.lif3.g_i = 0*nS
        self.syn12.w = 1
        self.syn23.c = params['c_initial']
        self.syn23.X = params['X_initial']
        self.syn23.delay = 'rand()*tau_r'

    def run(self, spk_input, duration, load_state=None, save_state=None):
        """Spike-driven synaptic plasticity unsupervised learning.

        Args:
            spk_input (TimedArray): The input stimulus.
            duration (TimedArray): The amount of simulation time to run for.
            load_state (string, optional): Name of the snapshot to load the state.
            save_state (string, optional): Name of the snapshot to save the state.

        Returns:
            list: Returns the list of monitors if option to monitor is True, otherwise None.
        """

        params['I'] = spk_input

        if not self.initialized:
            self.init_network()

        if load_state is not None:
            restore(load_state)

        self.net.run(duration=duration, report='stdout')

        if save_state is not None:
            store(save_state)

        if self.monitor:
            return self.mons
        else:
            return None


# A command-line interface for quick testing
if __name__ == '__main__':
    set_device('cpp_standalone', build_on_run=True, clean=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", help="enable Brian2 debug messages", action="store_true")
    parser.add_argument(
        "--monitor", help="enable spike/state monitors", action="store_true")
    parser.add_argument(
        "--openmp", help="enable openmp", action="store_true")
    args = parser.parse_args()
    if args.debug:
        BrianLogger.log_level_debug()
    if args.monitor:
        monitor = True
    else:
        monitor = False
    if args.openmp:
        prefs.devices.cpp_standalone.openmp_threads = multiprocessing.cpu_count()

    time_step = 10*ms
    duration = 1*second
    spk_input = TimedArray(
        (1.2e-2*np.random.rand(192, int(duration/time_step)).T)*nA, dt=time_step)

    model = WTA_SNN(192, 16, 36, wta=True, monitor=monitor)
    mons = model.run(spk_input, duration)

    if monitor:
        figure()
        title('input')
        brian_plot(mons['lif1'])
        figure()
        title('hidden')
        brian_plot(mons['lif2'])
        figure()
        title('output')
        brian_plot(mons['lif3'])
        show()
