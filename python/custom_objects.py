# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.dists import Choice, Uniform
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.numpy import clip, is_array_like


class SynapticSampling(nengo.learning_rules.LearningRuleType):
    """
    Synaptic Sampling learning rule.
    """

    modifies = "weights"
    probeable = (
        "pre_filtered",
        "post_filtered",
        "delta",
        "mu",
        "cov",
        "pre_memory",
        "post_memory",
    )

    time_constant = NumberParam("time_constant", default=0.005, low=0, readonly=True)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
        self,
        time_constant=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(0, size_in=0)
        self.time_constant = time_constant
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )


class SimSS(Operator):
    def __init__(
        self,
        pre_filtered,
        post_filtered,
        weights,
        delta,
        pre_memory,
        post_memory,
        mu,
        cov,
        time_constant,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.time_constant = time_constant

        self.sets = []
        self.incs = []
        self.reads = [weights, pre_filtered, post_filtered]
        self.updates = [delta, pre_memory, post_memory, mu, cov]

    @property
    def pre_filtered(self):
        return self.reads[1]

    @property
    def post_filtered(self):
        return self.reads[2]

    @property
    def weights(self):
        return self.reads[0]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_memory(self):
        return self.updates[1]

    @property
    def post_memory(self):
        return self.updates[2]

    @property
    def mu(self):
        return self.updates[3]

    @property
    def cov(self):
        return self.updates[4]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.weights}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        weights = signals[self.weights]
        delta = signals[self.delta]
        pre_memory = signals[self.pre_memory]
        post_memory = signals[self.post_memory]
        mu = signals[self.mu]
        cov = signals[self.cov]

        def step_simss():
            pre = pre_filtered * dt - pre_memory
            post = post_filtered * dt - post_memory
            pre_memory[...] = pre_filtered * dt
            post_memory[...] = post_filtered * dt
            drift = np.outer(post, pre)
            # mu[...] = weights + drift

            diffusion = (
                1e-3
                * np.random.normal(0, 1 / np.sqrt(1 + np.square(drift)), drift.shape)
                * np.exp(-drift)
            )
            delta[...] = drift + diffusion

        return step_simss


@Builder.register(SynapticSampling)
def build_ss(model, ss, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, ss.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, ss.post_synapse, post_activities)
    pre_memory = Signal(
        initial_value=np.zeros(pre_filtered.shape),
        shape=pre_filtered.shape,
        name="pre_memory",
    )
    post_memory = Signal(
        initial_value=np.zeros(post_filtered.shape),
        shape=post_filtered.shape,
        name="post_memory",
    )
    mu = Signal(shape=model.sig[conn]["weights"].shape, name="mu")
    cov = Signal(shape=model.sig[conn]["weights"].shape, name="cov")

    model.add_op(
        SimSS(
            pre_filtered,
            post_filtered,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            pre_memory,
            post_memory,
            mu,
            cov,
            ss.time_constant,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["pre_memory"] = pre_memory
    model.sig[rule]["post_memory"] = post_memory
    model.sig[rule]["mu"] = mu
    model.sig[rule]["cov"] = cov


class WTA(nengo.LIFRate):
    state = {
        "voltage": Uniform(low=0, high=1),
        "refractory_time": Choice([0]),
    }
    spiking = True

    min_voltage = NumberParam("min_voltage", high=0)

    def __init__(
        self, tau_rc=0.02, tau_ref=0.002, min_voltage=0, amplitude=1, initial_state=None
    ):
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            amplitude=amplitude,
            initial_state=initial_state,
        )
        self.min_voltage = min_voltage

    def make_state(self, n_neurons, rng=np.random, dtype=None):
        state = super().make_state(n_neurons, rng=rng, dtype=dtype)
        state["rng"] = rng
        return state

    def step(self, dt, J, output, voltage, refractory_time, rng):
        # look these up once to avoid repeated parameter accesses
        tau_rc = self.tau_rc
        min_voltage = self.min_voltage

        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = np.clip((dt - refractory_time), 0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        I = np.exp(voltage)
        output[:] = (
            spiked_mask
            * (self.amplitude / dt)
            * rng.poisson(I / I.sum(), output.size)
            * np.sign(J)
        )

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < min_voltage] = min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike
