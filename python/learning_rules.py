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
        "T",
        "pre_memory",
        "post_memory",
    )

    time_constant = NumberParam("time_constant", default=0.01, low=0, readonly=True)
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
        T,
        timer,
        time_constant,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.time_constant = time_constant

        self.sets = []
        self.incs = []
        self.reads = [weights, pre_filtered, post_filtered]
        self.updates = [delta, pre_memory, post_memory, mu, cov, T, timer]

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
    def T(self):
        return self.updates[5]

    @property
    def timer(self):
        return self.updates[6]

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
        T = signals[self.T]
        timer = signals[self.timer]

        def step_simss():
            pre = pre_filtered * dt
            post = post_filtered * dt
            if timer >= self.time_constant or timer == 0.0:
                timer[...] = dt
                pre_memory[...] = 0
                post_memory[...] = 0
                T[...] = 298.15
            else:
                pre_memory[...] *= timer
                post_memory[...] *= timer
                timer[...] += dt
            alpha = 2
            pre_memory[...] = (pre_memory + pre) / (timer / dt)
            post_memory[...] = (post_memory + post) / (timer / dt)
            drift = (np.exp(-alpha * np.abs(weights)) - np.exp(-alpha)) * np.outer(
                post - post_memory, pre - pre_memory
            )

            diffusion = (
                1e-3
                * np.exp(-1 / T)
                * np.random.normal(0, np.sqrt(np.exp(-1e-3 * drift)), drift.shape)
                * np.exp(-drift)
            )
            T[...] += -0.2 * T
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
    T = Signal(initial_value=298.15, name="T")
    timer = Signal(initial_value=0.0, name="timer")

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
            T,
            timer,
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
    model.sig[rule]["T"] = T
