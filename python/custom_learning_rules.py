# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam


class SynapticSampling(nengo.learning_rules.LearningRuleType):
    """
    Synaptic Sampling learning rule.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered")

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
        self, pre_filtered, post_filtered, weights, post_memory, time_constant, tag=None
    ):
        super().__init__(tag=tag)
        self.time_constant = time_constant

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
        self.updates = [weights, post_memory]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.updates[0]

    @property
    def post_memory(self):
        return self.updates[1]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.weights}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        weights = signals[self.weights]
        post_memory = signals[self.post_memory]

        def step_simss():
            delta = (
                post_filtered[...]*dt - post_memory[...] / self.time_constant * dt
            )
            post_memory[...] += delta
            mu = weights[...] + np.outer(post_filtered[...]*dt, pre_filtered[...]*dt)*1e-6
            cov = np.outer(np.exp(-delta), pre_filtered[...]*dt)*1e-2
            weights[...] = np.random.normal(mu, cov, size=mu.shape)
            weights[...] = np.clip(weights[...], 0, 1)
            
        return step_simss


@Builder.register(SynapticSampling)
def build_ss(model, ss, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, ss.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, ss.post_synapse, post_activities)
    post_memory = Signal(
        initial_value=np.zeros(post_filtered.shape),
        shape=post_filtered.shape,
        name="post_memory",
    )

    model.add_op(
        SimSS(
            pre_filtered,
            post_filtered,
            model.sig[conn]["weights"],
            post_memory,
            ss.time_constant,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
