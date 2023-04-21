# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.params import Default, NumberParam
from nengo.synapses import SynapseParam


class SynapticSampling(nengo.learning_rules.LearningRuleType):
    """
    Synaptic Sampling learning rule.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    time_constant = NumberParam("time_constant", low=0, readonly=True, default=0.1)
    pre_synapse = SynapseParam("pre_synapse", default=None, readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
        self,
        learning_rate=Default,
        time_constant=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(learning_rate, size_in=0)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.time_constant = time_constant


class SimSS(Operator):
    def __init__(
        self,
        timer,
        pre_filtered,
        post_filtered,
        delta,
        likelihood,
        learning_rate,
        time_constant,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.time_constant = time_constant

        self.sets = []
        self.incs = [timer]
        self.reads = [pre_filtered, post_filtered]
        self.updates = [delta, likelihood]

    @property
    def timer(self):
        return self.incs[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def likelihood(self):
        return self.updates[1]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.delta}"

    def make_step(self, signals, dt, rng):
        timer = signals[self.timer]
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        likelihood = signals[self.likelihood]

        def step_simss():
            # TODO: implement synaptic sampling
            timer[...] += dt
            a_i = pre_filtered * dt
            a_j = post_filtered * dt
            likelihood[...] += np.outer(a_j, a_i)
            dW = np.random.normal(0, dt, size=delta.shape)
            delta[...] = np.random.normal(0, 1e-3, size=delta.shape) * likelihood + dW

            if timer > self.time_constant:
                timer[...] = 0
                # likelihood[...] = np.zeros(likelihood.shape)

        return step_simss


@Builder.register(SynapticSampling)
def build_ss(model, ss, rule):
    conn = rule.connection
    timer = Signal(initial_value=0.0, name="{rule}.timer")
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, ss.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, ss.post_synapse, post_activities)
    likelihood = Signal(
        initial_value=np.zeros(model.sig[rule]["delta"].shape),
        name="{rule}.likelihood",
    )

    model.add_op(
        SimSS(
            timer,
            pre_filtered,
            post_filtered,
            model.sig[rule]["delta"],
            likelihood,
            learning_rate=ss.learning_rate,
            time_constant=ss.time_constant,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
