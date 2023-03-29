# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam


class MirroredSTDP(nengo.learning_rules.LearningRuleType):
    """
    Mirrored STDP learning rule.
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    time_constant = NumberParam("time_constant", low=0, readonly=True, default=0.005)
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


class SimMSTDP(Operator):
    def __init__(self, pre_filtered, post_filtered, delta, learning_rate, time_constant, tag=None):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.time_constant = time_constant
        
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.delta}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simmstdp():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - 0.1), pre_filtered
            )
            """
            delta_t = data[0] - data[1]
            stdp_update = np.where(delta_t > 0, np.exp(-delta_t / self.time_constant), -np.exp(delta_t / self.time_constant))
            data[2] += self.learning_rate * stdp_update
            """

        return step_simmstdp


@Builder.register(MirroredSTDP)
def build_mstdp(model, mstdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, mstdp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, mstdp.post_synapse, post_activities)

    model.add_op(
        SimMSTDP(
            pre_filtered,
            post_filtered,
            model.sig[rule]["delta"],
            learning_rate=mstdp.learning_rate,
            time_constant=mstdp.time_constant,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    
    """
    pre = model.sig[rule.pre_obj]['out']
    post = model.sig[rule.post_obj]['out']
    weights = model.sig[rule]['weights']

    # Create the operator
    op = Operator.Set([pre, post, weights], None, mirrored_stdp.make_step, tag=f"MirroredSTDP {rule}")
    model.add_op(op)
    
    """