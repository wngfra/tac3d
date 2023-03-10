# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.operator import Operator
from nengo.learning_rules import build_or_passthrough, get_post_ens
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam


class MSTDP(nengo.learning_rules.LearningRuleType):
    """
    Mirrored STDP learning rule.
    """

    modifies = "weights"
    probeable = ("theta", "pre_filtered", "post_filtered", "delta")

    learning_rate = NumberParam("learning_rate", low=0, readonly=True, default=1e-6)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    theta_synapse = SynapseParam(
        "theta_synapse", default=Lowpass(tau=1.0), readonly=True
    )

    def __init__(
        self,
        learning_rate=Default,
        pre_synapse=Default,
        post_synapse=Default,
        theta_synapse=Default,
    ):
        super().__init__(learning_rate, size_in=0)
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        self.theta_synapse = theta_synapse


class SimMSTDP(Operator):
    def __init__(
        self, pre_filtered, post_filtered, theta, delta, learning_rate, tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta]
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
    def theta(self):
        return self.reads[2]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.delta}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        def step_simmstdp():
            delta[...] = np.outer(
                alpha * post_filtered * (post_filtered - theta), pre_filtered
            )
        # TODO implement the mstdp update rule
        dw = np.zeros((pre.size_out, post.size_out))
        for i in range(pre.size_out):
            for j in range(post.size_out):
                if i != j:
                    t_diff = post.times - pre.times[i]
                    dw[i, j] = self.learning_rate * (
                        np.sum(t_diff[t_diff > 0]) - np.sum(t_diff[t_diff < 0])
                    )
        return dw

        return step_simmstdp


@Builder.register(MSTDP)
def build_mstdp(model, mstdp, rule):
    """
    Builds a `.MSTDP` object into a model.

    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimMSTDP` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    mstdp : MSTDP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.BCM` instance.
    """
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, mstdp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, mstdp.post_synapse, post_activities)
    theta = build_or_passthrough(model, mstdp.theta_synapse, post_activities)

    model.add_op(
        SimMSTDP(
            pre_filtered,
            post_filtered,
            theta,
            model.sig[rule]["delta"],
            learning_rate=mstdp.learning_rate,
        )
    )

    # expose these for probes
    model.sig[rule]["theta"] = theta
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
