6  # Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator, Reset
from nengo.builder.signal import Signal
from nengo.params import Default, NumberParam, TupleParam
from nengo.synapses import LinearFilter, Lowpass, SynapseParam


class SDSP(nengo.learning_rules.LearningRuleType):
    """
    Spike-Dependent Synaptic Plasticity (SDSP) learning rule.

    Parameters
    ----------
    learning_mode : bool, optional (Default: True)
    """

    modifies = "weights"
    probeable = (
        "learning_mode",
        "pre_filtered",
        "post_filtered",
        "X",
        "C",
    )

    J_C = NumberParam("J_C", default=1, low=0, readonly=True)
    tau_c = NumberParam("tau_c", default=0.06, low=0, readonly=True)
    X_min = NumberParam("X_min", default=0.0, low=0, readonly=True)
    X_max = NumberParam("X_max", default=1.0, low=0, readonly=True)
    X_coeff = TupleParam("X_coeff", default=(0.1, 0.1, 3.5, 3.5), readonly=True)
    theta_coeff = TupleParam("theta_coeff", default=(13, 3, 4, 3), readonly=True)
    pre_synapse = SynapseParam("pre_synapse", default=None, readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

    def __init__(
        self,
        J_C=Default,
        tau_c=Default,
        X_min=Default,
        X_max=Default,
        X_coeff=Default,
        theta_coeff=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(0, size_in=0)
        self.J_C = J_C
        self.tau_c = tau_c
        self.X_min = X_min
        self.X_max = X_max
        self.X_coeff = X_coeff
        self.theta_coeff = theta_coeff
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )


class SimSDSP(Operator):
    def __init__(
        self,
        pre_filtered,
        post_filtered,
        post_voltage,
        weights,
        X,
        C,
        sum_post,
        learning_mode,
        J_C,
        tau_c,
        X_min,
        X_max,
        X_coeff,
        theta_coeff,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.J_C = J_C
        self.tau_c = tau_c
        self.X_min, self.X_max = X_min, X_max
        self.a, self.b, self.alpha, self.beta = np.multiply(X_coeff, X_max)
        (
            self.theta_hup,
            self.theta_lup,
            self.theta_hdown,
            self.theta_ldown,
        ) = np.multiply(theta_coeff, J_C)
        self.theta_X = 0.5 * self.X_max
        self.theta_V = 0.8

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, post_voltage, learning_mode]
        self.updates = [weights, X, C, sum_post]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def post_voltage(self):
        return self.reads[2]

    @property
    def learning_mode(self):
        return self.reads[3]

    @property
    def weights(self):
        return self.updates[0]

    @property
    def X(self):
        return self.updates[1]

    @property
    def C(self):
        return self.updates[2]

    @property
    def sum_post(self):
        return self.updates[3]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.weights}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        post_voltage = signals[self.post_voltage]
        weights = signals[self.weights]
        X = signals[self.X]
        C = signals[self.C]
        sum_post = signals[self.sum_post]
        
        np.putmask(X, weights > 0, self.X_max)

        def step_simsdsp():
            # Update calcium variable
            sum_post[...] += self.J_C * post_filtered * dt
            sum_post[...] += -sum_post[...] * dt
            C[...] += (
                -C / self.tau_c + np.tile(sum_post[:, np.newaxis], C.shape[1])
            ) * dt

            # Pre-synaptic mask
            mask_pre = np.tile((pre_filtered > 0)[:, np.newaxis], weights.shape[0]).T

            # LTP
            mask_V = np.tile(
                (post_voltage > self.theta_V)[:, np.newaxis], weights.shape[1]
            )
            mask_C = (C > self.theta_lup) & (C < self.theta_hup)
            np.putmask(
                X, mask_pre & mask_V & mask_C, np.clip(X + self.a, self.X_min, self.X_max)
            )
            np.putmask(
                X,
                ~mask_pre | (mask_pre & mask_V & ~mask_C),
                np.clip(X + self.alpha * dt, self.X_min, self.X_max),
            )
            # LTD
            mask_C = (C > self.theta_ldown) & (C < self.theta_hdown)
            np.putmask(
                X, mask_pre & ~mask_V & mask_C, np.clip(X - self.b, self.X_min, self.X_max)
            )
            np.putmask(
                X,
                ~mask_pre | (mask_pre & ~mask_V & ~mask_C),
                np.clip(X - self.beta * dt, self.X_min, self.X_max),
            )

            # Update weights
            np.putmask(weights, mask_pre & (X > self.theta_X), 1)
            np.putmask(weights, mask_pre & (X <= self.theta_X), 0)

        return step_simsdsp


@Builder.register(SDSP)
def build_sdsp(model, sdsp, rule):
    conn = rule.connection

    # Add input to learning rule
    learning_mode = Signal(initial_value=True, name="learning_mode")
    model.add_op(Reset(learning_mode))
    model.sig[rule]["in"] = learning_mode

    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, sdsp.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, sdsp.post_synapse, post_activities)
    post_voltage = model.sig[get_post_ens(conn).neurons]["voltage"]
    weights = model.sig[conn]["weights"]

    X = Signal(initial_value=np.zeros(weights.shape), name="X")
    C = Signal(initial_value=2 * np.ones(weights.shape), name="C")
    sum_post = Signal(initial_value=np.zeros(post_filtered.shape), name="sum_post")

    model.add_op(
        SimSDSP(
            pre_activities,
            post_activities,
            post_voltage,
            weights,
            X,
            C,
            sum_post,
            model.sig[rule]["in"],
            sdsp.J_C,
            sdsp.tau_c,
            sdsp.X_min,
            sdsp.X_max,
            sdsp.X_coeff,
            sdsp.theta_coeff,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["post_voltage"] = post_voltage
    model.sig[rule]["X"] = X
    model.sig[rule]["C"] = C
    model.sig[rule]["learning_mode"] = learning_mode
