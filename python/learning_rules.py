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
    X_limit = TupleParam("X_limit", default=(0, 1.0), readonly=True)
    X_coeff = TupleParam("X_coeff", default=(0.1, 0.1, 3.5, 3.5), readonly=True)
    theta_coeff = TupleParam("theta_coeff", default=(13, 3, 4, 3), readonly=True)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.01), readonly=True)
    post_synapse = SynapseParam(
        "post_synapse", default=None, readonly=True
    )

    def __init__(
        self,
        J_C=Default,
        tau_c=Default,
        X_limit=Default,
        X_coeff=Default,
        theta_coeff=Default,
        pre_synapse=Default,
        post_synapse=Default,
    ):
        super().__init__(0, size_in=0)
        self.J_C = J_C
        self.tau_c = tau_c
        self.X_limit = X_limit
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
        delta,
        X,
        C,
        sum_post,
        learning_mode,
        J_C,
        tau_c,
        X_limit,
        X_coeff,
        theta_coeff,
        tag=None,
    ):
        super().__init__(tag=tag)
        self.J_C = J_C
        self.tau_c = tau_c
        self.X_limit = X_limit
        self.X_coeff = X_coeff
        self.theta_coeff = theta_coeff

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, post_voltage, learning_mode]
        self.updates = [weights, delta, X, C, sum_post]

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
    def delta(self):
        return self.updates[1]

    @property
    def X(self):
        return self.updates[2]

    @property
    def C(self):
        return self.updates[3]

    @property
    def sum_post(self):
        return self.updates[4]

    @property
    def _descstr(self):
        return f"pre={self.pre_filtered}, post={self.post_filtered} -> {self.weights}"

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        post_voltage = signals[self.post_voltage]
        weights = signals[self.weights]
        delta = signals[self.delta]
        X = signals[self.X]
        C = signals[self.C]
        sum_post = signals[self.sum_post]

        X_min, X_max = self.X_limit
        a, b = np.multiply(self.X_coeff[:2], X_max) * dt
        alpha, beta = np.multiply(self.X_coeff[2:], X_max) * dt
        theta_hup, theta_lup, theta_hdown, theta_ldown = np.multiply(
            self.theta_coeff, self.J_C
        )
        theta_X = 0.5 * X_max
        theta_V = 0.8

        def step_simsdsp():
            sum_post[...] += self.J_C * post_filtered * dt
            C[...] += (
                -C * dt / self.tau_c
                + np.tile(sum_post[:, np.newaxis], C.shape[1]) * dt
            )

            # Pre-synaptic update
            mask_Va = np.tile((post_voltage > theta_V)[:, np.newaxis], weights.shape[1])
            mask_Xa = (C > theta_lup) & (C < theta_hup)
            mask_Vb = np.tile(
                (post_voltage <= theta_V)[:, np.newaxis], weights.shape[1]
            )
            mask_Xb = (C > theta_ldown) & (C < theta_hdown)
            np.putmask(X, mask_Va & mask_Xa, np.clip(X + a, X_min, X_max))
            np.putmask(X, mask_Vb & mask_Xb, np.clip(X - b, X_min, X_max))

            np.putmask(X, X >theta_X, np.clip(X + alpha, X_min, X_max))
            np.putmask(X, X <= theta_X, np.clip(X - beta, X_min, X_max))

            np.putmask(weights, X > theta_X, X_max)
            np.putmask(weights, X <= theta_X, X_min)

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
    delta = model.sig[rule]["delta"]

    X = Signal(initial_value=np.random.random(weights.shape), name="X")
    C = Signal(initial_value=2 * np.ones(weights.shape), name="C")
    sum_post = Signal(initial_value=np.zeros(post_filtered.shape), name="sum_post")

    model.add_op(
        SimSDSP(
            pre_activities,
            post_activities,
            post_voltage,
            weights,
            delta,
            X,
            C,
            sum_post,
            model.sig[rule]["in"],
            sdsp.J_C,
            sdsp.tau_c,
            sdsp.X_limit,
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
