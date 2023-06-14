# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import nengo
import numpy as np
from nengo.builder.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens, build_or_passthrough
from nengo.builder.operator import Operator, Reset
from nengo.builder.signal import Signal
from nengo.dists import Choice, Uniform
from nengo.params import Default, NumberParam, TupleParam
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
        "pre_mean",
        "post_mean",
    )

    time_constant = NumberParam("time_constant", default=0.02, low=0, readonly=True)
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
        pre_mean,
        post_mean,
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
        self.updates = [delta, pre_mean, post_mean, mu, cov, T, timer]

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
    def pre_mean(self):
        return self.updates[1]

    @property
    def post_mean(self):
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
        pre_mean = signals[self.pre_mean]
        post_mean = signals[self.post_mean]
        mu = signals[self.mu]
        cov = signals[self.cov]
        T = signals[self.T]
        timer = signals[self.timer]

        alpha = 2
        beta = np.exp(-alpha)
        a = 1 / (1 - beta)
        b = beta / (beta - 1)

        def step_simss():
            pre = pre_filtered * dt
            post = post_filtered * dt
            pre_mean[...] = (pre_mean * timer + pre) / (timer / dt + 1)
            post_mean[...] = (post_mean * timer + post) / (timer / dt + 1)
            timer[...] += dt

            if timer >= self.time_constant:
                drift = (a * np.exp(-alpha * np.abs(weights)) + b) * np.outer(
                    post_mean, pre_mean
                )
                wtaIdx = np.argmax(drift, axis=0)
                drift = -np.abs(drift)
                drift[wtaIdx, :] *= -1

                # Flush synaptic cache/memory
                pre_mean[...] = 0
                post_mean[...] = 0
                T[...] = 298.15
                timer[...] = 0

                diffusion = (
                    1e-4
                    * np.exp(-1 / T)
                    * np.random.normal(0, np.sqrt(np.exp(-1e-3 * drift)), drift.shape)
                )
                T[...] += -0.2 * T
                delta[...] = drift

        return step_simss


@Builder.register(SynapticSampling)
def build_ss(model, ss, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, ss.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, ss.post_synapse, post_activities)
    pre_mean = Signal(
        initial_value=np.zeros(pre_filtered.shape),
        shape=pre_filtered.shape,
        name="pre_mean",
    )
    post_mean = Signal(
        initial_value=np.zeros(post_filtered.shape),
        shape=post_filtered.shape,
        name="post_mean",
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
            pre_mean,
            post_mean,
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
    model.sig[rule]["pre_mean"] = pre_mean
    model.sig[rule]["post_mean"] = post_mean
    model.sig[rule]["mu"] = mu
    model.sig[rule]["cov"] = cov
    model.sig[rule]["T"] = T


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
        "c",
        "delta",
    )

    J_C = NumberParam("J_C", default=1, low=0, readonly=True)
    tau_c = NumberParam("tau_c", default=0.06, low=0, readonly=True)
    X_limit = TupleParam("X_limit", default=(0, 1.0), readonly=True)
    X_coeff = TupleParam("X_coeff", default=(0.1, 0.1, 3.5, 3.5), readonly=True)
    theta_coeff = TupleParam("theta_coeff", default=(12, 3, 4, 2), readonly=True)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)

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
        c,
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
        self.reads = [weights, pre_filtered, post_filtered, post_voltage, learning_mode]
        self.updates = [delta, X, c]

    @property
    def weights(self):
        return self.reads[0]

    @property
    def pre_filtered(self):
        return self.reads[1]

    @property
    def post_filtered(self):
        return self.reads[2]

    @property
    def post_voltage(self):
        return self.reads[3]

    @property
    def learning_mode(self):
        return self.reads[4]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def X(self):
        return self.updates[1]

    @property
    def c(self):
        return self.updates[2]

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
        c = signals[self.c]

        X_min, X_max = self.X_limit
        a, b = np.multiply(self.X_coeff[:2], X_max)
        alpha, beta = np.multiply(self.X_coeff[2:], X_max)
        theta_hup, theta_lup, theta_hdown, theta_ldown = np.multiply(
            self.theta_coeff, self.J_C
        )
        theta_X = 0.5 * X_max
        theta_V = 0.8

        def step_simsdsp():
            if self.learning_mode:
                t_pre = pre_filtered[pre_filtered > 0]
                if len(t_pre) > 0:
                    mask_Va = post_voltage > theta_V
                    mask_up = np.logical_and(
                        c[mask_Va] > theta_lup, c[mask_Va] < theta_hup
                    )

                    mask_Vb = post_voltage <= theta_V
                    mask_down = np.logical_and(
                        c[mask_Vb] > theta_ldown, c[mask_Vb] < theta_hdown
                    )
                    try:
                        X[mask_Va][mask_up] += a
                        X[mask_Vb][mask_down] -= b
                    except _:
                        pass

                delta[X > theta_X] += alpha * dt
                delta[X <= theta_X] -= beta * dt
                c[...] += (-1 / self.tau_c * c + self.J_C * post_filtered) * dt

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

    X = Signal(shape=weights.shape, name="X")
    c = Signal(shape=weights.shape, name="c")

    model.add_op(
        SimSDSP(
            pre_filtered,
            post_filtered,
            post_voltage,
            weights,
            model.sig[rule]["delta"],
            X,
            c,
            learning_mode,
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
    model.sig[rule]["delta"] = model.sig[rule]["delta"]
    model.sig[rule]["X"] = X
    model.sig[rule]["c"] = c
    model.sig[rule]["learning_mode"] = learning_mode
