# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

from Environment import Environment
from IKSolver import IKSolver
import mujoco
import numpy as np

if __name__ == "__main__":
    env = Environment("models/scene.xml")
    ik = IKSolver(env.m, env.d)

    site_name = "attachment_site"
    target_xmat = ik.d.site(site_name).xmat
    target_quat = np.empty(4, dtype=ik.d.qpos.dtype)
    mujoco.mju_mat2Quat(target_quat, target_xmat)

    joint_names = ["joint{}".format(i + 1) for i in range(7)]
    target_pos = ik.d.site(site_name).xpos

    def callback(m, d):
        target_pos[2] -= 1e-3
        result = ik.qpos_from_site_xpos(
            site_name,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=joint_names,
        )
        if np.mean(d.sensordata) < 1.:
            d.ctrl = result.qpos[:10]

    env.control(callback)
