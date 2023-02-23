# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import weakref
import threading

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mujoco import viewer
from matplotlib.animation import FuncAnimation
from IKSolver import IKSolver


class Environment:
    def __init__(self, xml_path, animate_sensor=False):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._ik = IKSolver(self.m, self.d)
        self.reset(0)

        # Launch the GUI thread
        self._viewer_thread = threading.Thread(
            target=viewer.launch,
            args=(
                self._model,
                self._data,
            ),
        )
        self._viewer_thread.start()

        # Launch the simulate thread
        self._simulate_thread = threading.Thread(target=self.simulate, args=(1,))
        self._simulate_thread.start()

        # Start sensordata visualization
        if animate_sensor:
            fig, ax = plt.subplots(1, 1)
            self._im = ax.imshow(self.sensordata.reshape((15, 15)))
            self._animation = FuncAnimation(
                fig, self._animate, interval=200, repeat=False
            )
            plt.show()

    def __del__(self):
        self._simulate_thread.join()
        self._viewer_thread.join()
        del self._model

    @property
    def m(self) -> weakref.ref:
        return weakref.ref(self._model)

    @property
    def d(self) -> weakref.ref:
        return weakref.ref(self._data)

    @property
    def sensordata(self):
        return self._data.sensordata

    @property
    def time(self):
        return self._data.time

    def _animate(self, frame):
        self._im.set_array(self.sensordata.reshape((15, 15)))
        return [self._im]

    def reset(self, key_id: int):
        mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        mujoco.mj_forward(self._model, self._data)

    def step(self, controller=None):
        if controller:
            mujoco.set_mjcb_control(controller)
        mujoco.mj_step(self._model, self._data)

    def simulate(self, duration=None):
        joint_names = ["joint{}".format(i + 1) for i in range(7)]
        site_name = "attachment_site"

        def controller(m, d):
            target_pos = self._data.site(site_name).xpos
            result = self._ik.qpos_from_site_xpos(
                site_name=site_name,
                target_pos=target_pos,
                joint_names=joint_names
            )

        while self.time <= 5.0:
            self.step()
