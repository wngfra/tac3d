# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import mujoco
import threading
from mujoco import viewer


class TactileExploration:
    def __init__(self, xml_path):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)

        # Launch the GUI thread
        self._viewer_thread = threading.Thread(
            target=viewer.launch, args=(self._model,)
        )
        self._viewer_thread.start()

        # Launch the simulate thread
        self.simulate()

    def __del__(self):
        self._viewer_thread.join()

    @property
    def sensordata(self):
        return self._data.sensordata

    @property
    def time(self):
        return self._data.time

    def reset(self, key_name):
        # TODO implement reset from keyframe
        key_id = mujoco.mj_named2id(self._model, mujoco.mjOBJ_KEY, key_name)
        if key_id > 0:
            mujoco.mju_copy()

    def step(self, controller):
        mujoco.set_mjcb_control(controller)
        mujoco.mj_step(self._model, self._data)

    def simulate(self, duration=None):
        def ctrl(m, d):
            d.ctrl[1] = np.sin(d.time) * 0.5

        print(self.sensordata)
        self.step(ctrl)
