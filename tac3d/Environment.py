# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0
import weakref
import threading

import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mujoco import viewer
from matplotlib.animation import FuncAnimation


class Environment:
    def __init__(self, xml_path: str, animate_sensor=False):
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
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

        # Start sensordata visualization
        if animate_sensor:
            fig, ax = plt.subplots(1, 1)
            self._im = ax.imshow(self.sensordata.reshape((15, 15)))
            self._animation = FuncAnimation(
                fig, self._animate, interval=200, repeat=False
            )
            plt.show()

    def __del__(self):
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

    def control(self, controller_callback):
        # Install controller callback
        if controller_callback:
            mujoco.set_mjcb_control(controller_callback)

    def finish(self):
        mujoco.set_mjcb_control(None)
