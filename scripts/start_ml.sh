#!/bin/bash

colcon build --symlink-install --packages-select control_interfaces
source install/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash