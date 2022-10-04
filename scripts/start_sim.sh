#!/bin/bash

# sed -i 's/highResDisplay = -1/highResDisplay = 2/g' ${COPPELIASIM_ROOT_DIR}/system/usrset.txt
cp /shared/scripts/CMakeLists.txt /shared/scripts/package.xml src/ros2_packages/sim_ros2_interface/
cp /shared/scripts/interfaces.txt src/ros2_packages/sim_ros2_interface/meta/
wget https://github.com/CoppeliaRobotics/simExtPovRay/blob/master/binaries/ubuntu20_04/libsimExtPovRay.so -P ${COPPELIASIM_ROOT_DIR}/
source install/setup.bash
colcon build --symlink-install --packages-select control_interfaces
source install/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
bash ${COPPELIASIM_ROOT_DIR}/coppeliaSim.sh /shared/scenes/tactile_exploration.ttt