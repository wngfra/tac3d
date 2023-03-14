# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

bash -c "echo 'source /workspace/install/setup.bash' > ~/.bashrc"
source ~/.bashrc
pip3 install nengo nengo-extras
# ros2 run active_touch tactile_encoding --ros-args -r __ns:=/tac3d
bash