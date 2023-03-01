# Copyright 2023 wngfra.
# SPDX-License-Identifier: Apache-2.0

bash -c "echo 'source /workspace/install/setup.bash' > ~/.bashrc"
source ~/.bashrc
pip3 install nengo nengo-extras
bash