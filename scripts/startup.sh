#!/bin/bash

source /workspace/install/setup.bash
# sed -i 's/highResDisplay = -1/highResDisplay = 2/g' ${COPPELIASIM_ROOT_DIR}/system/usrset.txt
bash ${COPPELIASIM_ROOT_DIR}/coppeliaSim.sh /shared/scenes/tactile_exploration.ttt
