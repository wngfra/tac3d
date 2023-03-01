source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
pip3 install nengo
ros2 launch active_touch mujoco.launch.py
