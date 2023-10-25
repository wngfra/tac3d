source /opt/ros/humble/setup.bash
colcon build --symlink-install
source /workspace/install/setup.bash
pip3 install -q nengo nengo-extras
ros2 launch active_touch mujoco.launch.py
