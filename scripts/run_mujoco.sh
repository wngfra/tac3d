source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch mujoco_ros2 mujoco.launch.py
