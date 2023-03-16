source /opt/ros/humble/setup.bash
colcon build --symlink-install
bash -c "echo 'source /workspace/install/setup.bash' > ~/.bashrc"
source ~/.bashrc
pip3 install nengo nengo-extras
ros2 launch active_touch mujoco.launch.py
