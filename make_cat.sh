#!/bin/bash
echo "making catkin"
if [ -d catkin_ws ]; then rm -rf catkin_ws; fi
mkdir -p catkin_ws/src
cp -r stage_ros-add_pose_and_crash catkin_ws/src
cd catkin_ws
catkin_make
source devel/setup.bash
cd ..
find . -name "*.pyc" -exec rm -f {} \;
cp configs/main_config_default.yaml configs/main_config.yaml 