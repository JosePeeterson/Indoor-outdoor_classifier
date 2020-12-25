#!/bin/bash

start_port=12345
port_id_offset=100
for i in {0..8}
do
   num=$((port_id_offset*i+start_port))
   roscore -p $num &
   sleep 3
done


for i in {0..8}
do
   num=$((port_id_offset*i+start_port))
   export ROS_MASTER_URI="http://localhost:$num"
   rosrun stage_ros_add_pose_and_crash stageros  -g worlds/scene_open.world &
   echo "http://localhost:$num"
   sleep 3
done
