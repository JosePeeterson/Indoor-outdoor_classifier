import os
import random

import numpy as np
import rospy
import tf
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from visualization_msgs.msg import Marker

from env.base import StageBase

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


class StageWorld(StageBase):
    def __init__(self, beam_num, config, config_name, robot, teb, stageros=True):

        super(StageWorld, self).__init__(beam_num, 0, 1)

        # for scenes
        self.marker = self.default_marker()
        self.stageros = stageros
        self.direction = 1
        self.step_count = 1
        self.data_full = []
        crash_topic = ''
        object_topic = ''
        self.obstacles = None
        self.robot = robot
        self.teb = teb

        # -----------Publisher and Subscriber-------------
        if stageros:
            cmd_vel_topic = '/robot_0/cmd_vel'
            cmd_pose_topic = '/robot_0/cmd_pose'
            laser_topic = '/robot_0/base_scan'
            odom_topic = '/robot_0/odom'
            marker_topic = '/visualization_marker'
            crash_topic = 'robot_0/is_crashed'
            pose_topic = 'robot_0/base_pose_ground_truth'
        else:
            cmd_vel_topic = '/cmd_vel'
            cmd_pose_topic = '/cmd_pose'
            odom_topic = '/odometry/filtered'
            marker_topic = '/visualization_marker'
            object_topic = '/gazebo/set_model_state'
            if self.robot == 'droc':
                laser_topic = '/scan'
                pose_topic = '/droc/pose'
            else:
                laser_topic = '/front/scan'
                pose_topic = '/jackal/pose'

        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan,
                                          self.laser_scan_callback)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.odometry_callback)
        self.vis_marker = rospy.Publisher(marker_topic, Marker, queue_size=10)
        self.cmd_goal = rospy.Publisher('/robot_1/cmd_pose',
                                        Pose,
                                        queue_size=10)
        if stageros:
            self.check_crash = rospy.Subscriber(crash_topic, Int8,
                                                self.crash_callback)
            self.pose_sub = rospy.Subscriber(pose_topic, Odometry,
                                             self.ground_truth_callback)
        else:
            self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped,
                                             self.ground_truth_callback_gazebo)
            self.cmd_obj = rospy.Publisher(object_topic,
                                           ModelState,
                                           queue_size=10)
        self.set_obstacle_class(config, config_name, stageros)
        if not stageros:
            self.spawn_all()

        # # Wait until the first callback
        while self.scan is None or self.speed is None or self.state is None:
            pass

        rospy.sleep(1.)

    def generate_random_pose(self):
        pass

    def spawn_all(self):
        if not self.teb:
            self.spawn_goal()

    @staticmethod
    def spawn_goal():
        print('spawning goal')
        spawn_cmd = 'rosrun gazebo_ros spawn_model -file model/gazebo_models/goal.urdf -urdf -z 3 -model my_goal'
        os.system(spawn_cmd)

    @staticmethod
    def default_marker():
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time(0)  # rospy.Time.now()
        marker.ns = "marker1"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.lifetime = rospy.Duration(0)
        return marker

    def ground_truth_callback_gazebo(self, poseStamped):
        Quaternions = poseStamped.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state_GT = [
            poseStamped.pose.position.x, poseStamped.pose.position.y, Euler[2]
        ]

    def odometry_callback(self, odometry):
        super(StageWorld, self).odometry_callback(odometry)
        if not self.stageros:
            self.speed_GT = self.speed

    def check_laser_scan(self):
        obs = self.scan
        obs_less_t1 = np.less(obs, 0.3)
        if obs_less_t1.any():
            print "Laser-scan Collision detected: %.3f" % (min(obs))
            self.ready = False
            return True, min(obs)
        return False, min(obs)

    def obstacle_crash_check(self, is_crash, x, y):
        is_crash_1, min_dist1 = self.obstacles.is_robot_crash(x, y)
        is_crash_2, min_dist2 = self.check_laser_scan()
        return is_crash or is_crash_1 or is_crash_2

    def control_vel(self, action):
        super(StageWorld, self).control_vel(action)
        self.obstacles.step()
        self.step_count += 1
        self.log_data()

    def log_data(self):
        step_data = self.obstacles.box_pose.copy()
        step_data['pose_robot'] = self.get_self_stateGT()
        step_data['step_count'] = self.step_count
        step_data['ep'] = self.episode
        self.data_full.append(step_data)

    def generate_random_goal(self):
        [x, y, _] = self.get_self_stateGT()
        x_cor, y_cor = self.get_goal_point(x)
        self.goal_pose = [x_cor, y_cor]
        self.direction = -1

        # Markers
        self.marker.pose.position.x = self.goal_pose[0]
        self.marker.pose.position.y = self.goal_pose[1]
        self.vis_marker.publish(self.marker)
        if not self.teb:
            self.visualise_goal()
        self.ready = False

        # Stats
        self.init_pose = [x, y]
        self.prev_pose = [x, y]
        self.total_dist = 0
        self.speed_list = []
        return self.goal_pose

    def visualise_goal(self):
        if self.stageros:
            self.control_pose(self.cmd_goal, self.goal_pose + [0])
            rospy.sleep(0.1)
        else:
            move_cmd = ModelState()
            move_cmd.model_name = 'my_goal'
            move_cmd.pose.position.x = self.goal_pose[0]
            move_cmd.pose.position.y = self.goal_pose[1]
            move_cmd.reference_frame = 'world'
            self.cmd_obj.publish(move_cmd)
            rospy.sleep(0.1)

    def reset_pose(self):
        if self.obstacles.cfg.restart_episode:
            if self.stageros:
                print 'Reseting pose'
                [x, y, theta] = self.get_self_stateGT()
                x_cord, y_cord = self.get_start_point()
                while not (abs(x - x_cord) < 0.01 and abs(y - y_cord) < 0.01):
                    self.control_pose(self.cmd_pose, [x_cord, y_cord, 0])
                    rospy.sleep(0.1)
                    [x, y, theta] = self.get_self_stateGT()

                [v, w] = self.get_self_speedGT()
                while not (abs(v) < 0.01 and abs(w) < 0.05):
                    self.cmd_vel.publish(Twist())  # stop the robot
                    rospy.sleep(0.1)
                    [v, w] = self.get_self_speedGT()

                rospy.sleep(0.1)
            else:
                print('Resetting pose')
                move_cmd = ModelState()
                move_cmd.model_name = self.robot
                move_cmd.reference_frame = 'world'

                x_cord, y_cord = self.get_start_point()
                move_cmd.pose.position.x = x_cord
                move_cmd.pose.position.y = y_cord
                move_cmd.pose.position.z = 0
                move_cmd.pose.orientation.x = 0
                move_cmd.pose.orientation.y = 0
                move_cmd.pose.orientation.z = 0
                move_cmd.pose.orientation.w = 0

                [x, y, theta] = self.get_self_stateGT()
                while not (abs(x - x_cord) < 0.01 and abs(y - y_cord) < 0.01):
                    self.cmd_obj.publish(move_cmd)
                    rospy.sleep(0.1)
                    [x, y, theta] = self.get_self_stateGT()

                [v, w] = self.get_self_speedGT()
                while not (abs(v) < 0.01 and abs(w) < 0.05):
                    self.cmd_vel.publish(Twist())  # stop the robot
                    rospy.sleep(0.1)
                    [v, w] = self.get_self_speedGT()
                rospy.sleep(0.1)

    def get_start_point(self):
        offset = random.uniform(-1.0, 1.0)
        if self.obstacles.world == 'l-corridor' or (
                self.obstacles.world == 'plus-corridor' and self.obstacles.goal_direction == 'adjacent'):
            x = -10
            y = 0
        elif self.obstacles.world == 'open':
            x = 0
            y = 0
        elif self.obstacles.world == 'corridor' or (
                        self.obstacles.world == 'plus-corridor' and self.obstacles.goal_direction == 'straight'):
            x = -10
            y = 0
        else:
            x = 0 + offset
            y = 0 + offset
        return x, y

    def get_goal_point(self, x):
        offset = random.uniform(-1.0, 1.0)
        if self.obstacles.world == 'l-corridor' or (
                self.obstacles.world == 'plus-corridor' and self.obstacles.goal_direction == 'adjacent'):
            x_g = 0
            y_g = 10
        elif self.obstacles.world == 'open':
            x_g = x + 10
            y_g = 0
        elif self.obstacles.world == 'corridor' or (
                        self.obstacles.world == 'plus-corridor' and self.obstacles.goal_direction == 'straight'):
            x_g = x + 20
            y_g = 0
        else:
            x_g = offset
            y_g = offset
        return x_g, y_g

    def setup_obstacles(self, ep):
        self.obstacles.setup_obstacles(self.goal_pose, self.init_pose,
                                       self.direction, self.episode)
        self.step_count = 1

    def set_obstacle_class(self, config, config_name, stageros):
        if config.scene.world == 'open':
            from env.obstacle.open import Obstacles
            self.obstacles = Obstacles(config, config_name, self.robot, stageros)
        elif config.scene.world == 'l-corridor' or (config.scene.world == 'plus-corridor' and config.scene.goal_direction == 'adjacent'):
            from env.obstacle.corridor_l import Obstacles
            self.obstacles = Obstacles(config, config_name, self.robot, stageros)
        elif config.scene.world == 'corridor' or (config.scene.world == 'plus-corridor' and config.scene.goal_direction == 'straight'):
            from env.obstacle.corridor_straight import Obstacles
            self.obstacles = Obstacles(config, config_name, self.robot, stageros)
        else:
            pass

    def check_distance(self, x, y):
        return self.obstacles.is_robot_crash(x, y)

    def check_critical(self, x, y):
        return self.obstacles.is_robot_critical(x, y)