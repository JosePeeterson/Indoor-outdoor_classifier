import rospy
import copy
import tf
import numpy as np

from geometry_msgs.msg import Twist, Pose, Point, PoseStamped, Quaternion, Pose, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from std_msgs.msg import Int8


class StageBase(object):
    def __init__(self, beam_num, index, num_env):
        self.beam_num = beam_num
        self.index = index
        self.num_env = num_env

        self.goal_size = 0.5
        self.distance = None
        self.pre_distance = None
        rospy.init_node('StageEnv_' + str(index), anonymous=None)

        # Stats calculation
        self.init_pose = None
        self.prev_pose = None
        self.goal_pose = None
        self.total_dist = 0
        self.speed_list = []
        self.steps = 0
        self.episode = 0

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry,
                                                 self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan,
                                          self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8,
                                            self.crash_callback)

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.scan = None
        self.speed_GT = None
        self.state_GT = None
        self.is_crashed = None
        self.ready = False

        # Extra Flags
        self.obstacle_flag = False

        # Region Reward
        self.USE_REGION_REWARD = False
        self.reward_per_region = None
        self.distance_per_region = None
        self.num_regions = None
        self.dist_reach_masks = None
        self.pre_region_id = None

    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [
            GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y,
            Euler[2]
        ]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x ** 2 + v_y ** 2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan = np.array(scan.ranges)
        self.ready = True
        self.steps += 1

    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion(
            [Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [
            odometry.pose.pose.position.x, odometry.pose.pose.position.y,
            Euler[2]
        ]
        self.speed = [
            odometry.twist.twist.linear.x, odometry.twist.twist.angular.z
        ]

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        while not self.ready:
            continue
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_num
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate(
            (sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        self.ready = False
        return scan_sparse / 6.0 - 0.5

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        if self.steps < 5:
            return False
        else:
            return self.is_crashed

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_pose
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def obstacle_crash_check(self, is_crash, x, y):
        return is_crash

    def setup_region_reward(self,
                            max_distance_reward=15.0,
                            max_num_regions=16,
                            env_radius=10):
        self.USE_REGION_REWARD = True
        self.reward_per_region = max_distance_reward / (max_num_regions - 2)
        self.distance_per_region = (env_radius * 2 - self.goal_size) / max_num_regions
        self.num_regions = int((env_radius * 2 - self.goal_size) / self.distance_per_region) + 1
        self.dist_reach_masks = [0] * self.num_regions
        self.pre_region_id = None

    def compute_region_reward(self, x, y, step):
        dist = np.sqrt((self.goal_pose[0] - x) ** 2 + (self.goal_pose[1] - y) ** 2)
        cur_region_id = int((dist - self.goal_size) / self.distance_per_region)
        if step <= 2:
            self.dist_reach_masks[cur_region_id] = 1
            self.pre_region_id = cur_region_id
            return 0

        if cur_region_id > self.pre_region_id:
            r = -1 * self.reward_per_region
        elif cur_region_id < self.pre_region_id:
            # consider if previous regions has
            num_unvisited_regions = (self.pre_region_id - cur_region_id) - sum(
                self.dist_reach_masks[cur_region_id:self.pre_region_id])
            r = self.reward_per_region * num_unvisited_regions

            for idx in range(cur_region_id, self.pre_region_id):
                self.dist_reach_masks[idx] = 1
        else:
            r = 0
        self.pre_region_id = cur_region_id

        return r

    def get_reward_and_terminate(self, t, is_crash=None, timeout=200, reward_dist_scale=2.5):

        # if is_crash is None: is_crash = self.get_crash_state()

        # get agent info
        terminate = False
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_pose[0] - x) ** 2 +
                                (self.goal_pose[1] - y) ** 2)
        is_crash = self.obstacle_crash_check(is_crash, x, y)

        # for stats recorder
        episode_distance = np.sqrt((x - self.prev_pose[0]) ** 2 +
                                   (y - self.prev_pose[1]) ** 2)
        self.total_dist += episode_distance
        self.prev_pose = [x, y]

        # calculate rewards
        if self.USE_REGION_REWARD:
            reward_g = self.compute_region_reward(x, y, t)
        else:
            reward_g = (self.pre_distance - self.distance) * reward_dist_scale

        reward_c = 0
        reward_w = 0
        result = 0
        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        if np.abs(w) > 1.05:
            reward_w = -0.1 * np.abs(w)

        if t > timeout:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w
        # print(t, reward_g, reward_c , reward_w)

        return reward, terminate, result

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)

    def control_pose(self, func, pose):
        pose_cmd = Pose()
        assert len(pose) == 3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        func.publish(pose_cmd)

    def reset_pose(self):
        [x, y, theta] = self.generate_random_reset_pose()
        reset_pose = [x, y, theta]

        self.control_pose(self.cmd_pose, reset_pose)
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        rospy.sleep(0.1)

        while np.abs(reset_pose[0] - x_robot) > 0.2 or np.abs(reset_pose[1] -
                                                              y_robot) > 0.2:
            [x_robot, y_robot, theta] = self.get_self_stateGT()
            self.control_pose(self.cmd_pose, reset_pose)
            rospy.sleep(0.1)
        self.steps = 0

    def generate_random_reset_pose(self):
        return self.generate_random_pose()

    def generate_random_pose(self):
        raise NotImplementedError

    def generate_goal_point(self, ep_id=0):
        self.init_pose = self.get_self_stateGT()
        [x_g, y_g] = self.generate_random_goal()
        self.goal_pose = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        self.episode = ep_id
        self.prev_pose = self.init_pose

    def generate_random_goal(self):
        raise NotImplementedError
