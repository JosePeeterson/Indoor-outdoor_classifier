import os
import random
import re

import numpy as np
import rospy
import tf
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8


class ObstacleBase(object):
    def __init__(self, config, config_name, robot, stageros=True):
        print('Initialising Obstacles')

        self.cfg = config.scene
        if self.cfg.dual_direction:
            self.cfg.num_dynamic = 2
        if self.cfg.type != 'Static':
            self.cfg.num_box = 4

        self.config_name = config_name[-2:]
        self.num_box = self.cfg.num_box
        self.stageros = stageros
        self.box_pose = {}
        self.box_speed = {}
        self.radius = 0.770 if robot == 'droc' else 0.71
        self.direction = None
        #self.recover_count = self.num_box * [0]
        self.reversed = self.num_box * [1]
        self.setting_up = None
        self.num_dynamic = self.cfg.num_dynamic
        self.obstacle_vel = [[0.0, 0.0] for _ in range(self.num_dynamic)]
        self.opposite = self.num_dynamic * [1]
        self.num_spawn_box = 4
        self.moving = False
        self.world = self.cfg.world
        self.prev_velocity = {}
        self.goal_direction = self.cfg.goal_direction
        self.is_crashed_1 = False
        self.is_crashed_2 = False
        self.recovery = [False]*2  

        if stageros:
            self.cmd_dict = {}
            for i in range(self.num_spawn_box):
                self.cmd_dict['pos_%d' % (i + 1)] = rospy.Publisher(
                    '/robot_%d/cmd_pose' % (i + 2), Pose, queue_size=10)
                self.cmd_dict['vel_%d' % (i + 1)] = rospy.Publisher(
                    '/robot_%d/cmd_vel' % (i + 2), Twist, queue_size=10)
                self.state_listener = rospy.Subscriber(
                    '/robot_%d/base_pose_ground_truth' % (i + 2), Odometry,
                    self.odometry_callback)
            crash_topic_1 = 'robot_' + str(2) + '/is_crashed'
            self.check_crash_1 = rospy.Subscriber(crash_topic_1, Int8, self.crash_callback_1)

            crash_topic_2 = 'robot_' + str(3) + '/is_crashed'
            self.check_crash_2 = rospy.Subscriber(crash_topic_2, Int8, self.crash_callback_2)
        else:
            self.cmd_obj = rospy.Publisher('/gazebo/set_model_state',
                                           ModelState,
                                           queue_size=10)
            self.state_listener = rospy.Subscriber('/gazebo/model_states',
                                                   ModelStates,
                                                   self.model_state_callback)

    def odometry_callback(self, Odometry):
        topic = Odometry._connection_header['topic']  # This is attempting to access an internal used variable of class
        obj_ind = int(re.findall(r'\d+', topic)[0]) - 1
        self.box_pose['pos_%d' % obj_ind] = Odometry.pose.pose

    def model_state_callback(self, ModelStates):
        for i in range(self.num_box):
            self.box_pose['pos_%d' % (i + 1)] = ModelStates.pose[1 + i]

    def crash_callback_1(self, flag):
        self.is_crashed_1 = flag.data

    def crash_callback_2(self, flag):
        self.is_crashed_2 = flag.data

    def cmd_pose_gazebo(self, index, x_cor, y_cor):
        move_cmd = ModelState()
        move_cmd.model_name = 'my_obstacle_%s' % index
        move_cmd.pose.position.x = x_cor
        move_cmd.pose.position.y = y_cor
        move_cmd.pose.position.z = 0.1
        move_cmd.reference_frame = 'world'
        self.cmd_obj.publish(move_cmd)
        rospy.sleep(0.1)

    @staticmethod
    def cmd_pose_stage(func, pose, angle):
        pose_cmd = Pose()
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, angle, 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        func.publish(pose_cmd)

    @staticmethod
    def cmd_vel_stage(func, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.angular.z = action[1]
        func.publish(move_cmd)

    def cmd_vel_gazebo(self, i, v):
        move_cmd = ModelState()
        move_cmd.pose = self.box_pose['pos_%d' % (i + 1)]
        move_cmd.model_name = 'my_obstacle_%s' % i
        move_cmd.twist.linear.x = v[0]
        move_cmd.twist.linear.y = v[1]
        move_cmd.reference_frame = 'world'
        self.cmd_obj.publish(move_cmd)

    def step(self):
        if self.cfg.type in self.cfg.scene_list:
            if not self.moving:
                self.run_obstacles_velocity()
        else:
            pass

    def cmd_pose(self, i, x, y, angle=0):
        if self.stageros:
            self.cmd_pose_stage(self.cmd_dict['pos_%d' % (i + 1)], [x, y],
                                angle)
        else:
            index_check = False
            for j in range(self.num_dynamic):
                if i == j:
                    index_check = index_check or True
            if self.setting_up and index_check:
                self.box_pose['pos_%d' % (i + 1)].position.x = x
                self.box_pose['pos_%d' % (i + 1)].position.y = y
                x_new = y_new = 1000
                while not (abs(x_new - x) < 0.01 and abs(y_new - y) < 0.01):
                    self.box_pose['pos_%d' % (i + 1)].position.x = x
                    self.box_pose['pos_%d' % (i + 1)].position.y = y
                    self.cmd_pose_gazebo(i, x, y)
                    rospy.sleep(0.1)
                    x_new = self.box_pose['pos_%d' % (i + 1)].position.x
                    y_new = self.box_pose['pos_%d' % (i + 1)].position.y
                if i == self.num_dynamic: self.setting_up = False
            else:
                self.cmd_pose_gazebo(i, x, y)

    def remove_obstacles(self):
        for i in range(self.num_box):
            self.cmd_pose_gazebo(i, -2*i, -3)

    def spawn_obstacles(self):
        # only for gazebo
        for index in range(self.num_box):
            print('spawning obstacle %s' % index)
            spawn_cmd = 'rosrun gazebo_ros spawn_model -file model/gazebo_models/box.urdf -urdf -z 1 -model my_obstacle_%s' % index
            os.system(spawn_cmd)
            self.cmd_pose(index, index, index)
            rospy.sleep(0.1)

    def setup_obstacles(self, goal_point, start_point, direction, ep):
        random.seed(ep + int(self.config_name))
        n = self.num_box
        self.direction = direction
        if self.cfg.type in self.cfg.scene_list:
            self.dynamic_setup(goal_point, start_point)
        elif self.cfg.type == 'Static':
            coords = self.get_x_cord(n)
            offset = start_point if self.direction < 0 else goal_point
            for i in range(self.num_spawn_box):
                if i < n:
                    self.cmd_pose(i, coords[i] + offset[0],
                                  self.get_fixed_y() + offset[1])
                else:
                    self.cmd_pose(i, -100, i - 100)
        else:
            raise Exception('Scene Type Unknown!')
        self.moving = False

    def get_scene_seed(self):
        if self.cfg.type == 'Towards':
            scene_seed = 1
        elif self.cfg.type == 'Parallel':
            scene_seed = 2
        elif self.cfg.type == 'Perpendicular':
            scene_seed = 3
        elif self.cfg.type == 'Circular':
            scene_seed = 4
        elif self.cfg.type == 'Static':
            scene_seed = 5
        else:
            scene_seed = 0
        return scene_seed

    def swap_configs(self):
        if self.cfg.dual_direction:
            if self.cfg.type == 'Towards':
                self.cfg.type = 'Parallel'
            elif self.cfg.type == 'Parallel':
                self.cfg.type = 'Towards'
            else:
                pass

    def dynamic_setup(self, goal_point, start_point):
        # Reset Dynamic counters
        #self.recover_count = self.num_box * [0]
        self.reversed = self.num_box * [1]
        mov_x_list = []
        mov_y_list = []
        mov_x = mov_y = None
        for i in range(self.num_dynamic):
            while self.check_dynamic_collision(mov_x, mov_y, mov_x_list,
                                               mov_y_list):
                mov_x, mov_y = self.arrange_dynamic_obstacles(
                    i, goal_point, start_point)
            mov_x_list.append(mov_x)
            mov_y_list.append(mov_y)
            self.swap_configs()

        self.setting_up = True
        if self.cfg.dual_direction:
            self.opposite = [1, -1]
        else:
            self.opposite = [1 if random.random() < 0.5 else -1
                             ] * self.cfg.num_dynamic
        for i, (mov_x, mov_y) in enumerate(zip(mov_x_list, mov_y_list)):
            if self.cfg.type == 'Perpendicular':
                if self.cfg.world == 'l-corridor' and self.opposite[i] < 0: mov_y = random.uniform(0, 0.5)
                self.cmd_pose(i, mov_x, self.opposite[i] * mov_y, -np.pi / 2)
            else:
                self.cmd_pose(i, mov_x, mov_y)
        self.init_obstacles_velocity()

        # Static obstacles
        if self.cfg.use_static_obstacles:
            fixed_coords = self.get_fixed_cord(start_point, goal_point)
            for i in range(self.num_dynamic, self.num_box):
                self.arrange_static_obstacles(i, mov_x_list, mov_y_list,
                                              start_point, goal_point,
                                              fixed_coords[i - self.num_dynamic])
        else:
            for i in range(self.num_dynamic, self.num_box):
                self.cmd_pose(i, -100, i - 100)

    def check_dynamic_collision(self, mov_x, mov_y, mov_x_list, mov_y_list):
        if mov_x is None: return True
        collision = False
        if self.cfg.type in ['Perpendicular', 'Circular']:
            for ref_x in mov_x_list:
                if abs(ref_x - mov_x) < 1:
                    collision = True
                    break
        else:
            for ref_y in mov_y_list:
                if abs(ref_y - mov_y) < 1:
                    collision = True
        return collision

    def arrange_dynamic_obstacles(self, i, goal_point, start_point):
        # Moving obstacles
        offsets = self.get_offset_dynamic()
        if self.cfg.type == 'Towards':
            offset = 2 * random.uniform(offsets[0], offsets[1])
            mov_x = goal_point[0] + self.direction
        elif self.cfg.type == 'Parallel':
            offset = 2 * random.uniform(offsets[0], offsets[1])
            mov_x = start_point[0] - 3 * self.direction
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            offset = random.uniform(offsets[2], offsets[3])
            mov_x = (start_point[0] + goal_point[0]) / 2 + 3 * random.uniform(
                -1, 1)
        else:
            raise Exception('Unknown cfg.type encountered.')

        mov_y = goal_point[1] + offset
        return mov_x, mov_y

    def determine_range(self, start, goal):
        raise NotImplementedError

    def arrange_static_obstacles(self, i, mov_x_list, mov_y_list, start, goal, x):
        raise NotImplementedError

    def init_obstacles_velocity(self):
        for i in range(self.num_dynamic):
            self.obstacle_vel[i] = [random.uniform(0.6, 1.0), random.uniform(0.1, 0.5)]
        if self.num_dynamic == 2:
            self.obstacle_vel[i][0] = self.obstacle_vel[0][0] - 0.4

    def run_obstacles_velocity(self):
        for i in range(self.num_dynamic):
            pub_vel = False
            v = self.get_velocity(i, self.is_obstacle_crash(i + 1),
                                  self.stageros)
            if i + 1 not in self.prev_velocity.keys():
                pub_vel = True
            else:
                if self.prev_velocity[i + 1] != v:
                    pub_vel = True
            if pub_vel:
                if self.stageros:
                    self.cmd_vel_stage(self.cmd_dict['vel_%d' % (i + 1)], v)
                else:
                    self.cmd_vel_gazebo(i, v)
            self.prev_velocity[i + 1] = v
            self.swap_configs()

    def get_velocity(self, i, is_obstacle_crash, stage):
        [lx, ly] = self.obstacle_vel[i]
        if self.cfg.type == 'Towards':
            v = [self.direction * lx, 0]
        elif self.cfg.type == 'Parallel':
            v = [self.direction * -lx, 0]
        elif self.cfg.type == 'Perpendicular':
            v = [lx * self.opposite[i], 0
                 ] if stage else [0, -lx * 1]
        elif self.cfg.type == 'Circular':
            v = [lx, -ly]
        else:
            raise Exception('Scene Type unknown!')
        
        if is_obstacle_crash == False:
           self.recovery[i] = True     
     
        if is_obstacle_crash:
            if self.cfg.reverse_collision:
                if self.recovery[i] == True:#self.recover_count[i] > 20: #
                    self.reversed[i] *= -1
                    #self.recover_count[i] = 0
                    self.recovery[i] = False
            else:
                v = [0, 0]

        v = [self.reversed[i] * l for l in v]
        #self.recover_count[i] += 1
        return v

    def is_robot_crash(self, x, y):
        crash_flag = False
        dist_box = []
        for a_key in self.box_pose.keys():
            a_position = self.box_pose[a_key]
            x_1 = float(a_position.position.x)
            y_1 = float(a_position.position.y)
            dist = np.sqrt((x - x_1) ** 2 + (y - y_1) ** 2)
            if dist < self.radius:
                print "Collision from behind detected: %.3f" % dist
                crash_flag = True
            dist_box.append(dist)
        return crash_flag, min(dist_box)

    def is_robot_critical(self, x, y):
        dist_box = []
        obstacle_ref = {}
        for a_key in self.box_pose.keys():
            a_position = self.box_pose[a_key]
            x_1 = float(a_position.position.x)
            y_1 = float(a_position.position.y)
            dist = np.sqrt((x - x_1) ** 2 + (y - y_1) ** 2)
            dist_box.append(dist)
            obstacle_ref[dist] = [x_1, y_1]
        min_dist = min(dist_box)
        return min_dist, obstacle_ref[min_dist]

    def is_obstacle_crash(self, obstacle_id):
        # moving box
        a_key = 'pos_' + str(obstacle_id)
        a_position = self.box_pose[a_key]
        x_1 = float(a_position.position.x)
        y_1 = float(a_position.position.y)

        # check dist with static boxes
        crash_flag = False
        for b_key in self.box_pose.keys():
            distances = []
            if a_key != b_key:
                b_position = self.box_pose[b_key]
                x_2 = float(b_position.position.x)
                y_2 = float(b_position.position.y)
                dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
                distances.append(dist)
                min_dist = np.min(distances)
                if min_dist < 0.61:
                    # print "Collision between obstacle %s and %s , distance: %.3f" % (
                    #    a_key, b_key, dist)
                    return True
        if self.world == 'l-corridor' or self.world == 'corridor' or self.world == 'plus-corridor':
            crash_flag = self.check_wall_crash(x_1, y_1, obstacle_id)
        return crash_flag

    @staticmethod
    def get_x_cord(n):
        return [v + 3 for v in random.sample(range(5), 5)][:n]

    def get_fixed_cord(self, start_point, goal_point):
        if self.cfg.world in ['l-corridor', 'plus-corridor', 'corridor']:
            fixed_coords = self.get_x_cord(self.num_box - self.num_dynamic)
        else:
            l_bound, u_bound = self.determine_range(start_point, goal_point)
            coords = list(range(l_bound, u_bound + 1))
            fixed_coords = random.sample(coords, len(coords))[:self.num_box - 1]
        return fixed_coords

    @staticmethod
    def get_fixed_y():
        return random.uniform(-2.0, 2.0)

    @staticmethod
    def get_offset_dynamic():
        raise NotImplementedError

    def check_wall_crash(self, x, y, obstacle_id):
        raise NotImplementedError
