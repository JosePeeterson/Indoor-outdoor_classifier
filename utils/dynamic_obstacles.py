import os
import random
import re

import numpy as np
import rospy
import tf
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry


class Obstacles:
    def __init__(self, config, config_name, stageros=True):
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
        self.radius = self.cfg.collision_radius
        self.direction = None
        self.recover_count = self.num_box * [0]
        self.reversed = self.num_box * [1]
        self.setting_up = None
        self.num_dynamic = self.cfg.num_dynamic
        self.obstacle_vel = [[0.0, 0.0] for _ in range(self.num_dynamic)]
        self.opposite = self.num_dynamic * [1]
        self.num_spawn_box = 4
        self.moving = False

        if stageros:
            self.cmd_dict = {}
            for i in range(self.num_spawn_box):
                self.cmd_dict['pos_%d' % (i + 1)] = rospy.Publisher(
                    '/robot_%d/cmd_pose' % (i + 2), Pose, queue_size=10)
                self.cmd_dict['vel_%d' % (i + 1)] = rospy.Publisher(
                    '/robot_%d/cmd_vel' % (i + 2), Twist, queue_size=10)
                self.state_listener = rospy.Subscriber(
                    '/robot_%d/odom' % (i + 2), Odometry,
                    self.odometry_callback)
        else:
            self.cmd_obj = rospy.Publisher('/gazebo/set_model_state',
                                           ModelState,
                                           queue_size=10)
            self.state_listener = rospy.Subscriber('/gazebo/model_states',
                                                   ModelStates,
                                                   self.model_state_callback)

    def odometry_callback(self, Odometry):
        topic = Odometry._connection_header['topic']     # This is attempting to access an internal used variable of class
        obj_ind = int(re.findall(r'\d+', topic)[0]) - 1
        self.box_pose['pos_%d' % obj_ind] = Odometry.pose.pose

    def model_state_callback(self, ModelStates):
        for i in range(self.num_box):
            self.box_pose['pos_%d' % (i + 1)] = ModelStates.pose[1 + i]

    def cmd_pose_gazebo(self, index, x_cor, y_cor):
        move_cmd = ModelState()
        move_cmd.model_name = 'my_obstacle_%s' % index
        move_cmd.pose.position.x = x_cor
        move_cmd.pose.position.y = y_cor
        move_cmd.pose.position.z = 0.1
        move_cmd.reference_frame = 'world'
        self.cmd_obj.publish(move_cmd)
        rospy.sleep(0.1)

    def cmd_pose_stage(self, func, pose, angle):
        pose_cmd = Pose()
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, angle, 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        # print(pose_cmd)
        func.publish(pose_cmd)

    def cmd_vel_stage(self, func, action):
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
        # print(move_cmd)
        self.cmd_obj.publish(move_cmd)

    def step(self):
        if self.cfg.type in self.cfg.scene_list:
            if not self.moving:
                self.run_obstacles_velocity()
                #self.moving = True # TODO: Fix this for gazebo
        else:
            pass

    def cmd_pose(self, i, x, y, angle=0):
        #print('Placing obstacle %s at (%1.2f,%1.2f)' % (i, x, y))
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

    def spawn_obstacles(self):
        #only for gazebo
        for index in range(self.num_box):
            print('spawning obstacle %s' % index)
            spawn_cmd = 'rosrun gazebo_ros spawn_model -file model/gazebo_models/box.urdf -urdf ' \
                        '-z 1 -model my_obstacle_%s' % index
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
            coords = [v + 3 for v in random.sample(range(5), 5)][:n]
            offset = start_point if self.direction < 0 else goal_point
            for i in range(self.num_spawn_box):
                if i < n:
                    self.cmd_pose(i, coords[i] + offset[0],
                                  random.uniform(-2.0, 2.0) + offset[1])
                else:
                    self.cmd_pose(i, -100, i - 100)
        else:
            raise Exception('Scene Type Unknown!')
        self.moving = False

    def get_scene_seed(self):
        scene_seed = 0
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
        return scene_seed

    def swap_configs(self):
        if self.cfg.dual_direction:
            if self.cfg.type == 'Towards':
                self.cfg.type = 'Parallel'
            elif self.cfg.type == 'Parallel':
                self.cfg.type = 'Towards'

    def dynamic_setup(self, goal_point, start_point):
        # Reset Dynamic counters
        self.recover_count = self.num_box * [0]
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
                self.cmd_pose(i, mov_x, self.opposite[i] * mov_y, -np.pi / 2)
            else:
                self.cmd_pose(i, mov_x, mov_y)
        self.init_obstacles_velocity()

        # Static obstacles
        if self.cfg.use_static_obstacles:
            l_bound, u_bound = self.determine_range(start_point, goal_point)
            coords = list(range(l_bound, u_bound + 1))
            fixed_coords = random.sample(coords,
                                         len(coords))[:self.num_box - 1]
            for i in range(self.num_dynamic, self.num_box):
                self.arrange_static_obstacles(i, mov_x_list, mov_y_list,
                                              start_point, goal_point,
                                              fixed_coords[i - 1])
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
        if self.cfg.type == 'Towards':
            offset = 2 * random.uniform(-1, 1)
            mov_x = goal_point[0] + self.direction
        elif self.cfg.type == 'Parallel':
            offset = 2 * random.uniform(-1, 1)
            mov_x = start_point[0] - 3 * self.direction
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            offset = random.uniform(2, 4)
            mov_x = (start_point[0] + goal_point[0]) / 2 + 3 * random.uniform(
                -1, 1)
        else:
            raise Exception('Unknown cfg.type encountered.')

        mov_y = goal_point[1] + offset
        return mov_x, mov_y

    def determine_range(self, start, goal):
        if self.cfg.type in ['Towards', 'Parallel']:
            if start[0] < goal[0]:
                l_bound = start[0] + 3
                r_bound = goal[0] - 2
            else:
                l_bound = goal[0] + 3
                r_bound = start[0] - 2
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            min_v = min(start[1], goal[1])
            max_v = max(start[1], goal[1])
            l_bound = min_v - 2
            r_bound = max_v + 2
        else:
            raise Exception('Unknown cfg.type encountered.')
        return int(l_bound), int(r_bound)

    def arrange_static_obstacles(self, i, mov_x_list, mov_y_list, start, goal,
                                 x):
        if self.cfg.type in ['Towards', 'Parallel']:
            point1, point2 = start[1], goal[1]
            ref = mov_y_list
            offset = 2
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            point1, point2 = start[0], goal[0]
            ref = mov_x_list
            offset = -2
        else:
            raise Exception('Unknown cfg.type encountered.')

        min_x = min(point1, point2)
        max_x = max(point1, point2)
        y = random.uniform(min_x - offset, max_x + offset)
        min_dist = 0
        while min_dist < 1:
            y = random.uniform(min_x - offset, max_x + offset)
            ref_dist = 10
            for r in ref:
                ref_dist = min(ref_dist, abs(r - y))
            min_dist = max(min_dist, ref_dist)

        if self.cfg.type in ['Perpendicular', 'Circular']:
            y, x = x, y

        self.cmd_pose(i, x, y)

    def init_obstacles_velocity(self):
        for i in range(self.num_dynamic):
            self.obstacle_vel[i] = [
                random.uniform(0.6, 1.0),
                random.uniform(0.1, 0.5)
            ]
            #print("Velocity for box %d: %s:" % (i, self.obstacle_vel[i]))

    def run_obstacles_velocity(self):
        for i in range(self.num_dynamic):
            v = self.get_velocity(i, self.is_obstacle_crash(i + 1),
                                  self.stageros)
            if self.stageros:
                self.cmd_vel_stage(self.cmd_dict['vel_%d' % (i + 1)], v)
            else:
                self.cmd_vel_gazebo(i, v)
            self.swap_configs()

    def get_velocity(self, i, is_obstacle_crash, stage):
        [lx, ly] = self.obstacle_vel[i]
        if self.cfg.type == 'Towards': v = [self.direction * lx, 0]
        elif self.cfg.type == 'Parallel': v = [self.direction * -lx, 0]
        elif self.cfg.type == 'Perpendicular':
            v = [lx * self.opposite[i], 0
                 ] if stage else [0, -lx * self.opposite[i]]
        elif self.cfg.type == 'Circular':
            v = [lx, -ly]
        else:
            raise Exception('Scene Type unknown!')

        if is_obstacle_crash:
            if self.cfg.reverse_collision:
                if self.recover_count[i] > 20:
                    self.reversed[i] *= -1
                    self.recover_count[i] = 0
            else:
                v = [0, 0]

        v = [self.reversed[i] * l for l in v]
        self.recover_count[i] += 1
        return v

    def is_robot_crash(self, x, y):
        crash_flag = False
        dist_box = []
        for a_key in self.box_pose.keys():
            a_position = self.box_pose[a_key]
            x_1 = float(a_position.position.x)
            y_1 = float(a_position.position.y)
            dist = np.sqrt((x - x_1)**2 + (y - y_1)**2)
            if dist < self.radius:
                print "Collision from behind detected: %.2f" % dist
                crash_flag = True
            dist_box.append(dist)
        return crash_flag, min(dist_box)

    def is_obstacle_crash(self, obstacle_id):
        # moving box
        a_key = 'pos_' + str(obstacle_id)
        a_position = self.box_pose[a_key]
        x_1 = float(a_position.position.x)
        y_1 = float(a_position.position.y)

        #check dist with static boxes
        crash_flag = False
        for b_key in self.box_pose.keys():
            distances = []
            if a_key != b_key:
                b_position = self.box_pose[b_key]
                x_2 = float(b_position.position.x)
                y_2 = float(b_position.position.y)
                dist = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                distances.append(dist)
                min_dist = np.min(distances)
                if min_dist < 0.61:
                    #print "Collision between obstacle %s and %s , distance: %.3f" % (
                    #    a_key, b_key, dist)
                    crash_flag = True
        return crash_flag
