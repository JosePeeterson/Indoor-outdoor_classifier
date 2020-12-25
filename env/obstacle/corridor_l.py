import random

import numpy as np

from env.obstacle.base import ObstacleBase


class Obstacles(ObstacleBase):
    def __init__(self, config, config_name, robot, stageros=True):
        super(Obstacles, self).__init__(config, config_name, robot, stageros)
        if self.cfg.world == 'l-corridor':
            self.x_axis = [-10, 1]
            self.y_axis = [-1, 10]
        else:
            self.x_axis = [-10, -1, 1, 10]
            self.y_axis = [-10, -1, 1, 10]

    @staticmethod
    def get_x_cord(n):
        size = n
        if n < 2:
            x_cord = [random.uniform(-8, 0)]
        else:
            x_cord = np.asarray([0.0] * size)
            start = random.uniform(-9.0, -7.0)
            diff = 10 / float(size)
            for i in range(1, size + 1):
                x_cord[i - 1] = start + (i - 1) * diff
                x_cord[i - 1] += random.uniform(0.1, 0.5)
        return x_cord

    def get_fixed_y(self, x):
        offset = self.y_axis[0] if self.cfg.world == 'l-corridor' else self.y_axis[1]
        if -1 <= x < 0.30:
            return offset + 0.35
        y = [offset + 0.35, offset + 1.62]
        return random.choice(y)

    def get_fixed_x(self, y):
        offset = self.x_axis[1] if self.cfg.world == 'l-corridor' else self.x_axis[2]
        if -0.90 <= y < 0.30:
            return offset - 0.35
        x = [offset - 0.4, offset - 1.65]
        return random.choice(x)

    def crash_callback_1(self, flag):
        self.is_crashed_1 = flag.data

    def crash_callback_2(self, flag):
        self.is_crashed_2 = flag.data

    def setup_obstacles(self, goal_point, start_point, direction, ep):
        random.seed(ep + int(self.config_name))
        n = self.num_box
        self.direction = direction
        if self.cfg.type in self.cfg.scene_list:
            self.dynamic_setup(goal_point, start_point)
        elif self.cfg.type == 'Static':
            coords = self.get_x_cord(n)
            for i in range(self.num_spawn_box):
                if i < n:
                    if i >= n / 2:
                        self.cmd_pose(i, coords[i], self.get_fixed_y(coords[i]))
                    else:
                        self.cmd_pose(i, self.get_fixed_x(-coords[i]), -coords[i])
                else:
                    self.cmd_pose(i, -100, i - 100)
        else:
            raise Exception('Scene Type Unknown!')
        self.moving = False

    def arrange_dynamic_obstacles(self, i, goal_point, start_point):
        if self.cfg.type == 'Towards':
            mov_x = self.x_axis[-1] - 2 + self.direction
        elif self.cfg.type == 'Parallel':
            mov_x = start_point[0] - 1.5 * self.direction
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            mov_y = goal_point[1] - 2
            mov_x = self.get_fixed_x(mov_y)
            return mov_x, mov_y
        else:
            raise Exception('Unknown cfg.type encountered.')
        mov_y = self.get_fixed_y(mov_x)
        return mov_x, mov_y

    def check_wall_crash(self, x, y, obstacle_id):
        if self.stageros:
            if obstacle_id == 1: return self.is_crashed_1
            if obstacle_id == 2: return self.is_crashed_2
        else:
            if self.cfg.type in ['Parallel', 'Towards']:
                if x > self.x_axis[-1] - 1 or x < self.x_axis[0] - 3: return True
            if self.cfg.type == 'Perpendicular':
                if y < self.y_axis[0] + 1 or y > self.y_axis[-1] - 1: return True
        return False

    def arrange_static_obstacles(self, i, mov_x_list, mov_y_list, start, goal,
                                 x):
        if self.cfg.type in ['Towards', 'Parallel']:
            y = - x
            if 0 < y < 1:
                y += 1
            x = self.get_fixed_x(y)
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            y = self.get_fixed_y(x)
        else:
            raise Exception('Unknown cfg.type encountered.')
        self.cmd_pose(i, x, y)
