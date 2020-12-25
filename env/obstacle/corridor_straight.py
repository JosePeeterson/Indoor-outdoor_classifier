import random

import numpy as np

from env.obstacle.base import ObstacleBase


class Obstacles(ObstacleBase):
    def __init__(self, config, config_name, robot, stageros=True):
        super(Obstacles, self).__init__(config, config_name, robot, stageros)
        self.x_axis = [-10.0, 10.0]
        if self.cfg.world == 'corridor':
            self.y_axis = [-1.0, 1.0]
        else:
            self.y_axis = [-10.0, -1.0, 1.0, 10.0]

    def get_x_cord(self, n):
        size = n
        if n < 2:
            x_cord = [random.uniform((self.x_axis[0] + 2), (self.x_axis[1] - 2))]
        else:
            x_cord = np.asarray([0.0] * size)
            start = random.uniform((self.x_axis[0] + 1), (self.x_axis[0] + 3))
            diff = 20.0 / float(size)
            for i in range(1, size + 1):
                x_cord[i - 1] = start + (i - 1) * diff
                x_cord[i - 1] += random.uniform(0.1, 0.9)
        return x_cord

    def get_fixed_y(self):
        offset_1 = self.y_axis[0] if self.cfg.world == 'corridor' else self.y_axis[1]
        offset_2 = self.y_axis[1] if self.cfg.world == 'corridor' else self.y_axis[2]
        y = [offset_1 + 0.6, offset_2 - 0.6] # +/-0.38
        return random.choice(y)

    @staticmethod
    def get_offset_dynamic():
        return [-0.78, 0.78, -0.56, 0.56]

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
                    self.cmd_pose(i, coords[i], self.get_fixed_y())
                else:
                    self.cmd_pose(i, -100, i - 100)
        else:
            raise Exception('Scene Type Unknown!')
        self.moving = False

    def arrange_static_obstacles(self, i, mov_x_list, mov_y_list, start, goal,
                                 x):
        if self.cfg.type in ['Towards', 'Parallel'] and len(mov_y_list) == 1:
            y = self.get_fixed_y()
            if (mov_y_list[0] > 0 and y > 0) or (mov_y_list[0] < 0 and y < 0):
                y = -y
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            y = self.get_fixed_y()
            while -1 < x < 1:
                x = x + 2
        else:
            raise Exception('Unknown cfg.type encountered.')
        self.cmd_pose(i, x, y)

    def arrange_dynamic_obstacles(self, i, goal_point, start_point):
        # Moving obstacles
        offsets = self.get_offset_dynamic()
        if self.cfg.type == 'Towards':
            mov_x = goal_point[0] + self.direction
        elif self.cfg.type == 'Parallel':
            mov_x = start_point[0] - 3 * self.direction
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            offset = random.uniform(offsets[2], offsets[3])
            if self.cfg.world == 'corridor':
                mov_x = (start_point[0] + goal_point[0]) / 2 + 3 * random.uniform(-1, 1)
            else:
                mov_y = self.y_axis[-1] - 2
                mov_x = self.get_fixed_y()
                return mov_x, mov_y
        else:
            raise Exception('Unknown cfg.type encountered.')
        mov_y = self.get_fixed_y()
        return mov_x, mov_y

    def check_wall_crash(self, x, y, obstacle_id):
        if self.stageros:
            if obstacle_id == 1: return self.is_crashed_1
            if obstacle_id == 2: return self.is_crashed_2
        else:
            if self.cfg.type in ['Towards', 'Parallel']:
                if x < self.x_axis[0] - 5 or x > self.x_axis[1] + 5: return True
            if self.cfg.type == 'Perpendicular':
                if y < self.y_axis[0] + 0.5 or y > self.y_axis[-1] - 0.5: return True
        return False

    def determine_range(self, start, goal):
        if self.cfg.type in ['Towards', 'Parallel']:
            if start[0] < goal[0]:
                l_bound = start[0] + 7  # 3
                r_bound = goal[0] - 7  # 2
            else:
                l_bound = goal[0] + 3
                r_bound = start[0] - 2
        elif self.cfg.type in ['Perpendicular', 'Circular']:
            min_v = min(start[1], goal[1])
            max_v = max(start[1], goal[1])
            l_bound = min_v - 1  # 2
            r_bound = max_v + 1  # 2
        else:
            raise Exception('Unknown cfg.type encountered.')
        return int(l_bound), int(r_bound)
