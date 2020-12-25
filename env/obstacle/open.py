import random
from env.obstacle.base import ObstacleBase


class Obstacles(ObstacleBase):
    def __init__(self, config, config_name, robot, stageros=True):
        super(Obstacles, self).__init__(config, config_name, robot, stageros)

    @staticmethod
    def get_offset_dynamic():
        return [-1, 1, 2, 4]

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
