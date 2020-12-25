import numpy as np
import rospy
import random

from env.base import StageBase
from utils.misc import get_init_pose, get_goal_point


class StageWorld(StageBase):
    def __init__(self, beam_num, index, num_env):

        super(StageWorld, self).__init__(beam_num, index, num_env)
        self.floor_scale = 1.5
        self.angle = np.pi / 4

        self.groups = [[0, 1], [2, 3], [6, 7, 8, 9], [10, 11, 12, 13, 14],
                       [15, 16], [17, 18], [19, 20, 21, 22, 23],
                       [34, 35, 36, 37, 38], [39, 40, 41, 42, 43]]
        self.group_dict = {}
        for i, group in enumerate(self.groups):
            for v in group:
                self.group_dict[v] = i

        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
            pass

        rospy.sleep(1.)

    def generate_random_pose(self):
        [x, y, theta] = get_init_pose(self.index)
        theta += random.uniform(-self.angle, self.angle)
        x, y = self.offset_coordinates(x, y)
        return [x, y, theta]

    def offset_coordinates(self, x, y):
        x += random.uniform(-0.5, 0.5)
        y += random.uniform(-0.5, 0.5)
        x *= self.floor_scale
        y *= self.floor_scale
        return x, y

    def generate_random_goal(self):
        fixed = [4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        if self.index in fixed:
            [x, y] = get_goal_point(self.index)
        else:
            # Find group id and val position
            group_idx = self.group_dict[self.index]
            group = self.groups[group_idx]
            val_idx = group.index(self.index)
            # Shuffle each group according to episode number
            random.seed(self.episode)
            random.shuffle(group)

            [x, y] = get_goal_point(group[val_idx])
        x, y = self.offset_coordinates(x, y)
        return [x, y]
