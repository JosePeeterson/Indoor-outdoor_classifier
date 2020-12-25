import numpy as np
import math
import rospy
import random
from env.base import StageBase

class StageWorld(StageBase):
    def __init__(self, beam_num, index, num_env):
        # initialise
        self.default_pose = None

        super(StageWorld, self).__init__(beam_num, index, num_env)

        # # Wait until the first callback
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
            pass

        rospy.sleep(1.)

    def generate_random_pose(self):
        r = 5  # random.uniform(5, self.circle_radius - 1)
        angle = random.uniform(0, 2 * np.pi)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        [x, y] = self.apply_offset(x, y)
        theta = random.uniform(0, 2 * np.pi)
        self.default_pose = angle
        return [x, y, theta]

    def apply_offset(self, x, y):
        x_offset = [-90, -30, 30, 90, -90, -30, 30, 90, -90, -30, 30, 90]
        y_offset = [-60, -60, -60, -60, 0, 0, 0, 0, 60, 60, 60, 60]
        return x + x_offset[self.index], y + y_offset[self.index]

    def generate_random_goal(self):        
        r = 5
        angle = self.default_pose + np.pi * (1 - random.uniform(0, 0.2))
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        [x, y] = self.apply_offset(x, y)
        return [x, y]
