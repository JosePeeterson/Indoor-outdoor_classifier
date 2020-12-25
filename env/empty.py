import numpy as np
import math
import rospy
import copy
import random
from env.base import StageBase


class StageWorld(StageBase):
    def __init__(self, beam_num, index, num_env):

        super(StageWorld, self).__init__(beam_num, index, num_env)

        # # Wait until the first callback
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
            pass

        rospy.sleep(1.)

    def generate_random_pose(self):
        if self.index == 0: [x, y, theta] = [0, 0, np.pi / 2]
        if self.index == 1: [x, y, theta] = [0, 10, np.pi * 3 / 2]
        return [x, y, theta]

    def generate_random_goal(self):
        if self.index == 0: [x, y] = [0, 10]
        if self.index == 1: [x, y] = [0, 0]
        return [x, y]