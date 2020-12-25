import numpy as np
import math
import rospy
import random

from env.base import StageBase


class StageWorld(StageBase):
    def __init__(self, beam_num, index, num_env):

        super(StageWorld, self).__init__(beam_num, index, num_env)
        self.circle_radius = 10

        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1)

    def generate_random_reset_pose(self):
        return self.generate_random_pose(self.circle_radius - 2)

    def generate_random_pose(self, radius_lower=0):
        # Initialisation
        x = y = theta = 0
        dist_max = 0
        obs_coords = [[-2, -2], [-1, 1], [2, -2], [2, 2], [3, 4], [-3, -5],
                      [1, -6], [-2, 4]]
        while dist_max < 1.0:
            [x, y, theta] = self.generate_random_single(radius_lower)
            dist2obs = 100
            for coord in obs_coords:
                dist2obs = min(dist2obs,
                               np.sqrt((x - coord[0])**2 + (y - coord[1])**2))
            dist_max = max(dist_max, dist2obs)
        return [x, y, theta]

    def generate_random_single(self, radius_lower):
        r = random.uniform(radius_lower, self.circle_radius - 1)
        angle = random.uniform(0, 2 * np.pi)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        theta = random.uniform(0, 2 * np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        # Initialisation
        x = y = 0
        dis_goal = 0
        while (dis_goal > self.circle_radius or dis_goal <
               (self.circle_radius - 1)) and not rospy.is_shutdown():
            [x, y, _] = self.generate_random_pose()
            dis_goal = np.sqrt((x - self.init_pose[0])**2 +
                               (y - self.init_pose[1])**2)
        return [x, y]
