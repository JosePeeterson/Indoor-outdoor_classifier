import numpy as np
import copy


class Collision:
    def __init__(self, env, config, comm):
        self.comm = comm
        self.env = env
        self.config = config
        self.stuck_pos_list = []
        self.obs = None

    def update_observation(self, obs):
        self.obs = obs

    def mutual_collision_check(self, liveflag):
        [x, y, _] = self.env.get_self_stateGT()
        crash_list = self.comm.gather((self.env.get_crash_state() and liveflag),
                                      root=0)
        pose_list = copy.deepcopy(self.comm.gather([x, y], root=0))
        crash_mutual_list = [False] * self.config.NUM_ENV
        if self.env.index == 0:
            for i, is_crashed in enumerate(crash_list):
                if is_crashed:
                    for j, y_pose in enumerate(pose_list):
                        x_pose = pose_list[i]
                        dist = np.sqrt((x_pose[0] - y_pose[0]) ** 2 +
                                       (x_pose[1] - y_pose[1]) ** 2)
                        if dist < 1.1:  # collision check
                            crash_mutual_list[j] = True
        crash_mutual = self.comm.scatter(crash_mutual_list, root=0)
        return crash_mutual

    def control_evade(self):
        """
        Evasion behaviour on simple run-based control
        """

        [speed, angle] = self.env.get_self_speedGT()
        obs = self.obs
        evade = [speed, angle]
        obstacle_pos = np.argmin(obs)

        evade[0] = self.config.EVASION_SPEED  # speed
        # check the relative position between agent and obstacle
        if obstacle_pos < len(obs):  # obstacle at left
            evade[1] = 1  # turn right
        else:
            evade[1] = -1  # turn left

        return evade

    def critical_check(self):
        """
        Check if the moving agent has been close to an obstacle.

        Return True if too close, otherwise False.
        """
        [x, y, _] = self.env.get_self_stateGT()
        obs = self.obs
        dist = min(obs)

        dist_to_goal = np.sqrt((self.env.goal_pose[0] - x) ** 2 +
                               (self.env.goal_pose[1] - y) ** 2)

        if dist < self.config.CRITICAL_THRES and \
                dist_to_goal > self.config.GOAL_DIST:
            return True
        else:
            return False

    def stuck_check(self):
        """
        Check if the moving agent has been stuck for certain steps.

        Return True if stuck, otherwise False.
        """

        [x, y, _] = self.env.get_self_stateGT()
        if not self.stuck_pos_list:
            self.stuck_pos_list.append([x, y])
            return False
        else:
            [last_x, last_y] = self.stuck_pos_list[-1]

            if abs(x - last_x) < self.config.STUCK_DIST and \
                    abs(y - last_y) < self.config.STUCK_DIST:
                self.stuck_pos_list.append([x, y])
            else:
                self.stuck_pos_list = []
                self.stuck_pos_list.append([x, y])

        if len(self.stuck_pos_list) > self.config.STUCK_THRES:
            self.stuck_pos_list = []
            return True
        else:
            return False
