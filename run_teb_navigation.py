import argparse
import os
import shutil
import signal
import socket
import sys
import time
from datetime import datetime

import numpy as np
import rospy
from omegaconf import OmegaConf

import env.scene_all as env_class
from utils.logger import init_logger
from utils.robot_move_base import WaypointNavigator
from utils.statistics import Statistics

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

launch_time = None
time_str = None


def signal_handler(signal, frame):
    clear_temp(launch_time, time_str)
    sys.exit()


def clear_temp(launch_time, time_str):
    curr_time = datetime.now()
    hostname = socket.gethostname()
    exe_time = (curr_time - launch_time).total_seconds()
    output_folder = './output/' + hostname + '/' + time_str
    if exe_time < 60 * 5:
        print '\n\nExecution time(mins): %f \nDeleted output folder: %s \n' % (exe_time / 60, output_folder)
        shutil.rmtree(output_folder)


class Planner:
    def __init__(self, time_str, main_config, world_type, logger, write_outputs):
        self.logger = logger
        self.env = None
        self.env_class = env_class
        self.world_type = world_type
        self.config = None  # Scene config
        self.scene_config_flag = False  # Flag for using config.yaml
        self.scene_config_list = None
        self.robot_name = None
        self.time_str = time_str
        self.parse_config(main_config)
        self.stats_recorder = Statistics(
            self.config, self.logger,
            write_outputs, self.time_str, main_config.MAX_EPISODES)  # Object to record statistics
        self.scene_description = None

    def parse_config(self, main_config):
        config = main_config.single
        self.robot_name = config.robot
        self.scene_config_flag = True
        file_list = sorted(os.listdir(config.config_folder))
        self.scene_config_list = [s for s in file_list if self.world_type in s]
        config = OmegaConf.merge(main_config.base, config)
        self.config = config
        self.logger.info(config.pretty())
        self.logger.info('##### Evaluation Mode : TEB Planner #####')

    def load_environment(self, scene_config_file, config_name=None, agent_flag=False):
        config = self.config
        if not agent_flag:
            scene_path = os.path.join(config.config_folder, scene_config_file)
            scene_config = OmegaConf.load(scene_path)
            config_name = os.path.splitext(scene_config_file)[0]
            self.logger.info('Scene Config: %s' % scene_path)
            self.logger.info(scene_config.pretty())
        else:
            scene_config = scene_config_file
        self.env = self.env_class.StageWorld(config.OBS_SIZE, scene_config, config_name, self.robot_name, True,
                                             stageros=config.stageros)
        self.scene_description = config_name

        self.env.obstacle_flag = True

        self.stats_recorder.env = self.env

    def reset_env_episode(self, result, ep_id):
        if result != 'Reach Goal' or self.config.reset_flag:
            self.env.reset_pose()
        self.env.generate_goal_point(ep_id)
        if self.env.obstacle_flag:
            self.env.setup_obstacles(ep_id)

    def run_single(self, env, ep_id):
        """
        run a single episode using teb.
        """
        # localise objects
        step = 1
        navigator = WaypointNavigator()
        navigator.set_goal(env.goal_pose, ep_id)
        step += 1

        speed_next = np.asarray(env.get_self_speed())
        env.speed_list.append(speed_next)

        # sleep for 2 seconds to keep consistency with RL
        rospy.sleep(2)

    def run_batch(self):
        """
        run a batch episodes
        """
        # localise objects
        config = self.config
        ep_id = 0
        rospy.sleep(1.0)
        self.stats_recorder.initialize_stat()

        while not rospy.is_shutdown() and ep_id < config.MAX_EPISODES:
            # localise objects
            env = self.env
            config = self.config
            result = ''

            self.reset_env_episode(result,
                                   ep_id)  # reset env & episode variables
            env.obstacles.step()
            terminal = False
            step = 1
            navigator = WaypointNavigator()
            navigator.set_goal(env.goal_pose, ep_id)
            start_time = time.time()

            while not terminal:
                r, terminal, result = env.get_reward_and_terminate(step, 0,
                                                                   config.TIMEOUT,
                                                                   config.reward_dist_scale)
                step += 1
                env.obstacles.step()
                env.ready = False
                end_time = time.time()
                speed_next = np.asarray(env.get_self_speed())
                print('Elapsed: %.2fs, Odom: %s' % (
                    end_time - start_time, speed_next))
                env.speed_list.append(speed_next)
                while not env.ready:
                    continue
            if terminal:
                if result in ['Crashed', 'Time out']:
                    env.control_vel([0, 0])
                    navigator.cancel_goal()

            distance = np.sqrt((env.goal_pose[0] - env.init_pose[0]) ** 2 +
                               (env.goal_pose[1] - env.init_pose[1]) ** 2)
            self.logger.info(
                'Env %02d, Goal (%2.1f, %2.1f), Ep %03d, Steps %03d, Dist %2.1f, %s'
                % (
                env.index, env.goal_pose[0], env.goal_pose[1], ep_id + 1, step,
                distance, result))

            # statistics
            self.stats_recorder.store_results(step, result,
                                              ep_id)
            if (((ep_id + 1) % config.stats_print_interval) == 0) or (
                    ep_id == config.MAX_EPISODES - 1):
                self.stats_recorder.print_stats(self.env.index)
                if ep_id == config.MAX_EPISODES - 1:
                    self.stats_recorder.write_all('TEB-Navigation', self.scene_description, self.config.stageros)
            ep_id += 1
            # sleep for 2 seconds to keep consistency with RL
            rospy.sleep(2)


def main(args):
    global launch_time
    global time_str
    launch_time = datetime.now()
    logger, time_str = init_logger(debug_flag=args.debug)
    agent = Planner(time_str, main_config=OmegaConf.load(args.config), world_type=args.world, logger=logger,
                    write_outputs=args.save)

    try:
        for scene_config_file in agent.scene_config_list:
            agent.load_environment(scene_config_file, config_name=None, agent_flag=False)
            agent.run_batch()
    except KeyboardInterrupt:
        pass
    clear_temp(launch_time, time_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world',
                        required=True,
                        help='Enter world type (same as .world file)\
                            e.g. python main.py --w circle')
    parser.add_argument('--config',
                        default='./configs/main_config.yaml',
                        help='Specify relative path to config folder')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug mode for verbosity')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Enable eval mode, off exploration \
                            & model update')
    parser.add_argument('--save',
                        action='store_true',
                        help='Enable saving of logs, results & data')
    arguments = parser.parse_args()
    signal.signal(signal.SIGQUIT, signal_handler)
    main(arguments)
