import os
import threading
from collections import deque

import numpy as np
import rospy
import torch
from mpi4py import MPI
from omegaconf import OmegaConf

from run_teb_navigation import Planner
from utils.collision import Collision
from utils.misc import get_group_terminal
from utils.statistics import Statistics

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class Agent:
    def __init__(self, main_config, world_type, eval_flag, logger,
                 write_outputs):

        self.comm = MPI.COMM_WORLD
        self.world_type = world_type
        self.train_flag = not eval_flag
        self.logger = logger

        self.rank = None
        self.env = None
        self.obstacle = None
        self.config = None  # Scene config
        self.async_flag = True
        self.scene_config_flag = False  # Flag for using config.yaml
        self.scene_config_list = []
        self.scene_config = None
        self.robot_name = None
        self.env_class = None
        self.scene_description = ''
        self.stats_recorder = Statistics(
            self.config, self.logger,
            write_outputs, self.time_str, main_config.single.MAX_EPISODES)  # Object to record statistics
        self.invisible_list = None
        self.parse_config(main_config)
        self.collision_checker = None

        # Config TEB is hybrid_flag is True. Only apply to scene_open world
        if self.config.USE_HYBRID:
            self.hybrid_mode = self.config.HYBRID_MODE
        else:
            self.hybrid_mode = None
        self.stuck_pos_list = []
        if self.config.USE_HYBRID and self.hybrid_mode == 'TEB' \
                and world_type in ['scene_open',
                                   'scene_straight',
                                   'scene_l',
                                   'scene_plus']:
            self.teb_agent = Planner(main_config=main_config,
                                     logger=logger,
                                     write_outputs=write_outputs)
        else:
            self.teb_agent = None

    def parse_config(self, main_config):

        world_type = self.world_type

        if world_type in ['box']:
            config = main_config.stage1
            import env.box as env_class
            config.MAX_EPISODES = 1500
            config.NUM_ENV = 12

        elif world_type in [
                'circle', 'circle_static', 'circle_test',
                'circle_static_invisible'
        ]:
            config = main_config.stage1
            import env.circle as env_class
        elif world_type in ['multi']:
            config = main_config.stage2
            import env.multi as env_class
        elif world_type in [
                'scene_open', 'scene_straight', 'scene_l', 'scene_plus'
        ]:
            config = main_config.single
            self.robot_name = config.robot
            import env.scene_all as env_class
            self.scene_config_flag = True
            file_list = sorted(os.listdir(config.config_folder))
            self.scene_config_list = [s for s in file_list if world_type in s]
        elif world_type in ['empty']:
            config = main_config.stage2
            import env.empty as env_class
            config.NUM_ENV = 2
        else:
            raise Exception('Stage Type not supported!')
        config = OmegaConf.merge(main_config.base, config)

        error_msg = 'Incorrect number of MPI,' \
                    'should be %d for %s' % (
                        config.NUM_ENV, world_type)
        assert self.comm.Get_size() == config.NUM_ENV, error_msg

        self.config = config
        self.env_class = env_class
        self.rank = self.comm.Get_rank()

        if self.rank == 0:
            self.logger.info(config.pretty())
            if self.train_flag:
                self.logger.info(
                    '##### Training Mode (Exploration + Model Update) #####')
            else:
                self.logger.info(
                    '##### Testing/Eval Mode (No Exploration + No Model Update) #####'
                )

            if config.invisible_flag:  # Hardcoded based on circle_invisible.world
                self.invisible_list = [False] * 3 + [True] * 7

    def load_environment(self, scene_config_file=None):

        config = self.config
        if scene_config_file:
            scene_path = os.path.join(config.config_folder, scene_config_file)
            scene_config = OmegaConf.load(scene_path)
            self.logger.info('Scene Config: %s' % scene_path)
            self.logger.info(scene_config.pretty())
            config_name = os.path.splitext(scene_config_file)[0]
            self.scene_config = scene_config

            self.env = self.env_class.StageWorld(config.OBS_SIZE,
                                                 scene_config,
                                                 config_name,
                                                 self.robot_name, False,
                                                 stageros=config.stageros)
            self.scene_description = config_name
        else:
            self.env = self.env_class.StageWorld(config.OBS_SIZE,
                                                 index=self.rank,
                                                 num_env=config.NUM_ENV)
            self.scene_description = ''
        if config.USE_REGION_REWARD:
            self.env.setup_region_reward()

        if self.world_type in ['multi', 'empty']:
            self.async_flag = False
        if self.world_type in [
                'scene_open', 'scene_straight', 'scene_l', 'scene_plus'
        ]:
            self.env.obstacle_flag = True

        self.stats_recorder.env = self.env

    def reset_env_episode(self, result, ep_id):
        if result != 'Reach Goal' or self.config.reset_flag:
            self.env.reset_pose()
        self.env.generate_goal_point(ep_id)
        if self.env.obstacle_flag: self.env.setup_obstacles(ep_id)

    def clear_obstacles(self):
        from env.obstacle.base import ObstacleBase
        self.obstacle = ObstacleBase(self.scene_config,'scene_l',self.robot_name,False)
        self.obstacle.remove_obstacles()
                                                 
    def get_state_observations(self, obs_stack=None):
        obs = self.env.get_laser_observation()
        self.collision_checker.update_observation(obs)
        if obs_stack is None:
            obs_stack = deque([obs, obs, obs])
        else:
            obs_stack.popleft()
            obs_stack.append(obs)

        goal = np.asarray(self.env.get_local_goal())
        speed = np.asarray(self.env.get_self_speed())
        self.env.speed_list.append(speed)
        state = [obs_stack, goal, speed]

        return state, obs_stack

    def run_batch(self, policy):  # MAIN LOOP

        # localise objects
        env = self.env
        config = self.config
        comm = self.comm

        # init batch variables
        buff = []
        global_update = 0 + config.resume_point
        global_step = 0
        result = ''
        last_v = None
        self.stats_recorder.initialize_stat()

        for ep_id in range(config.MAX_EPISODES):

            # reset env & episode variables
            self.clear_obstacles()
            self.reset_env_episode(result, ep_id)
            terminal_flag = False
            liveflag = True
            ep_reward = 0
            step = 1
            v, a, logprob, scaled_action = None, None, None, None
            real_action = None
            state, obs_stack = self.get_state_observations()
            evade = []
            if self.config.network_type == 'lstm' and env.index == 0:
                act_lstm_hx = torch.zeros(
                    (self.config.NUM_ENV, 128)).to(policy.DEVICE)
                act_lstm_cx = torch.zeros(
                    (self.config.NUM_ENV, 128)).to(policy.DEVICE)
                cri_lstm_hx = torch.zeros(
                    (self.config.NUM_ENV, 128)).to(policy.DEVICE)
                cri_lstm_cx = torch.zeros(
                    (self.config.NUM_ENV, 128)).to(policy.DEVICE)
                lstm_states = (act_lstm_hx, act_lstm_cx, cri_lstm_hx,
                               cri_lstm_cx)
            elif env.index == 0:
                lstm_states, lstm_states_next = None, None

            # Check if hybrid mode is on and need to switch to TEB
            if self.teb_agent:
                if self.collision_checker.stuck_check():
                    self.teb_agent.run_single(env, ep_id)
                    continue

            while not terminal_flag and not rospy.is_shutdown():
                # gather state observations from all agents
                state_list = comm.gather(state, root=0)

                # generate actions at rank==0
                if env.index == 0:
                    v, a, logprob, scaled_action, std, lstm_states_next = policy.generate_action(
                        env=env,
                        state_list=state_list,
                        lstm_states=lstm_states)

                # execute actions
                real_action = comm.scatter(scaled_action, root=0)
                # Hybrid

                if self.hybrid_mode == 'Control':
                    if self.collision_checker.stuck_check() or \
                            self.collision_checker.critical_check():
                        evade = self.collision_checker.control_evade()
                        real_action = np.array(evade)
                if self.teb_agent and self.collision_checker.critical_check():
                    self.teb_agent.run_single(env, ep_id)
                    continue

                if liveflag or self.async_flag:
                    env.control_vel(real_action)
                    step += 1
                if not liveflag:
                    env.control_vel([0, 0])

                # mutual collision check
                crash_mutual = self.collision_checker.mutual_collision_check(
                    liveflag)

                # get reward information
                if liveflag or self.async_flag:
                    r, terminal, result = env.get_reward_and_terminate(
                        step, crash_mutual, config.TIMEOUT,
                        config.reward_dist_scale)
                    ep_reward += r
                global_step += 1

                # get terminal info
                if terminal:
                    liveflag = False
                r_list = comm.gather(r, root=0)
                terminal_list = comm.gather(terminal, root=0)

                if self.async_flag:
                    terminal_flag = terminal
                else:
                    terminal_list = comm.bcast(terminal_list, root=0)
                    group_terminal = get_group_terminal(
                        terminal_list, env.index)
                    terminal_flag = group_terminal

                # get next state
                state_next, obs_stack = self.get_state_observations(obs_stack)

                # append additional value on last step to calculate Advantage
                state_next_list = comm.gather(state_next, root=0)
                if global_step % config.HORIZON == 0 and env.index == 0:
                    last_v, _, _, _, _, _ = policy.generate_action(
                        env=env,
                        state_list=state_next_list,
                        lstm_states=lstm_states_next)

                if self.hybrid_mode == 'Control':
                    evasion_list = comm.gather(evade, root=0)
                else:
                    evasion_list = []
                # save experience into agent 0 buffer
                if env.index == 0 and self.train_flag:
                    if self.hybrid_mode == 'Control':
                        for index, evade in enumerate(evasion_list):
                            if evade:
                                a[index] = evade
                    policy.add_buffer(state_list, a, r_list, terminal_list,
                                      logprob, v, self.invisible_list,
                                      lstm_states)
                    lstm_states = lstm_states_next

                    # update policy (background processing)
                    if policy.buffer_length > config.HORIZON - 1:
                        policy.generate_train_data(buff,
                                                   last_value=last_v,
                                                   update_step=global_update)
                        policy.copy_buffer_array()
                        update_thread = threading.Thread(
                            target=policy.ppo_update_stage,
                            kwargs=dict(async_flag=self.async_flag))
                        update_thread.setDaemon(True)
                        update_thread.start()
                        policy.reset_buffer()

                        # completion update prints
                        global_update += 1
                        self.logger.info(
                            '>>>>>>>>>>>>>>>>>>>global_update: %d' %
                            global_update)
                        if global_update != 0 and global_update % config.model_save_interval == 0:
                            torch.save(
                                policy.network.state_dict(),
                                config.policy_folder + '/{}_{}_{:05d}'.format(
                                    self.world_type, config.model_suffix,
                                    global_update))
                            self.logger.info(
                                '########################## model saved when update {} times#########'
                                '################'.format(global_update))

                state = state_next
            # print info
            distance = np.sqrt((env.goal_pose[0] - env.init_pose[0])**2 +
                               (env.goal_pose[1] - env.init_pose[1])**2)
            self.logger.info(
                'Env %02d, Goal (%2.1f, %2.1f), Ep %03d, Steps %03d, Reward %3.1f, Dist %2.1f, %s'
                % (env.index, env.goal_pose[0], env.goal_pose[1], ep_id + 1,
                   step - 1, ep_reward, distance, result))

            # statistics
            self.stats_recorder.store_results(step, result, ep_id)
            if (((ep_id + 1) % config.stats_print_interval) == 0) or\
                    (ep_id == config.MAX_EPISODES - 1):
                self.stats_recorder.print_stats(self.env.index)
                if ep_id == config.MAX_EPISODES - 1:
                    self.stats_recorder.write_all(
                        self.config.preload_filepath, self.scene_description,
                        self.config.stageros)

    def run(self, policy):
        if self.scene_config_flag:  # scene config.yaml mode
            for scene_config_file in self.scene_config_list:
                config_name = scene_config_file.split('.')[0]
                self.load_environment(scene_config_file)
                if self.teb_agent:
                    self.teb_agent.load_environment(self.scene_config,
                                                    config_name, True)
                self.collision_checker = Collision(self.env, self.config,
                                                   self.comm)
                self.run_batch(policy)
        else:
            self.load_environment()
            if self.teb_agent:
                self.teb_agent.load_environment()
            self.collision_checker = Collision(self.env, self.config,
                                               self.comm)
            self.run_batch(policy)
