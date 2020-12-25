import numpy as np
import rospy
import time
import torch
import torch.nn as nn
import copy
import os
import threading
from mpi4py import MPI
from collections import deque
from utils.misc import get_group_terminal
from omegaconf import OmegaConf
from utils.statistics import Statistics
import multiprocessing as mp
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def env_runner(rank, env_class, request, reply, config, scene_config_file=None):
    # the port id is specified in the following pattern
    os.environ["ROS_MASTER_URI"] = "http://localhost:%d/" % (config.BASE_PORT_ID + rank * config.PORT_ID_OFFSET)
    print("Agent %d Set ROS_MASTER_URI to: http://localhost:%d/" %
          (rank, config.BASE_PORT_ID + rank * config.PORT_ID_OFFSET))

    scene_path = os.path.join(config.config_folder, scene_config_file)
    scene_config = OmegaConf.load(scene_path)

    config_name = os.path.splitext(scene_config_file)[0]
    env = env_class.StageWorld(config.OBS_SIZE, scene_config, config_name,
                               stageros=config.stageros)
    if config.USE_REGION_REWARD:
        env.setup_region_reward()

    env.obstacle_flag = True
    server_ep = 0

    while True:
        if not request.empty():
            obj = request.get()
            request_id = obj[0]

            # reset
            if request_id == "reset":
                result, ep_id = obj[1], obj[2]
                if result != 'Reach Goal' or config.reset_flag:
                    env.reset_pose()
                env.generate_goal_point(ep_id)
                if env.obstacle_flag: env.setup_obstacles(ep_id)
                server_ep += 1
                reply.put((server_ep))

            elif request_id == "get_state":
                obs = env.get_laser_observation()
                goal = np.asarray(env.get_local_goal())
                speed = np.asarray(env.get_self_speed())

                reply.put((obs, goal, speed))

            elif request_id == 'get_loc_crash':
                [x, y, theta] = env.get_self_stateGT()
                crash_state = env.get_crash_state()
                reply.put((x, y, theta, crash_state))

            elif request_id == 'get_init_goal_pose':
                reply.put((env.goal_pose, env.init_pose))

            elif request_id == "step":
                real_action = obj[1]
                env.control_vel(real_action)
                reply.put((1))

            elif request_id == "get_reward":
                step = obj[1]
                crash_mutual = obj[2]
                r, terminal, result = env.get_reward_and_terminate(
                    step, crash_mutual, config.TIMEOUT, config.reward_dist_scale)
                reply.put((r, terminal, result))


class ParallelAgent():
    def __init__(self, main_config, world_type, eval_flag, logger,
                 write_outputs):

        self.main_config = main_config
        self.comm = MPI.COMM_WORLD
        self.world_type = world_type
        self.train_flag = not eval_flag
        self.logger = logger

        self.rank = None
        self.env = None
        self.config = None  #Scene config
        self.async_flag = True
        self.scene_config_flag = False  # Flag for using config.yaml
        self.parse_config()
        self.stats_recorder = Statistics(
            self.config, self.logger,
            write_outputs)  # Object to record statistics
        self.request_queue, self.reply_queue = None, None

    def parse_config(self):

        main_config = self.main_config
        world_type = self.world_type

        if world_type in ['box']:
            config = main_config.stage1
            import env.box as env_class
            config.MAX_EPISODES = 1500
            config.NUM_ENV = 12
        elif world_type in [
                'circle', 'circle_static', 'circle_test',
                'circle_test_invisible'
        ]:
            config = main_config.stage1
            import env.circle as env_class

        elif world_type in ['multi']:
            config = main_config.stage2
            import env.multi as env_class
        elif world_type in ['scene_open']:
            config = main_config.single
            import env.scene_open as env_class
            self.scene_config_flag = True
            self.scene_config_list = sorted(os.listdir(config.config_folder))
        else:
            raise Exception('Stage Type not supported!')
        config = OmegaConf.merge(main_config.base, config)

        error_msg = 'Incorrect number of MPI, should be %d for %s' % (
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

    def load_environment(self, scene_config_file=None):
        if scene_config_file:
            self.request_queue = mp.Queue(maxsize=1)
            self.reply_queue = mp.Queue(maxsize=1)
            env_proc = mp.Process(target=env_runner,
                                  args=(self.rank, self.env_class, self.request_queue,
                                        self.reply_queue, self.config, scene_config_file))
            env_proc.start()
            self.scene_description = 'scene_open'
        else:
            # self.env = self.env_class.StageWorld(config.OBS_SIZE,
            #                                      index=self.rank,
            #                                      num_env=config.NUM_ENV)
            # self.scene_description = ''
            raise NotImplementedError

        if self.world_type in ['multi']: self.async_flag = False
        # if self.world_type in ['scene_open']: self.env.obstacle_flag = True

        self.stats_recorder.env = self.env

    def reset_env_episode(self, result, ep_id):
        self.request_queue.put(("reset", result, ep_id))
        while self.reply_queue.empty():
            pass
        self.reply_queue.get()

    def get_state_observations(self, obs_stack=None):
        self.request_queue.put(("get_state",))
        while self.reply_queue.empty():
            pass
        obs, goal, speed = self.reply_queue.get()
        if obs_stack is None:
            obs_stack = deque([obs, obs, obs])
        else:
            obs_stack.popleft()
            obs_stack.append(obs)
        # self.env.speed_list.append(speed)
        state = [obs_stack, goal, speed]
        return state, obs_stack

    def take_action(self, real_action):
        self.request_queue.put(("step", real_action))
        while self.reply_queue.empty():
            pass
        self.reply_queue.get()

    def get_reward(self, step, crash_mutual):
        self.request_queue.put(("get_reward", step, crash_mutual))
        while self.reply_queue.empty():
            pass
        r, terminal, result = self.reply_queue.get()
        return r, terminal, result

    def get_goal_init_pose(self):
        self.request_queue.put(("get_init_goal_pose",))
        while self.reply_queue.empty():
            pass
        goal_pose, init_pose = self.reply_queue.get()
        return goal_pose, init_pose


    def mutual_collision_check(self, liveflag):
        comm = self.comm
        self.request_queue.put(("get_loc_crash",))
        while self.reply_queue.empty():
            pass
        x, y, theta, crash_state = self.reply_queue.get()
        crash_list = comm.gather((crash_state and liveflag),
                                 root=0)
        pose_list = copy.deepcopy(comm.gather([x, y], root=0))
        crash_mutual_list = [False] * self.config.NUM_ENV
        if self.rank == 0:
            for i, is_crashed in enumerate(crash_list):
                if is_crashed:
                    for j, y_pose in enumerate(pose_list):
                        x_pose = pose_list[i]
                        dist = np.sqrt((x_pose[0] - y_pose[0])**2 +
                                       (x_pose[1] - y_pose[1])**2)
                        if dist < 1.1:
                            crash_mutual_list[j] = True
        crash_mutual = comm.scatter(crash_mutual_list, root=0)
        return crash_mutual

    def run_batch(self, policy):  #MAIN LOOP

        # localise objects
        env = self.env
        config = self.config
        comm = self.comm

        # init batch variables
        buff = []
        global_update = 0
        global_step = 0
        result = ''
        for ep_id in range(config.MAX_EPISODES):

            # reset env & episode variables
            self.reset_env_episode(result, ep_id)
            terminal_flag = False
            liveflag = True
            ep_reward = 0
            step = 1
            v, a, logprob, scaled_action = None, None, None, None
            state, obs_stack = self.get_state_observations()
            last_time_stamp, eps_step_intervals = None, []

            if self.config.network_type == 'lstm' and self.rank == 0:
                act_lstm_hx = torch.zeros((self.config.NUM_ENV, 128)).to(policy.DEVICE)
                act_lstm_cx = torch.zeros((self.config.NUM_ENV, 128)).to(policy.DEVICE)
                cri_lstm_hx = torch.zeros((self.config.NUM_ENV, 128)).to(policy.DEVICE)
                cri_lstm_cx = torch.zeros((self.config.NUM_ENV, 128)).to(policy.DEVICE)
                lstm_states = (act_lstm_hx, act_lstm_cx, cri_lstm_hx, cri_lstm_cx)
            elif self.rank == 0:
                lstm_states, lstm_states_next = None, None
            eps_stds = []
            while not terminal_flag and not rospy.is_shutdown():
                # gather state observations from all agents
                state_list = comm.gather(state, root=0)
                # generate actions at rank==0
                if self.rank == 0:
                    v, a, logprob, scaled_action, std, lstm_states_next = policy.generate_action(
                        env=env, state_list=state_list, lstm_states=lstm_states)
                    eps_stds.append(std)
                # execute actions
                real_action = comm.scatter(scaled_action, root=0)
                if liveflag == True or self.async_flag:
                    self.take_action(real_action)
                    step += 1

                # mutual collision check
                crash_mutual = self.mutual_collision_check(liveflag)

                # get reward informtion
                if liveflag == True or self.async_flag:
                    r, terminal, result = self.get_reward(step, crash_mutual)
                    ep_reward += r
                global_step += 1

                # get terminal info
                if terminal == True:
                    liveflag = False
                r_list = comm.gather(r, root=0)
                terminal_list = comm.gather(terminal, root=0)
                if self.async_flag:
                    terminal_flag = terminal
                else:
                    terminal_list = comm.bcast(terminal_list, root=0)
                    group_terminal = get_group_terminal(
                        terminal_list, self.rank)
                    terminal_flag = group_terminal

                # get next state
                state_next, obs_stack = self.get_state_observations(obs_stack)
                cur_time_stamp = time.time()
                if last_time_stamp is not None:
                    eps_step_intervals.append(cur_time_stamp - last_time_stamp)
                last_time_stamp = cur_time_stamp
                # append additional value on last step to calculate Advantage
                state_next_list = comm.gather(state_next, root=0)
                if global_step % config.HORIZON == 0 and self.rank == 0:
                    last_v, _, _, _, _, _ = policy.generate_action(
                        env=env, state_list=state_next_list,
                        lstm_states=lstm_states_next)

                # save experience into agent 0 buffer
                if self.rank == 0 and self.train_flag:
                    policy.add_buffer(state_list, a, r_list, terminal_list,
                                      logprob, v, None, lstm_states)
                    lstm_states = lstm_states_next
                    # update policy (background processing)
                    if policy.buffer_length > config.HORIZON - 1:
                        policy.generate_train_data(buff, last_value=last_v, update_step=global_update)
                        policy.copy_buffer_array()
                        update_thread = threading.Thread(
                            target=policy.ppo_update_stage,
                            kwargs=dict(async_flag=self.async_flag))
                        update_thread.setDaemon(True)
                        update_thread.start()

                        # completion update prints
                        while len(policy.buffer['sensor']) != 0:
                            pass
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
            goal_pose, init_pose = self.get_goal_init_pose()
            distance = np.sqrt((goal_pose[0] - init_pose[0])**2 +
                               (goal_pose[1] - init_pose[1])**2)
            eps_step_intervals = np.array(eps_step_intervals) if len(eps_step_intervals) > 0 else np.zeros((1))
            eps_stds = np.array(eps_stds) if len(eps_stds) > 0 else np.zeros((1))
            self.logger.info(
                'Env %02d, Goal (%2.1f, %2.1f), Ep %03d, Steps %03d, Reward %3.1f, Dist %2.1f, %s, step-interval: %.2f (%.2f), std: %.2f'
                % (self.rank, goal_pose[0], goal_pose[1], ep_id + 1,
                   step, ep_reward, distance, result, eps_step_intervals.mean(), eps_step_intervals.std(), eps_stds.mean()))

            # statistics
            # self.stats_recorder.store_results(step, terminal_flag, result,
            #                                   ep_id)
            # if (((ep_id + 1) % config.stats_print_interval)
            #         == 0) or (ep_id == config.MAX_EPISODES - 1):
            #     self.stats_recorder.print_stats(
            #         "%s_%s" % (self.world_type, self.scene_description),
            #         self.rank)
            #     if ep_id == config.MAX_EPISODES - 1:
            #         self.stats_recorder.write_stats(
            #             self.config.preload_filepath, self.scene_description,
            #             self.config.stageros)

    def run(self, policy):
        if self.scene_config_flag:  # scene config.yaml mode
            for scene_config_file in self.scene_config_list:
                self.load_environment(scene_config_file)
                self.run_batch(policy)
        else:
            self.load_environment()
            self.run_batch(policy)