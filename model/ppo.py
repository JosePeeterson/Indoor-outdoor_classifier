import torch
import os
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.autograd import Variable
from model.net import MLPPolicy, CNNPolicy, LSTMPolicy, RewardShapingNetwork
from torch.optim import Adam
from utils.misc import get_filter_index
from utils.logger import init_logger


class PPO:
    def __init__(self, config, eval_flag, logger, debug_flag):

        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.logger = logger

        self.logger_ppo = init_logger(debug_flag=debug_flag, log_name='ppo')

        self.network = None
        self.optimizer = None
        self.rnd_pred_network = None
        self.rnd_target_network = None
        self.rnd_optimizer = None
        self.init_policy()

        self.action_bound = [[0, -1], [config.MAX_SPEED, 1]]
        self.train_flag = not eval_flag

        self.buffer_keys = [
            'sensor', 'goal', 'speed', 'action', 'reward', 'done', 'logprob',
            'value', 'target', 'adv', 'invisible', 'act_lstm_hx',
            'act_lstm_cx', 'cri_lstm_hx', 'cri_lstm_cx', 'rnd_target'
        ]
        self.buffer = None
        self.buffer_length = 0
        self.buffer_array = None
        self.reset_buffer()

    def init_network(self):
        config = self.config
        if config.network_type == 'cnn':
            network = CNNPolicy(frames=config.LASER_HIST,
                                action_space=2,
                                max_speed=config.MAX_SPEED,
                                use_softplus=config.USE_SOFTPLUS)
        elif config.network_type == 'mlp':
            network = MLPPolicy(obs_space=config.OBS_SIZE,
                                action_space=2,
                                max_speed=config.MAX_SPEED,
                                use_softplus=config.USE_SOFTPLUS)
        elif config.network_type == 'lstm':
            network = LSTMPolicy(
                frames=1,  #config.LASER_HIST,
                action_space=2,
                max_speed=config.MAX_SPEED,
                use_cnn=config.lstm_use_cnn,
                use_intr_feature=False,
                use_softplus=config.USE_SOFTPLUS,
                use_noisynet=config.lstm_use_noisynet)
        else:
            raise Exception('Network type not found or implemented!')

        return network

    def init_policy(self):

        config = self.config
        logger = self.logger
        policy_folder = config.policy_folder

        network = self.init_network()
        network.to(self.DEVICE)
        opt = Adam(network.parameters(), lr=config.LEARNING_RATE)

        if config.USE_RND:
            self.rnd_pred_network = RewardShapingNetwork(
                obs_space=config.OBS_SIZE, frames=config.LASER_HIST)
            self.rnd_target_network = RewardShapingNetwork(
                obs_space=config.OBS_SIZE, frames=config.LASER_HIST)
            self.rnd_optimizer = Adam(self.rnd_pred_network.parameters(),
                                      lr=config.RND_LEARNING_RATE)

            self.rnd_pred_network.to(self.DEVICE)
            self.rnd_target_network.to(self.DEVICE)

        if not os.path.exists(policy_folder):
            os.makedirs(policy_folder)

        filename = 'null' if not config.preload_filepath else config.preload_filepath
        f = policy_folder + '/' + filename
        if os.path.exists(f):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            if torch.cuda.is_available():
                state_dict = torch.load(f)
            else:
                state_dict = torch.load(f, map_location='cpu')
            try:
                network.load_state_dict(state_dict)
            except:
                raise Exception(
                    'Saved model incompatible with network selection')
            self.logger.info('Loaded model from: %s' % f)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
            logger.info('######!NO PRETRAIN MODEL LOADED!#####')

        self.network = network
        self.optimizer = opt

    def add_buffer(self,
                   state_list,
                   a,
                   r_list,
                   terminal_list,
                   logprob,
                   v,
                   invisible_list=None,
                   lstm_states=None):

        for env_state in state_list:  # iterate across each environment
            self.buffer['sensor'].append(env_state[0])
            self.buffer['goal'].append(env_state[1])
            self.buffer['speed'].append(env_state[2])

        self.buffer['action'].append(a)
        self.buffer['reward'].append(r_list)
        self.buffer['done'].append(terminal_list)
        self.buffer['logprob'].append(logprob)
        self.buffer['value'].append(v)
        self.buffer['invisible'].append(invisible_list)
        if lstm_states is not None:
            self.buffer['act_lstm_hx'].append(
                lstm_states[0].data.cpu().numpy())
            self.buffer['act_lstm_cx'].append(
                lstm_states[1].data.cpu().numpy())
            self.buffer['cri_lstm_hx'].append(
                lstm_states[2].data.cpu().numpy())
            self.buffer['cri_lstm_cx'].append(
                lstm_states[3].data.cpu().numpy())
        self.buffer_length += 1

    def reset_buffer(self):
        self.buffer = {k: [] for k in self.buffer_keys}
        self.buffer_length = 0

    def generate_action(self, env, state_list, lstm_states=None):
        s_list, goal_list, speed_list = [], [], []
        for env_state in state_list:
            s_list.append(env_state[0])
            goal_list.append(env_state[1])
            speed_list.append(env_state[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)

        s_list = self.to_tensor(s_list)
        goal_list = self.to_tensor(goal_list)
        speed_list = self.to_tensor(speed_list)
        if self.config.network_type == 'lstm':
            v, action, logprob, mean, std, _, lstm_states = self.network(
                s_list, goal_list, speed_list, lstm_states)
        else:
            v, action, logprob, mean, std = self.network(
                s_list, goal_list, speed_list)
        v, action, logprob, mean, std = v.data.cpu().numpy(), action.data.cpu(
        ).numpy(), logprob.data.cpu().numpy(), mean.data.cpu().numpy(
        ), std.data.cpu().numpy()

        if self.train_flag:
            scaled_action = np.clip(action,
                                    a_min=self.action_bound[0],
                                    a_max=self.action_bound[1])
        else:
            scaled_action = np.clip(mean,
                                    a_min=self.action_bound[0],
                                    a_max=self.action_bound[1])

        return v, action, logprob, scaled_action, std, lstm_states

    def generate_train_data(self, buff, last_value, update_step):

        rewards = np.asarray(self.buffer['reward'])
        values = np.asarray(self.buffer['value'])
        dones = np.asarray(self.buffer['done'])
        gamma = self.config.GAMMA
        lam = self.config.LAMDA

        num_step = rewards.shape[0]
        num_env = rewards.shape[1]

        # compute reward bonus
        if self.config.USE_RND:
            rewards = np.asarray(self.buffer['reward'])
            obss = np.asarray(self.buffer['sensor']).reshape(
                (num_step * num_env, self.config.LASER_HIST, -1))
            goals = np.asarray(self.buffer['goal']).reshape(
                (num_step * num_env, 2))
            speeds = np.asarray(self.buffer['speed']).reshape(
                (num_step * num_env, 2))

            obss = Variable(torch.from_numpy(obss)).float().to(self.DEVICE)
            goals = Variable(torch.from_numpy(goals)).float().to(self.DEVICE)
            speeds = Variable(torch.from_numpy(speeds)).float().to(self.DEVICE)

            rnd_target = self.rnd_target_network(obss, goals, speeds)
            if update_step >= self.config.RND_IGNORE_STEP:
                rnd_pred = self.rnd_pred_network(obss, goals, speeds)
                rnd_rew = (rnd_target - rnd_pred)**2
                rnd_rew = rnd_rew.sum(-1) * self.config.RND_REWARD_SCALE
                rnd_rew = rnd_rew.data.cpu().numpy()
                rnd_rew = rnd_rew.reshape(num_step, num_env)
                print('reward', rewards.shape, rnd_rew.shape)
                rewards = rewards + rnd_rew
            self.buffer['rnd_target'].append(rnd_target.data.cpu().numpy())

        values = list(values)
        values.append(last_value)
        values = np.asarray(values).reshape((num_step + 1, num_env))

        targets = np.zeros((num_step, num_env))
        gae = np.zeros((num_env, ))

        for t in range(num_step - 1, -1, -1):
            delta = rewards[t, :] + gamma * values[t + 1, :] * (
                1 - dones[t, :]) - values[t, :]
            gae = delta + gamma * lam * (1 - dones[t, :]) * gae

            targets[t, :] = gae + values[t, :]

        self.buffer['adv'] = targets - values[:-1, :]
        self.buffer['target'] = targets

    def copy_buffer_array(self):
        self.buffer_array = {
            k: np.asarray(v)
            for (k, v) in self.buffer.items()
        }

    def to_tensor(self, input_array):
        """
        input_array as input np.array
        """
        output_tensor = torch.tensor(input_array,
                                     requires_grad=True,
                                     dtype=torch.float).to(self.DEVICE)
        return output_tensor

    def ppo_update_stage(self, async_flag):

        config = self.config
        if not async_flag:
            filter_index = get_filter_index(self.buffer_array['done'])
        elif config.invisible_flag:
            filter_index = get_filter_index(self.buffer_array['invisible'])
        else:
            filter_index = None

        # localise variable
        batch_size = config.BATCH_SIZE
        epoch = config.EPOCH
        coeff_entropy = config.COEFF_ENTROPY
        clip_value = config.CLIP_VALUE
        num_step = config.HORIZON
        num_env = config.NUM_ENV
        frames = config.LASER_HIST
        obs_size = config.OBS_SIZE

        # localise arrays
        buffer_array = self.buffer_array
        obss = buffer_array['sensor'].reshape(
            (num_step * num_env, frames, obs_size))
        goals = buffer_array['goal'].reshape((num_step * num_env, 2))
        speeds = buffer_array['speed'].reshape((num_step * num_env, 2))
        actions = buffer_array['action'].reshape(num_step * num_env, 2)
        logprobs = buffer_array['logprob'].reshape(num_step * num_env, 1)
        targets = buffer_array['target'].reshape(num_step * num_env, 1)
        if len(buffer_array['act_lstm_hx']) > 0:
            act_lstm_hx = buffer_array['act_lstm_hx'].reshape(
                (num_step * num_env, -1))
            act_lstm_cx = buffer_array['act_lstm_cx'].reshape(
                (num_step * num_env, -1))
            cri_lstm_hx = buffer_array['cri_lstm_hx'].reshape(
                (num_step * num_env, -1))
            cri_lstm_cx = buffer_array['cri_lstm_cx'].reshape(
                (num_step * num_env, -1))
        else:
            act_lstm_hx, act_lstm_cx, cri_lstm_hx, cri_lstm_cx = None, None, None, None

        advs = buffer_array['adv']
        advs = (advs - advs.mean()) / advs.std()
        advs = advs.reshape(num_step * num_env, 1)
        if config.USE_RND:
            rnd_target = buffer_array['rnd_target'].squeeze(0)
            print('rnd_target', rnd_target.shape)
        else:
            rnd_target = None
        self.reset_buffer()
        # find filter for group terminals
        if filter_index:
            obss = np.delete(obss, filter_index, 0)
            goals = np.delete(goals, filter_index, 0)
            speeds = np.delete(speeds, filter_index, 0)
            actions = np.delete(actions, filter_index, 0)
            logprobs = np.delete(logprobs, filter_index, 0)
            advs = np.delete(advs, filter_index, 0)
            targets = np.delete(targets, filter_index, 0)

            if act_lstm_hx is not None:
                act_lstm_hx = np.delete(act_lstm_hx, filter_index, 0)
                act_lstm_cx = np.delete(act_lstm_cx, filter_index, 0)
                cri_lstm_hx = np.delete(cri_lstm_hx, filter_index, 0)
                cri_lstm_cx = np.delete(cri_lstm_cx, filter_index, 0)

        # main update
        for update in range(epoch):
            sampler = BatchSampler(SubsetRandomSampler(
                list(range(advs.shape[0]))),
                                   batch_size=batch_size,
                                   drop_last=False)
            for i, index in enumerate(sampler):
                sampled_obs = self.to_tensor(obss[index])
                sampled_goals = self.to_tensor(goals[index])
                sampled_speeds = self.to_tensor(speeds[index])
                sampled_actions = self.to_tensor(actions[index])
                sampled_logprobs = self.to_tensor(logprobs[index])
                sampled_targets = self.to_tensor(targets[index])
                sampled_advs = self.to_tensor(advs[index])
                if config.USE_RND:
                    sampled_rnd_targets = self.to_tensor(rnd_target[index])

                if self.config.network_type == 'lstm':
                    sampled_act_hx = self.to_tensor(act_lstm_hx[index])
                    sampled_act_cx = self.to_tensor(act_lstm_cx[index])
                    sampled_cri_hx = self.to_tensor(cri_lstm_hx[index])
                    sampled_cri_cx = self.to_tensor(cri_lstm_cx[index])
                    sampled_lstm_states = (sampled_act_hx, sampled_act_cx,
                                           sampled_cri_hx, sampled_cri_cx)

                    new_value, new_logprob, dist_entropy = self.network.evaluate_actions(
                        sampled_obs, sampled_goals, sampled_speeds,
                        sampled_actions, sampled_lstm_states)
                else:
                    new_value, new_logprob, dist_entropy = self.network.evaluate_actions(
                        sampled_obs, sampled_goals, sampled_speeds,
                        sampled_actions)

                if config.USE_RND:
                    rnd_preds = self.rnd_pred_network(sampled_obs,
                                                      sampled_goals,
                                                      sampled_speeds)
                    rnd_loss = F.mse_loss(rnd_preds, sampled_rnd_targets)
                    self.rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    self.rnd_optimizer.step()

                sampled_logprobs = sampled_logprobs.view(-1, 1)
                ratio = torch.exp(new_logprob - sampled_logprobs)

                sampled_advs = sampled_advs.view(-1, 1)
                surrogate1 = ratio * sampled_advs
                surrogate2 = torch.clamp(ratio, 1 - clip_value,
                                         1 + clip_value) * sampled_advs
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                sampled_targets = sampled_targets.view(-1, 1)
                value_loss = F.mse_loss(new_value, sampled_targets)

                loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                        float(value_loss.detach().cpu().numpy()), float(
                                                        dist_entropy.detach().cpu().numpy())
                self.logger_ppo.debug('{}, {}, {}'.format(
                    info_p_loss, info_v_loss, info_entropy))
