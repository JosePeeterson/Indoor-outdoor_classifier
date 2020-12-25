import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.misc import log_normal_density
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, in_tensor):
        assert torch.is_tensor(in_tensor)
        return in_tensor.view(in_tensor.shape[0], 1, -1)


class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space, max_speed, use_softplus=False):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space),
                                   requires_grad=True)
        self.use_softplus = use_softplus

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames,
                                     out_channels=32,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.act_fc1 = nn.Linear(128 * 32, 256)
        self.act_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)
        if self.use_softplus:
            self.actor_std = nn.Linear(128, action_space)
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames,
                                     out_channels=32,
                                     kernel_size=5,
                                     stride=2,
                                     padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.crt_fc1 = nn.Linear(128 * 32, 256)
        self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.critic = nn.Linear(128, 1)

        if use_softplus:
            torch.nn.init.xavier_uniform_(self.act_fc1.weight)
            torch.nn.init.xavier_uniform_(self.act_fc2.weight)
            torch.nn.init.xavier_uniform_(self.actor1.weight)
            torch.nn.init.xavier_uniform_(self.actor2.weight)
            torch.nn.init.xavier_uniform_(self.actor_std.weight)
            torch.nn.init.xavier_uniform_(self.crt_fc1.weight)
            torch.nn.init.xavier_uniform_(self.crt_fc2.weight)
            torch.nn.init.xavier_uniform_(self.critic.weight)
        self.max_speed = max_speed

    def forward(self, x, goal, speed, action=None):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = self.max_speed * torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        if self.use_softplus:
            std = F.softplus(self.actor_std(a)) + 0.001
            log_std = torch.log(std)
        else:
            log_std = self.logstd.expand_as(mean)
            std = torch.exp(log_std)

        if action is None:
            action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=log_std)

        # value
        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean, std

    def evaluate_actions(self, x, goal, speed, action):
        v, _, logprob, _, std = self.forward(x, goal, speed, action)
        # evaluate
        logstd = torch.log(std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        del x, goal, speed, action

        return v, logprob, dist_entropy


class LSTMPolicy(nn.Module):
    def __init__(self,
                 frames,
                 action_space,
                 max_speed,
                 use_cnn=True,
                 use_intr_feature=True,
                 use_softplus=False,
                 use_noisynet=True,
                 lstm_latent_dim=128):
        super(LSTMPolicy, self).__init__()

        self.use_intr_feature = use_intr_feature
        self.use_softplus = use_softplus
        self.use_cnn = use_cnn
        fc_class = NoisyLinear if use_noisynet else nn.Linear

        self.logstd = nn.Parameter(torch.zeros(action_space))

        if use_cnn:
            self.act_fea_cv1 = nn.Conv1d(in_channels=frames,
                                         out_channels=32,
                                         kernel_size=5,
                                         stride=2,
                                         padding=1)
            self.act_fea_cv2 = nn.Conv1d(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
            self.act_fc1 = fc_class(128 * 32, lstm_latent_dim)
        else:
            self.act_fc1 = fc_class(512, lstm_latent_dim)
        if use_intr_feature:
            self.act_goal_fc = fc_class(2, lstm_latent_dim)
            self.act_speed_fc = fc_class(2, lstm_latent_dim)
            self.act_fc2 = fc_class(lstm_latent_dim * 1, 128)
        else:
            self.act_fc2 = fc_class(lstm_latent_dim + 4, 128)
        self.actor1 = fc_class(128, 1)
        self.actor2 = fc_class(128, 1)
        if use_softplus:
            self.actor_std = fc_class(128, action_space)

        if use_cnn:
            self.crt_fea_cv1 = nn.Conv1d(in_channels=frames,
                                         out_channels=32,
                                         kernel_size=5,
                                         stride=2,
                                         padding=1)
            self.crt_fea_cv2 = nn.Conv1d(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)
            self.crt_fc1 = fc_class(128 * 32, lstm_latent_dim)
        else:
            self.crt_fc1 = fc_class(512, lstm_latent_dim)
        if use_intr_feature:
            self.crt_goal_fc = fc_class(2, lstm_latent_dim)
            self.crt_speed_fc = fc_class(2, lstm_latent_dim)
            self.crt_fc2 = fc_class(lstm_latent_dim * 1, 128)
        else:
            self.crt_fc2 = fc_class(lstm_latent_dim + 4, 128)
        self.critic = fc_class(128, 1)

        self.act_lstm = nn.LSTMCell(lstm_latent_dim, lstm_latent_dim)
        self.crt_lstm = nn.LSTMCell(lstm_latent_dim, lstm_latent_dim)

        self.act_lstm.bias_ih.data.fill_(0.)
        self.act_lstm.bias_hh.data.fill_(0.)
        self.crt_lstm.bias_ih.data.fill_(0.)
        self.crt_lstm.bias_hh.data.fill_(0.)
        torch.nn.init.xavier_uniform_(self.act_fc1.weight)
        if use_intr_feature:
            torch.nn.init.xavier_uniform_(self.act_goal_fc.weight)
            torch.nn.init.xavier_uniform_(self.act_speed_fc.weight)
            torch.nn.init.xavier_uniform_(self.crt_goal_fc.weight)
            torch.nn.init.xavier_uniform_(self.crt_speed_fc.weight)
        torch.nn.init.xavier_uniform_(self.act_fc1.weight)
        torch.nn.init.xavier_uniform_(self.act_fc2.weight)
        torch.nn.init.xavier_uniform_(self.actor1.weight)
        torch.nn.init.xavier_uniform_(self.actor2.weight)
        if use_softplus:
            torch.nn.init.xavier_uniform_(self.actor_std.weight)
        torch.nn.init.xavier_uniform_(self.crt_fc1.weight)
        torch.nn.init.xavier_uniform_(self.crt_fc2.weight)
        torch.nn.init.xavier_uniform_(self.critic.weight)
        self.max_speed = max_speed

    def forward(self, x, goal, speed, lstm_states, action=None):
        """
            returns value estimation, action, log_action_prob
        """
        actor_lstm_h, actor_lstm_c, critic_lstm_h, critic_lstm_c = lstm_states
        # get the last frame
        x = x[:, -2:-1, :]
        if self.use_cnn:
            a = F.relu(self.act_fea_cv1(x))
            a = F.relu(self.act_fea_cv2(a))
        else:
            a = x
        a = a.view(a.shape[0], -1)
        a_x = F.relu(self.act_fc1(a))
        a_hx, a_cx = self.act_lstm(a_x, (actor_lstm_h, actor_lstm_c))

        if self.use_intr_feature:
            a_g = F.relu(self.act_goal_fc(goal))
            a_v = F.relu(self.act_speed_fc(speed))
            a = a_hx * a_g * a_v
        else:
            a = torch.cat((a_hx, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = self.max_speed * torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        if self.use_softplus:
            std = F.softplus(self.actor_std(a)) + 0.001
            logstd = torch.log(std)
        else:
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)

        # sample an action
        if action is None:
            action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        # value
        if self.use_cnn:
            v = F.relu(self.crt_fea_cv1(x))
            v = F.relu(self.crt_fea_cv2(v))
        else:
            v = x
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v_hx, v_cx = self.crt_lstm(v, (critic_lstm_h, critic_lstm_c))

        if self.use_intr_feature:
            v_goal = F.relu(self.crt_goal_fc(goal))
            v_speed = F.relu(self.crt_speed_fc(speed))
            v = v_hx * v_goal * v_speed
        else:
            v = torch.cat((v_hx, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()

        return v, action, logprob, mean, std, dist_entropy, (a_hx, a_cx, v_hx,
                                                             v_cx)

    def evaluate_actions(self, x, goal, speed, action, lstm_states):
        actor_lstm_h, actor_lstm_c, critic_lstm_h, critic_lstm_c = lstm_states
        v, _, logprob, mean, std, dist_entropy, _ = self.forward(
            x, goal, speed, lstm_states, action)

        return v, logprob, dist_entropy


class MLPPolicy(nn.Module):
    def __init__(self, obs_space, action_space, max_speed):
        super(MLPPolicy, self).__init__()
        # action network
        self.act_fc1 = nn.Linear(obs_space, 64)
        self.act_fc2 = nn.Linear(64, 128)
        self.mu = nn.Linear(128, action_space)
        with torch.no_grad():
            self.mu.weight = torch.mul(self.mu.weight, 0.1)
        # torch.log(std)
        self.logstd = nn.Parameter(torch.zeros(action_space),
                                   requires_grad=True)

        # value network
        self.value_fc1 = nn.Linear(obs_space, 64)
        self.value_fc2 = nn.Linear(64, 128)
        self.value_fc3 = nn.Linear(128, 1)
        with torch.no_grad():
            self.value_fc3.weight = torch.mul(self.value_fc3.weight, 0.1)
        self.max_speed = max_speed

    def forward(self, x):
        """
            returns value estimation, a, log_action_prob
        """
        # action
        act = self.act_fc1(x)
        act = F.tanh(act)
        act = self.act_fc2(act)
        act = F.tanh(act)
        mean = self.mu(act)  # N, num_actions
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        a = torch.normal(mean, std)

        # value
        v = self.value_fc1(x)
        v = F.tanh(v)
        v = self.value_fc2(v)
        v = F.tanh(v)
        v = self.value_fc3(v)

        # action prob on log scale
        logprob = log_normal_density(a, mean, std=std, log_std=logstd)
        return v, a, logprob, mean

    def evaluate_actions(self, x, action):
        v, _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features,
                                          bias=True)  # TODO: Adapt for no bias

        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight',
                             torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def reset_parameters(self):
        if hasattr(
                self, 'sigma_weight'
        ):  # Only init after all params added (otherwise super().__init__() fails)
            nn.init.uniform_(self.weight, -math.sqrt(3 / self.in_features),
                             math.sqrt(3 / self.in_features))
            nn.init.uniform_(self.bias, -math.sqrt(3 / self.in_features),
                             math.sqrt(3 / self.in_features))
            nn.init.constant_(self.sigma_weight, self.sigma_init)
            nn.init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(
            input,
            self.weight + self.sigma_weight * Variable(self.epsilon_weight),
            self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

        self.epsilon_weight = self.epsilon_weight.to(self.DEVICE)
        self.epsilon_bias = self.epsilon_bias.to(self.DEVICE)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class RewardShapingNetwork(nn.Module):
    def __init__(self, obs_space, frames):
        super(RewardShapingNetwork, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=frames, out_channels=16, kernel_size=5, stride=2, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(obs_space, 256)
        self.fc2 = nn.Linear(256 + 2, 128)
        self.fc3 = nn.Linear(128, 64)

        # initialize
        # torch.nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        # torch.nn.init.zeros_(self.conv1.bias)
        # torch.nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        # torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x, goal, speed):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = x[:, -1, :]
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x + 0.5))
        x = torch.cat([x, goal], dim=-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
