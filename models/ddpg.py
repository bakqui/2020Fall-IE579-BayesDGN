import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from utils.replay_buffer import ReplayBuffer

def mlp(sizes, activation, bn=False):
    layers = []
    L = len(sizes)
    for i in range(L-1):
        if i != (L-2):
            act = activation
            if bn:
                layers += [nn.Linear(sizes[i], sizes[i+1]), nn.BatchNorm1d(sizes[i+1]), act()]
            else:
                layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
        else:
            layers += [nn.Linear(sizes[i], sizes[i+1])]
    return nn.Sequential(*layers)

def get_batch(experiences):
    assert len(experiences[0]) == 5
    batch_state = torch.cat([ex[0] for ex in experiences], 0).float()
    batch_act = torch.as_tensor([ex[1] for ex in experiences]).unsqueeze(1)
    batch_reward = torch.as_tensor([ex[2] for ex in experiences]).unsqueeze(1)
    batch_next = torch.cat([ex[3] for ex in experiences], 0).float()
    batch_mask = 1 - torch.as_tensor([ex[4] for ex in experiences]).int().unsqueeze(1)
    return batch_state, batch_act, batch_reward, batch_next, batch_mask

class Actor(nn.Module):
    def __init__(self, act_loc, act_scale, state_dim: int, action_dim: int,
                 hidden_sizes: list, activation=nn.ReLU, bn=True):
        super(Actor, self).__init__()
        self.policy_net = mlp([state_dim]+hidden_sizes+[action_dim],
                              activation=activation, bn=bn)
        self.loc = act_loc
        self.scale = act_scale

    def forward(self, x):
        x = self.policy_net(x)
        # Squashing action to minimum / maximum value
        x = self.scale*F.tanh(x)+self.loc
        return x

# Critic to estimate action-value function
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: list, activation=nn.ReLU):
        super(Critic, self).__init__()
        assert len(hidden_sizes) > 1
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0]+action_dim, hidden_sizes[1])
        self.net = mlp(hidden_sizes[1:]+[1], activation=activation)

    def forward(self, x, a):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(torch.cat([x, a], dim=-1)))
        return self.net(x)

class DDPGAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 action_min: float, action_max: float,
                 q_hidden_sizes: list = [128, 128],
                 p_hidden_sizes: list = [128, 128],
                 activation=nn.ReLU,
                 buffer_size: int = 1000000, batch_size: int = 32,
                 q_lr: float = 1e-4, p_lr: float = 1e-3, gamma: float = 0.95,
                 theta: float = 0.01,
                 eps: float = 0.3):
        super(DDPGAgent, self).__init__()

        # Actor, frozen actor
        # Bound the action to minimum, maximum value using tanh
        loc = (action_min+action_max)/2
        scale = (action_min-action_max)/2
        self.policy = Actor(loc, scale, state_dim, action_dim, p_hidden_sizes,
                            activation=activation, bn=True)
        self.policy_target = Actor(loc, scale, state_dim, action_dim, p_hidden_sizes,
                                   activation=activation, bn=True)
        self.policy_target.load_state_dict(self.policy.state_dict())

        # Critic, frozen critic
        self.q_net = Critic(state_dim, action_dim, q_hidden_sizes, activation)
        self.q_target = Critic(state_dim, action_dim, q_hidden_sizes, activation)
        self.q_target.load_state_dict(self.q_net.state_dict())

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # Learner
        self.q_optimizer = Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=p_lr)
        self.gamma = gamma

        # Polyak averaging parameter
        self.theta = theta

        # Exploration coefficient
        self.eps = eps

        # To use batchnorm
        self.policy.eval()
        self.policy_target.eval()

    # Get action
    def forward(self, x, step):
        x = self.policy(x)
        x_exp = x.clone() + self.eps*torch.randn(x.shape)
        return x, x_exp

    def save_memory(self, ex):
        self.buffer.push(ex)

    def train(self, k=1, q_max_norm=5., policy_max_norm=5.):
        q_losses = []

        # q update
        # To stabilize the learning,
        # critic is updated several times per each training step
        # and gradient clipping is also used
        for _ in range(k):
            experiences = self.buffer.sample(self.batch_size)
            s, a, r, t, mask = get_batch(experiences)
            mu_t = self.policy_target(t)
            next_q = self.q_target(t, mu_t)
            target = r + self.gamma*mask*next_q.detach()
            pred = self.q_net(s, a)
            q_loss = F.mse_loss(pred, target)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            clip_grad_norm_(self.q_net.parameters(), q_max_norm)
            self.q_optimizer.step()
            q_losses.append(q_loss.item())

        # policy update
        # To stablize the learning,
        # batchnorm and gradient clipping is used
        self.policy.train()
        mu = self.policy(s)
        policy_loss = torch.mean(-self.q_net(s, mu))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy.parameters(), policy_max_norm)
        self.policy_optimizer.step()

        # Polyak averaging
        self.target_update()

        # batchnorm
        self.policy.eval()
        return np.mean(q_losses), policy_loss.item()

    def train_start(self):
        return (len(self.buffer) >= self.batch_size)

    def target_update(self):
        for target, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
        for target, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data
