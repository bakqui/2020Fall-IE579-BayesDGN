import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from utils.replay_buffer import ReplayBuffer

def mlp(sizes, activation):
    layers = []
    L = len(sizes)
    for i in range(L-1):
        if i != (L-2):
            act = activation
        else:
            act = nn.Identity
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def get_batch(experiences):
    assert len(experiences[0]) == 5
    batch_state = torch.cat([ex[0] for ex in experiences], 0).float()
    batch_act = torch.as_tensor([ex[1] for ex in experiences]).unsqueeze(1)
    batch_reward = torch.as_tensor([ex[2] for ex in experiences]).unsqueeze(1)
    batch_next = torch.cat([ex[3] for ex in experiences], 0).float()
    batch_mask = 1 - torch.as_tensor([ex[4] for ex in experiences]).int().unsqueeze(1)
    return batch_state, batch_act, batch_reward, batch_next, batch_mask

class DQNAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: list = [128, 128], activation=nn.ReLU,
                 buffer_size: int = 1000000, batch_size: int = 32,
                 lr: float = 1e-4, gamma: float = 0.95,
                 theta: float = 0.05):
        super(DQNAgent, self).__init__()
        self.q_net = mlp([state_dim]+hidden_sizes+[action_dim],
                         activation=activation)
        self.target_net = mlp([state_dim]+hidden_sizes+[action_dim],
                              activation=activation)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.theta = theta

    def forward(self, x):
        return self.q_net(x)

    def save_memory(self, ex):
        self.buffer.push(ex)

    def train(self, k=4, max_norm=5.):
        losses = []
        for _ in range(k):
            experiences = self.buffer.sample(self.batch_size)
            s, a, r, t, mask = get_batch(experiences)
            next_q = self.target_net(t).max(-1, keepdim=True)[0]
            target = r + self.gamma*mask*next_q.detach()
            pred = self.q_net(s).gather(-1, a)
            loss = F.mse_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.q_net.parameters(), max_norm)
            self.optimizer.step()
            losses.append(loss.item())
        self.target_update()
        return np.mean(losses)

    def train_start(self):
        return (len(self.buffer) >= self.batch_size)

    def target_update(self):
        for target, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target.data = (1-self.theta)*target.data + self.theta*param.data

#%%