import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dgn_tools import GraphBuffer
from torch.optim import Adam

class DotGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, save_weights=False):
        super(DotGATLayer, self).__init__()
        self.fc_q = nn.Linear(in_dim, out_dim)
        self.fc_k = nn.Linear(in_dim, out_dim)
        self.fc_v = nn.Linear(in_dim, out_dim)
        self.tau = 1/math.sqrt(out_dim)
        if save_weights:
            self.weights = []
        self.save_weights = save_weights

    def edge_attention(self, edges):
        k = self.fc_k(edges.src['z'])
        q = self.fc_q(edges.dst['z'])
        a = (k*q).sum(-1, keepdims=True)*self.tau
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        s = nodes.mailbox['e']
        alpha = F.softmax(s, dim=1)
        v = self.fc_v(nodes.mailbox['z'])
        h = torch.sum(alpha * v, dim=1)
        if self.save_weights:
            self.weights.append(alpha.squeeze(-1))
        return {'h': h}

    def forward(self, g, z):
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('h')
        g.ndata.pop('z')
        g.edata.pop('e')
        return h

    def get_weights(self):
        if self.save_weights:
            return self.weights
        else:
            return []

    def clean_weights(self):
        if self.save_weights:
            self.weights = []

class MultiHeadDotGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, save_weights=False):
        super(MultiHeadDotGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        h_dim = out_dim // num_heads
        assert (h_dim*num_heads) == out_dim
        for _ in range(num_heads):
            self.heads.append(DotGATLayer(in_dim, h_dim, save_weights))

        self.save_weights = save_weights

    def forward(self, g, h):
        hs = [head(g, h) for head in self.heads]
        h = F.relu(torch.cat(hs, dim=1))
        return h

    def get_weights(self):
        outputs = []
        if self.save_weights:
            n_oper = len(self.heads[0].weights)
            for oper in range(n_oper):
                output = [head.weights[oper] for head in self.heads]
                outputs.append(torch.stack(output).mean(0))
        return outputs

    def clean_weights(self):
        for head in self.heads:
            head.clean_weights()

class ObsEncoder(nn.Module):
    def __init__(self, in_dim, o_dim=128, h_dim=512):
        super(ObsEncoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, o_dim)

    def forward(self, o):
        o = F.relu(self.fc1(o))
        o = F.relu(self.fc2(o))
        return o

class DGN_Conv(nn.Module):
    def __init__(self, obs_dim, h_dim=128, num_heads=8,
                 target=False):
        super(DGN_Conv, self).__init__()
        self.encoder = ObsEncoder(in_dim=obs_dim, o_dim=h_dim)
        self.conv1 = MultiHeadDotGATLayer(h_dim, h_dim, num_heads, False)
        if target:
            self.conv2 = MultiHeadDotGATLayer(h_dim, h_dim, num_heads, False)
        else:
            self.conv2 = MultiHeadDotGATLayer(h_dim, h_dim, num_heads, True)
        self.target = target

    def forward(self, graph):
        obs = graph.ndata['obs']
        z1 = self.encoder(obs)
        z2 = self.conv1(graph, z1)
        z3 = self.conv2(graph, z2)
        out = torch.cat([z1, z2, z3], dim=1)
        return out

    def get_weights(self):
        return self.conv2.get_weights()

    def clean_weights(self):
        self.conv2.clean_weights()

class DGNregAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, h_dim=128,
                 num_heads=8, gamma=0.95, batch_size=64,
                 buffer_size=80000, lr=1e-4, neighbors=3,
                 lamb=0.03, beta=0.01, *args, **kwargs):
        super(DGNregAgent, self).__init__()
        self.conv_net = DGN_Conv(obs_dim, h_dim, num_heads)
        self.target_conv = DGN_Conv(obs_dim, h_dim, num_heads, target=True)
        self.q_net = nn.Linear(3*h_dim, act_dim)
        self.target_q = nn.Linear(3*h_dim, act_dim)
        self.target_conv.load_state_dict(self.conv_net.state_dict())
        self.target_q.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(list(self.conv_net.parameters())+list(self.q_net.parameters()),
                              lr=lr)
        self.beta = beta
        self.gamma = gamma
        self.buffer = GraphBuffer(buffer_size)
        self.batch_size = batch_size

        self.n_act = act_dim
        self.n_neighbor = neighbors
        self.lamb = lamb

        self._new_add = 0

    def act(self, graph, epsilon):
        if random.random() < epsilon:
            action = torch.randint(0, self.n_act, size=(graph.num_nodes(),))
        else:
            q_value, _ = self.get_q(graph)
            action = q_value.argmax(dim=-1).detach()

        return action.numpy().astype(np.int32)

    def get_q(self, graph):
        z = self.conv_net(graph)
        q = self.q_net(z)
        weights = self.conv_net.get_weights()
        self.conv_net.clean_weights()
        return q, weights

    def get_target(self, graph):
        z = self.target_conv(graph)
        q = self.target_q(z)
        return q

    def save_samples(self, g, a, r, n_g, t):
        self.buffer.push(g, a, r, n_g, t)
        self._new_add += 1

    def train(self):
        batch_num = self._new_add * 2 // self.batch_size
        for _ in range(batch_num):
            state, act, reward, n_state, done = self.buffer.sample(self.batch_size)
            curr_qs, curr_weights = self.get_q(state)
            selected_qs = curr_qs.gather(1, act).reshape(-1)
            next_qs = self.get_target(n_state).max(dim=1)[0].detach()
            target = reward + self.gamma * next_qs * (1 - done)

            _, next_weights = self.get_q(n_state)
            n_oper = len(curr_weights)
            KL = 0
            for oper in range(n_oper):
                KL += (curr_weights[oper] * torch.log(curr_weights[oper]/next_weights[oper])).sum(-1)

            loss = F.mse_loss(selected_qs, target) + self.lamb*KL.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update()
        self._new_add = 0

    def update(self):
        for target, param in zip(self.target_conv.parameters(),
                                 self.conv_net.parameters()):
            target.data = (1-self.beta)*target.data + self.beta*param.data

        for target, param in zip(self.target_q.parameters(),
                                 self.q_net.parameters()):
            target.data = (1-self.beta)*target.data + self.beta*param.data

    def save(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "dgn_r_{}.pt".format(step))
        torch.save({
            'conv_state_dict': self.conv_net.state_dict(),
            'q_state_dict': self.q_net.state_dict()}, file_path)

    def load(self, dir_path, step=0):
        file_path = os.path.join(dir_path, "dgn_r_{}.pt".format(step))
        checkpoint = torch.load(file_path)
        self.q_net.load_state_dict(checkpoint['q_state_dict'])
        self.conv_net.load_state_dict(checkpoint['conv_state_dict'])
