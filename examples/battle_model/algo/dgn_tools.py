import dgl
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

Transition = namedtuple('Transition', ('graph', 'action', 'reward', 'next_graph', 'done'))

class GraphBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save Transitions"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)

        graphs = [sample[0] for sample in samples]
        actions = [sample[1] for sample in samples]
        rewards = [sample[2] for sample in samples]
        next_graphs = [sample[3] for sample in samples]
        dones = [sample[4] for sample in samples]

        ret_graph = dgl.batch(graphs)
        ret_action = torch.Tensor(actions).reshape(-1, 1)
        ret_reward = torch.Tensor(rewards).reshape(-1)
        ret_next_graph = dgl.batch(next_graphs)
        ret_dones = torch.Tensor(dones).reshape(-1)

        return ret_graph, ret_action, ret_reward, ret_next_graph, ret_dones

    def __len__(self):
        return len(self.memory)

class DotGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DotGATLayer, self).__init__()
        self.fc_q = nn.Linear(in_dim, out_dim)
        self.fc_k = nn.Linear(in_dim, out_dim)
        self.fc_v = nn.Linear(in_dim, out_dim)
        self.tau = 1/math.sqrt(out_dim)

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
        return {'h': h, 'alpha': alpha.squeeze()}

    def forward(self, g, z):
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('h')
        alpha = g.ndata.pop('alpha')
        dummy = g.ndata.pop('z')
        return h, alpha

class MultiHeadDotGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadDotGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        h_dim = out_dim // num_heads
        assert (h_dim*num_heads) == out_dim
        for _ in range(num_heads):
            self.heads.append(DotGATLayer(in_dim, h_dim))

    def forward(self, g, h):
        hs, alphas = map(list, zip(*[head(g, h)
                                     for head in self.heads]))
        alpha = torch.stack(alphas).mean(0)
        h = F.relu(torch.cat(hs, dim=1))
        return h, alpha

class BayesGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 se_dim=1, sigma=1e-15, sigma_0=1e15):
        super(BayesGATLayer, self).__init__()
        self.fc_q = nn.Linear(in_dim, out_dim)
        self.fc_k = nn.Linear(in_dim, out_dim)
        self.fc_v = nn.Linear(in_dim, out_dim)
        self.tau = 1/math.sqrt(out_dim)

        self.sigma = torch.tensor(sigma).type(torch.float32)
        self.se_fc1 = nn.Linear(in_dim, se_dim)
        self.se_fc2 = nn.Linear(se_dim, 1)
        self.se_act = nn.ReLU()
        self.sigma_0 = torch.tensor(sigma_0).type(torch.float32)
        self.KL_backward = 0.

    def edge_attention(self, edges):
        k = edges.src['z']
        k2 = self.se_fc1(k)
        k2 = self.se_fc2(self.se_act(k2))
        k = self.fc_k(k)
        q = self.fc_q(edges.dst['z'])
        a = (k*q).sum(-1, keepdims=True)*self.tau

        return {'e': a, 'p': k2}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'p': edges.data['p']}

    def reduce_func(self, nodes):
        s = nodes.mailbox['e']
        p = F.softmax(nodes.mailbox['p'], dim=1)
        mean_prior = torch.log(p+1e-20)
        alpha = F.softmax(s, dim=1)
        logprobs = torch.log(alpha+1e-20)
        if self.training:
            mean_posterior = logprobs - self.sigma**2 / 2
            out_weight = F.softmax(mean_posterior + self.sigma*torch.randn_like(logprobs), dim=1)
            KL = torch.log(self.sigma_0 / self.sigma + 1e-20) + (
                self.sigma**2 + (mean_posterior - mean_prior)**2) / (2 * self.sigma_0**2) - 0.5
        else:
            out_weight = alpha
            KL = torch.zeros_like(out_weight)
        v = self.fc_v(nodes.mailbox['z'])
        h = torch.sum(out_weight * v, dim=1)
        return {'h': h, 'alpha': alpha.squeeze(), 'kl': KL.mean(dim=1)}

    def forward(self, g, z):
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        self.KL_backward = g.ndata.pop('kl').mean()
        h = g.ndata.pop('h')
        alpha = g.ndata.pop('alpha')
        dummy = g.ndata.pop('z')
        return h, alpha

class BayesMultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,
                 se_dim=1, sigma=1e-15, sigma_0=1e15):
        super(BayesMultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        h_dim = out_dim // num_heads
        assert (h_dim*num_heads) == out_dim
        for _ in range(num_heads):
            self.heads.append(BayesGATLayer(in_dim, h_dim,
                                            se_dim, sigma, sigma_0))
        self.KL_backward = 0.

    def forward(self, g, h):
        hs, alphas = map(list, zip(*[head(g, h)
                                     for head in self.heads]))
        alpha = torch.stack(alphas).mean(0)
        KL = [head.KL_backward for head in self.heads]
        self.KL_backward = torch.mean(torch.stack(KL))
        h = F.relu(torch.cat(hs, dim=1))
        return h, alpha
