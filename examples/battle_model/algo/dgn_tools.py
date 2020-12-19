import dgl
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'

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
        actions = [torch.Tensor(sample[1]) for sample in samples]
        rewards = [torch.Tensor(sample[2]) for sample in samples]
        next_graphs = [sample[3] for sample in samples]
        dones = [sample[4] for sample in samples]

        ret_graph = dgl.batch(graphs)
        ret_action = torch.cat(actions).reshape(-1, 1)
        ret_reward = torch.cat(rewards).reshape(-1)
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

class SummaryObj:
    """Summary holder"""
    def __init__(self, log_dir, log_name):
        self.summary = dict()

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_path = log_dir

    def register(self, name_list):
        """Register summary name and start writing

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """
        for name in name_list:
            if name in self.summary.keys():
                raise Exception("Name already exists: `{}`".format(name))
            self.summary[name] = []
            f = open(os.path.join(self.log_path, name+'.txt'), 'w')
            f.close()

    def write(self, summary_dict):
        """Write summary

        Parameters
        ----------
        summary_dict: dict, summary value dict
        """
        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.item():
            if key not in self.summary.keys():
                raise Exception("Undefined name: `{}`".format(key))
            self.summary[key].append(value)
            f = open(os.path.join(self.log_path, key+'.txt'), 'a')
            f.write('%f\n' % float(value))
            f.close()

class Runner(object):
    def __init__(self, env, handles, map_size, max_steps, models,
                 play_handle, render_every=None, save_every=None, tau=None,
                 log_name=None, log_dir=None, model_dir=None, train=False):
        """Initialize runner
        Parameters
        ----------
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.tau = tau

        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'total_reward', 'kill',
                             "Sum_Reward", "Kill_Sum"]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def run(self, variant_eps, it, win_cnt=None):
        info = {'main': None, 'oppo': None}

        # pass
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        info['oppo'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}

        if self.render_every > 0:
            render = (it + 1) % self.render_every
        else:
            render = False
        rst = self.play(env=self.env, n_round=it, map_size=self.map_size,
                        max_steps=self.max_steps, handles=self.handles,
                        models=self.models, print_every=50, eps=variant_eps,
                        render=render, train=self.train)
        max_nums, nums, agent_r_records, total_rewards = rst
        for i, tag in enumerate(['main', 'oppo']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]

        if self.train:
            print('\n[INFO] {}'.format(info['main']))

            # if self.save_every and (it + 1) % self.save_every == 0:
            if info['main']['total_reward'] > info['oppo']['total_reward']:
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))
                # self-play: opponent update
                for target, param in zip(self.models[1].conv_net.parameters(),
                                         self.models[0].conv_net.parameters()):
                    target.data = (1.-self.tau)*param.data + self.tau*target.data
                for target, param in zip(self.models[1].q_net.parameters(),
                                         self.models[0].q_net.parameters()):
                    target.data = (1.-self.tau)*param.data + self.tau*target.data
                print(Color.INFO.format('[INFO] Self-play Updated!\n'))

                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models[0].save(self.model_dir + '-0', it)
                self.models[1].save(self.model_dir + '-1', it)

                self.summary.write(info['main'], it)
        else:
            print('\n[INFO] {0} \n {1}'.format(info['main'], info['oppo']))
            if info['main']['kill'] > info['oppo']['kill']:
                win_cnt['main'] += 1
            elif info['main']['kill'] < info['oppo']['kill']:
                win_cnt['oppo'] += 1
            else:
                win_cnt['main'] += 1
                win_cnt['oppo'] += 1
