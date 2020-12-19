import dgl
import math
import numpy as np
import random
import torch


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)

def get_edges(feature, n_agents, n_neighbor=3):
    from_idx = [] # source
    to_idx = [] # destination
    dis = []
    for src in range(n_agents):
        x, y = feature[src][-2], feature[src][-1]
        dis.append((x, y, src))
    for src in range(n_agents):
        f = []
        for dst in range(n_agents):
            distance = (dis[dst][0]-dis[src][0])**2+(dis[dst][1]-dis[src][1])**2
            f.append([distance, dst])
        f.sort(key=lambda x: x[0]) # sort w.r.t. distance
        for order in range(n_neighbor+1):
            from_idx.append(src)
            to_idx.append(f[order][1])
    return from_idx, to_idx

def observation(view, feature, n_agents):
    obs = []
    for j in range(n_agents):
        obs.append(np.hstack(((view[j][:, :, 1]-view[j][:, :, 5]).flatten(),
                              feature[j][-1:-3:-1])))
    return obs

def gen_graph(view, feature, n_neighbor=3):
    """get state as a graph"""
    g = dgl.DGLGraph()

    n_agents = len(feature)
    g.add_nodes(n_agents)

    from_idx, to_idx = get_edges(feature, n_agents, n_neighbor)
    g.add_edges(from_idx, to_idx)

    # we save observation as the feature of the nodes
    obs = observation(view, feature, n_agents)
    g.ndata['obs'] = torch.Tensor(obs) # shape = (n_agents, view_size**2 + 2)

    return g

# self-play (training a model by playing with itself)
def play(env, n_round, map_size, max_steps, handles, models, eps,
         print_every, n_neighbor=3, render=False, train=True):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    before_state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    # get graph from observation of each group
    for i in range(n_group):
        view, feature = env.get_observation(handles[i])
        state[i] = gen_graph(view, feature, n_neighbor)

    while not done and step_ct < max_steps:
        # take actions for every group
        for i in range(n_group):
            acts[i] = models[i].act(graph=state[i], epsilon=eps)
            before_state[i] = state[i]

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # next graph
        for i in range(n_group):
            view, feature = env.get_observation(handles[i])
            state[i] = gen_graph(view, feature, n_neighbor)

        buffer = {
            'g': before_state[0], 'a': acts[0], 'r': rewards[0],
            'n_g': state[0], 't': done
        }

        # save experience
        if train:
            models[0].save_samples(**buffer)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # train model after an episode ends
    if train:
        models[0].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle(env, n_round, map_size, max_steps, handles, models, eps,
           print_every, n_neighbor=3, render=False):
    """play a ground"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            view, feature = env.get_observation(handles[i])
            state[i] = gen_graph(view, feature, n_neighbor)

        for i in range(n_group):
            acts[i] = models[i].act(graph=state[i], epsilon=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
