from argparse import ArgumentParser
from SUMO.TrafficEnv_DDPG import TrafficEnv

import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

from models.ddpg import DDPGAgent

def main_DDPG(args):

    exp_dir = './SUMO/Single_Intersection'
    exp_type = 'binary'

    max_episode = 300
    max_epi_step = 800

    state_dim = 12
    action_dim = 1

    action_min = 5
    action_max = 20
    q_hidden = [args.q_hidden]*2
    p_hidden = [args.p_hidden]*2

    env = TrafficEnv(exp_dir, exp_type)
    agent = DDPGAgent(state_dim, action_dim, action_min, action_max,
                      q_hidden_sizes=q_hidden, p_hidden_sizes=p_hidden,
                      q_lr=args.q_lr, p_lr=args.p_lr, theta=args.theta,
                      eps=args.eps_exp)

    actor_loss_list = []
    critic_loss_list = []
    reward_list = []

    # torch.autograd.set_detect_anomaly(True)
    for episode in range(max_episode):

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)

        actor_loss_epi = []
        critic_loss_epi = []
        reward_epi = []
        action = None
        step = 0

        for epi_step in range(max_epi_step):

            # make an action based on epsilon greedy action

            action, action_norm = agent.forward(state, step)

            before_state = state

            state, reward, done = env.step(action_norm)

            state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)
            reward_epi.append(reward)

            # make a transition and save to replay memory
            transition = [before_state, action, reward, state, done]
            agent.save_memory(transition)

            if agent.train_start():
                critic_loss, actor_loss = agent.train(args.k)

                critic_loss_epi.append(critic_loss)
                actor_loss_epi.append(actor_loss)

            if done:
                if agent.train_start():
                    critic_mean = sum(critic_loss_epi) / len(critic_loss_epi)
                    actor_mean = sum(actor_loss_epi) / len(actor_loss_epi)

                    critic_loss_list.append(critic_mean)
                    actor_loss_list.append(actor_mean)
                break

            step += 1

        reward_list.append(sum(reward_epi))

        env.close()

        print(episode+1, reward_list[-1])

    return critic_loss_list, actor_loss_list, reward_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--q_hidden', type=int, default=128)
    parser.add_argument('--p_hidden', type=int, default=128)
    parser.add_argument('--q_lr', type=float, default=1e-4)
    parser.add_argument('--p_lr', type=float, default=1e-3)
    parser.add_argument('--theta', type=float, default=1e-2)
    parser.add_argument('--eps_exp', type=float, default=0.3)
    parser.add_argument('--k', type=int, default=1)
    args = parser.parse_args()
    saveroot = "./Visualization/DDPG/"
    i = 1
    path = os.path.join(saveroot, str(i))
    while True:
        if os.path.exists(path):
            i += 1
            path = os.path.join(saveroot, str(i))
        else:
            break

    os.makedirs(path)
    set_path = os.path.join(path, 'setting.txt')
    pd.Series(vars(args)).to_csv(set_path, sep='\t', header=None)

    critic_loss_list, actor_loss_list, reward_list = main_DDPG(args)
    plt.plot(critic_loss_list)
    plt.title("Critic Loss of DDPG Agent")
    fig_path = os.path.join(path, 'critic_loss.png')
    plt.savefig(fig_path)

    plt.close('all')

    plt.plot(actor_loss_list)
    plt.title("Actor Loss of DDPG Agent")
    fig_path = os.path.join(path, 'actor_loss.png')
    plt.savefig(fig_path)
    plt.close('all')

    plt.plot(reward_list)
    plt.title("Reward of DDPG Agent")
    fig_path = os.path.join(path, 'reward.png')
    plt.savefig(fig_path)
    plt.close('all')
