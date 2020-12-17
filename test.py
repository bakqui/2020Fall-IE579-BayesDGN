### Training (Do not revise this code) ###
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

from models.dqn import DQNAgent

def main_DQN():

    # exp_dir = './SUMO/Single_Intersection'
    # exp_type = 'binary'

    max_episode = 300
    max_epi_step = 800

    epsilon = 0.9
    epsilon_min = 0.005
    decay_rate = 0.005

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # state_dim = 12
    # action_dim = 2

    # env = TrafficEnv(exp_dir, exp_type)

    agent = DQNAgent(state_dim, action_dim)
    print(agent.q_net)
    loss_list = []
    reward_list = []

    for episode in range(max_episode):

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)

        loss_epi = []
        reward_epi = []
        action = None

        for epi_step in range(max_epi_step):

            # make an action based on epsilon greedy action
            before_action = action

            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                action = agent(state)
                action = np.argmax(action.detach().numpy())

            before_state = state

            state, reward, done, _ = env.step(action)

            state = torch.tensor(state, dtype=torch.float32).reshape(-1, state_dim)
            reward_epi.append(reward)

            # make a transition and save to replay memory
            transition = [before_state, action, reward, state, done]
            agent.save_memory(transition)

            if agent.train_start():
                loss = agent.train()
                loss_epi.append(loss)

            if done:
                break

        if epsilon > epsilon_min:
            epsilon -= decay_rate
        else:
            epsilon = epsilon_min

        if agent.train_start():
            loss_list.append(sum(loss_epi)/len(loss_epi))

        env.close()
        reward_list.append(sum(reward_epi))

        print(episode+1, reward_list[-1])

    return loss_list, reward_list


if __name__ == "__main__":

    loss_list, reward_list = main_DQN()

    plt.plot(loss_list)
    plt.title("Loss of DQN Agent")
    plt.savefig("./Visualization/DQN/loss.png")
    plt.close('all')

    plt.plot(reward_list)
    plt.title("Reward of DQN Agent")
    plt.savefig("./Visualization/DQN/reward.png")
    plt.show()
    plt.close('all')