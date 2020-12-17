'''
Input

env     : Stochastic game
T       : Total training time

Output

state-value function V*
action-value function Q*
'''
def f(agent, q_values):
    pass

def train(env, gamma, lr, T):
    global N
    agents = [None for i in range(N)] # defined agents
    state = env.reset()

    for i in range(T):
        actions = [agent.get_action(state) for agent in agents]
        # observations = [agent.get_obs(state) for agent in agents]
        # actions = [agent.get_action(obs) for agent, obs
        #            in zip(agents, observations)]
        Qs = [agent(state) for agent in agents]
        # Qs = [agent(state, action) for agent, action
        #       in zip(agents, actions)]
        next_state, rewards, dones, infos = env.step(actions)
        # next_observations = [agent.get_obs(next_state) for agent in agents]

        Vs = [f(agent, Qs) for agent in agents]
        for agent, r, v in zip(agents, rewards, Vs):
            agent.update(state, actions, r, next_state, v, dones)
        # for obs, agent, r, next_obs, v in zip(observations, agents, rewards,
        #                                       next_observations, Vs):
        #     agent.update(obs, actions, r, next_obs, v, dones)

        done = all(dones)
        if done:
            # break
            state = env.reset()

        state - next_state
