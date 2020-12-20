import argparse
import os
import magent

from examples.battle_model.algo import dgn, bayes_dgn, dgn_r
from examples.battle_model.algo import dgn_tools
from examples.battle_model.scenario import play

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def linear_decay(epoch, x, y):
    min_v = y[0]
    start = x[0]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps

'''
obs_dim, act_dim, h_dim=128,
num_heads=8, gamma=0.95, batch_size=64,
buffer_size=80000, lr=1e-4, neighbors=3,
lamb=0.03, beta=0.01, *args, **kwargs
'''
def spawn_ai(algo_name, env, handle, *args, **kwargs):
    view_space = env.get_view_space(handle)
    assert len(view_space) == 3
    obs_dim = view_space[0]*view_space[1] + 2
    act_dim = env.get_action_space(handle)[0]
    if algo_name == 'dgn':
        model = dgn.DGNAgent(obs_dim=obs_dim, act_dim=act_dim, **kwargs)
    elif algo_name == 'dgn_r':
        model = dgn_r.DGNregAgent(obs_dim=obs_dim, act_dim=act_dim, **kwargs)
    elif algo_name == 'bayes_dgn':
        model = bayes_dgn.BayesDGNAgent(obs_dim=obs_dim, act_dim=act_dim,
                                        **kwargs)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'dgn', 'dgn_r', 'bayes_dgn'},
                        help='choose an algorithm from the preset',
                        required=True)
    parser.add_argument('--save_every', type=int, default=10,
                        help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5,
                        help='decide the udpate interval for q-learning')
    parser.add_argument('--n_round', type=int, default=2000,
                        help='set the trainning round')
    parser.add_argument('--render', action='store_true',
                        help='render or not (if true, render every save)')
    parser.add_argument('--map_size', type=int, default=40,
                        help='set the size of map')
    parser.add_argument('--max_steps', type=int, default=400,
                        help='set the max steps')

    args = parser.parse_args()

    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR,
                                    'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    log_dir = os.path.join(BASE_DIR, 'data/tmp/{}'.format(args.algo))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))
    start_from = 0

    models = [spawn_ai(args.algo, env, handles[0]),
              spawn_ai(args.algo, env, handles[1])]

    if args.render:
        render_every = args.save_every
    else:
        render_every = 0
    runner = dgn_tools.Runner(env, handles, args.map_size, args.max_steps,
                              models, play, render_every=render_every,
                              save_every=args.save_every, tau=0.01,
                              log_name=args.algo, log_dir=log_dir,
                              model_dir=model_dir, train=True)
    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k,
                           [0, int(args.n_round * 0.8), args.n_round],
                           [1, 0.2, 0.1])
        runner.run(eps, k)
