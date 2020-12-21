import argparse
import os
import magent

from examples.battle_model.algo import dgn, bayes_dgn
from examples.battle_model.algo import dgn_tools
from examples.battle_model.scenario import battle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def spawn_ai(algo_name, env, handle, *args, **kwargs):
    view_space = env.get_view_space(handle)
    assert len(view_space) == 3
    obs_dim = view_space[0]*view_space[1] + 2
    act_dim = env.get_action_space(handle)[0]
    if algo_name == 'dgn':
        model = dgn.DGNAgent(obs_dim=obs_dim, act_dim=act_dim, **kwargs)
    elif algo_name == 'bayes_dgn':
        model = bayes_dgn.BayesDGNAgent(obs_dim=obs_dim, act_dim=act_dim,
                                        **kwargs)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'dgn', 'bayes_dgn'},
                        help='choose an algorithm from the preset',
                        required=True)
    parser.add_argument('--oppo', type=str,
                        choices={'dgn', 'bayes_dgn', 'ac',
                                 'mfac', 'mfq', 'il'},
                        help='indicate the opponent model', required=True)
    parser.add_argument('--n_round', type=int, default=50,
                        help='set the battle round')
    parser.add_argument('--render', action='store_true',
                        help='render or not (if true, render every save)')
    parser.add_argument('--map_size', type=int, default=40,
                        help='set the size of map')
    parser.add_argument('--max_steps', type=int, default=400,
                        help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR,
                                    'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    main_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.algo))
    oppo_model_dir = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.oppo))

    models = [spawn_ai(args.algo, env, handles[0]),
              spawn_ai(args.oppo, env, handles[1])]

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir, step=args.idx[1])

    runner = dgn_tools.Runner(env, handles, args.map_size, args.max_steps,
                              models, battle, render_every=0)

    win_cnt = {'main': 0, 'oppo': 0}

    for k in range(0, args.n_round):
        runner.run(0.0, k, win_cnt=win_cnt)

    main_rate = win_cnt['main']/args.n_round
    oppo_rate = win_cnt['oppo']/args.n_round
    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3}'.format(args.algo,
                                                             main_rate,
                                                             args.oppo,
                                                             oppo_rate))
