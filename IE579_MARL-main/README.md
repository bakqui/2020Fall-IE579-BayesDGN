# IE579_MARL (Last updated (grading policy) at Nov 26, 11:25 am)

An tutorial code for the Homework #4 in KAIST IE579 class.

## Code structure
`DGN-R.ipynb`: contains tutorial code for implementing GRAPH CONVOLUTIONAL REINFORCEMENT LEARNING (ICLR, 2020)

paper link: https://arxiv.org/pdf/1810.09202.pdf

`simple_spread.py`: contains a **modified** version of `simple_spread.py` in multiagent-particle-envs package.

original code: https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/scenarios/simple_spread.py


## Requirements
Multi-Agent Particle Environment package (from https://github.com/openai/multiagent-particle-envs)

`pip install git+https://github.com/openai/multiagent-particle-envs`

`pip install gym==0.10.5` # to downgrade gym

## Deliverable (score: 50)

- Code with consistent regularization term (.ipynb or .py)
- Figure with results
   - should include loss, reward of two algorithms (w/, w/o consistent regulization)

## Grading policy
- Just adding KL regulation would be enough.
- It would be best if you achieve good performance, but it is difficult to achieve good results since this environment is originally designed more suitable for continuous action space than discrete action space.
- So I will not include the performance to the score of this homework.

## Contact

ahnkjuree@kaist.ac.kr
