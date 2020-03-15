from gym.envs.registration import register

register(
    id='Gridworld-v0',
    entry_point='Control_Algorithms.gridworlds.envs:GridWorld',
)