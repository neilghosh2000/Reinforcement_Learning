import gym
from gym import spaces
import random
import numpy as np


class GridWorldEnv(gym.Env):

    def __init__(self):
        self.height = 12
        self.width = 12
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.width), spaces.Discrete(self.height)))
        self.moves = {0: (-1, 0),   # West (Left)
                      1: (0, 1),    # North (Up)
                      2: (1, 0),    # East (Right)
                      3: (0, -1)}   # South (Down)

        self.rewards = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0],
                        [0, 0, 0, -1, -2, -2, -2, -2, -2, -1, 0, 0],
                        [0, 0, 0, -1, -2, -3, -3, -3, -2, -1, 0, 0],
                        [0, 0, 0, -1, -2, -2, -2, -3, -2, -1, 0, 0],
                        [0, 0, 0, -1, -1, -1, -2, -2, -2, -1, 0, 0],
                        [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.target = np.array([[11, 11], [9, 9], [7, 5]])

        self.reset()       # Next episode

    def step(self, action, target):

        if random.uniform(0, 1) < 0.5 and (target != self.target[2]).all():
            self.position = (self.position[0] + 1, self.position[1])        # To account for wind with probability 0.5

        if random.uniform(0, 1) < 0.9 - 0.1/3:          # Picking the desired action with 0.9 probabilty
            x, y = self.moves[action]
        else:
            x, y = self.moves[random.randint(0, 3)]     # Picking the remaining actions with 0.1/3 probability each

        self.position = (self.position[0] + x, self.position[1] + y)

        self.position = (max(0, self.position[0]), max(0, self.position[1]))
        self.position = (min(self.position[0], self.width - 1), min(self.position[1], self.height - 1))     # Ensuring that the state remains in the grid

        if self.position == target:
            return self.position, 10, True, {}

        return self.position, self.rewards[self.position[0], self.position[1]] , False, {}

    def reset(self):

        start = np.array([[0, 6], [0, 5], [0, 1], [0, 0]])        # Bottom left corner is taken as (0, 0)
        self.position = start[random.randint(0, 3)]

        return self.position
