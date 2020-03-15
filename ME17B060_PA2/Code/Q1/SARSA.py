import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class SARSA:

    def __init__(self, env, episodes, target, gamma=0.9, alpha=0.1):
        self.Q = np.zeros((env.height, env.width, env.action_space.n))
        self.total_Q = np.zeros((env.height, env.width, env.action_space.n))
        self.gamma = gamma
        self.alpha = alpha
        self.env = env
        self.target = target
        self.episodes = episodes
        self.env.reset()

    def action(self, curr_state):

        prob = np.exp(self.Q[curr_state[0], curr_state[1], :])  # Finding out the probabilities of each action
        prob = np.true_divide(prob, np.sum(prob))
        action = np.random.choice(a=np.arange(0, self.env.action_space.n), p=prob)  # Picking an action based on the probabilities

        return action

    def run(self):

        steps = []
        rewards = []
        self.Q = np.zeros((self.env.height, self.env.width, self.env.action_space.n))

        for i in range(self.episodes):

            curr_state = self.env.reset()
            total_rewards = 0
            j = 0
            while True:
                j += 1

                next_action = self.action(curr_state)
                next_state, reward, terminal, k = self.env.step(next_action, self.target)

                # TD Update
                target = reward + self.gamma * self.Q[next_state[0], next_state[1], self.action(next_state)]
                td_error = target - self.Q[curr_state[0], curr_state[1], next_action]
                self.Q[curr_state[0], curr_state[1], next_action] += self.alpha * td_error

                total_rewards += reward

                if terminal:
                    steps.append(j)
                    rewards.append(total_rewards)
                    break

                curr_state = next_state

        self.total_Q += self.Q

        return rewards, steps

    def show_policy(self):

        grid = self.env.rewards
        grid[self.target[0]][self.target[1]] = 1
        grid[0][6] = 2
        grid[0][5] = 2
        grid[0][1] = 2
        grid[0][0] = 2

        plt.figure(figsize=(12, 12))
        cmap = colors.ListedColormap(
            [(0, 0, 0), (0.3, 0.3, 0.3), (0.6, 0.6, 0.6), (0.9, 0.9, 0.9), (1, 0, 0), (0, 0, 0.8)])

        plt.pcolor([*zip(*grid)], cmap=cmap, edgecolors='k', linewidths=5)

        ax = plt.axes()
        for i in range(self.env.width):
            for j in range(self.env.height):

                max_q = max(self.total_Q[i, j, :])
                policy = (np.random.choice([index for index, val in enumerate(self.total_Q[i, j, :]) if val == max_q]))

                if policy == 0:
                    ax.arrow(i + 0.8, j + 0.5, -0.4, 0, head_width=0.25, head_length=0.25, fc='g', ec='g')
                elif policy == 1:
                    ax.arrow(i + 0.5, j + 0.2, 0, 0.4, head_width=0.25, head_length=0.25, fc='g', ec='g')
                elif policy == 2:
                    ax.arrow(i + 0.2, j + 0.5, 0.4, 0, head_width=0.25, head_length=0.25, fc='g', ec='g')
                else:
                    ax.arrow(i + 0.5, j + 0.8, 0, -0.4, head_width=0.25, head_length=0.25, fc='g', ec='g')

        plt.show()





