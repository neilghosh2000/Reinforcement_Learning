import gym
import gridworlds
from Q_learning import QLearning
from SARSA import SARSA
from SARSA_lambda import SARSA_Lambda
import numpy as np
import matplotlib.pyplot as plt

env = gym.envs.make('GridWorld-v0')

n_runs = 50
n_episodes = 2500
gamma = 0.9
alpha = 0.15
epsilon = 0.1

targets = {'A': (11, 11), 'B': (9, 9), 'C': (7, 5)}     # Bottom left corner is taken as (0,0)
lambda_val = [0, 0.3, 0.5, 0.9, 0.99, 1]

def results(algo, episodes, runs):

    x = np.arange(episodes)
    avg_steps = np.zeros(episodes)
    avg_rewards = np.zeros(episodes)

    for i in range(runs):
        rewards, steps = algo.run()
        avg_steps += steps
        avg_rewards += rewards

        print(i, rewards)
        print(steps)

    avg_steps = np.divide(avg_steps, runs)
    avg_rewards = np.divide(avg_rewards, runs)

    fig1 = plt.figure().add_subplot(111)
    fig2 = plt.figure().add_subplot(111)

    fig1.plot(x, avg_rewards)
    # fig1.title.set_text(algo.__class__.__name__ )
    fig1.set_xlabel('Number of Episodes')  # Setting the label for x-axis
    fig1.set_ylabel('Average Reward per Episode')  # Setting the label for y-axis

    fig2.plot(x, avg_steps)
    # fig2.title.set_text(algo.__class__.__name__)
    fig2.set_xlabel('Number of Episodes')  # Setting the label for x-axis
    fig2.set_ylabel('Average Steps to reach the goal')  # Setting the label for y-axis

    plt.show()

    algo.show_policy()


def results_sarsa_lambda(algo, episodes, runs):

    x = np.array(lambda_val)
    avg_steps = np.zeros((len(lambda_val), episodes))
    avg_rewards = np.zeros((len(lambda_val), episodes))

    fig1 = plt.figure().add_subplot(111)
    fig2 = plt.figure().add_subplot(111)

    y1 = []
    y2 = []

    for l in range(len(lambda_val)):

        for i in range(runs):
            rewards, steps = algo.run()
            avg_steps[l] += steps
            avg_rewards[l] += rewards

            print(i, rewards)
            print(steps)

        avg_steps[l] = np.divide(avg_steps[l], runs)
        avg_rewards[l] = np.divide(avg_rewards[l], runs)

        y1.append(avg_rewards[l][-1])
        y2.append(avg_steps[l][-1])

    fig1.plot(x, y1)
    fig2.plot(x, y2)

    # fig1.title.set_text(algo.__class__.__name__ )
    fig1.set_xlabel(r'$\lambda$')  # Setting the label for x-axis
    fig1.set_ylabel('Average Reward per Episode')  # Setting the label for y-axis


    # fig2.title.set_text(algo.__class__.__name__)
    fig2.set_xlabel(r'$\lambda$')  # Setting the label for x-axis
    fig2.set_ylabel('Average Steps to reach the goal')  # Setting the label for y-axis

    plt.show()


results(SARSA(env, n_episodes, targets['B'], gamma, alpha), n_episodes, n_runs)
# results(QLearning(env, n_episodes, targets['C'], gamma, alpha, epsilon), n_episodes, n_runs)
# results(SARSA_Lambda(env, n_episodes, targets['A'], lambda_val[0], gamma, alpha), n_episodes, n_runs)
# results_sarsa_lambda(SARSA_Lambda(env, 25, targets['C'], gamma, alpha), 25, n_runs)

