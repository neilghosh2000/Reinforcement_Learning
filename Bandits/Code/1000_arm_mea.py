# Importing the required libraries
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import random

n_bandit = 200 # No. of bandit problems
n_arms = 1000 # No. of arms in each bandit problem
epsilon = 0.6
delta = 0.4

col = ['r', 'b']


def median_elimination(estimates, count):
    available_actions = set(np.arange(0, n_arms))
    global e, d, time
    epsilon_l = e / 4
    delta_l = d / 2
    rewards = []
    optimal = []

    while len(available_actions) != 1:
        l = int(4 / (epsilon_l * epsilon_l) * math.log(3 / delta_l))        # No. of times each arm has to be sampled in the current step

        if optimal_arm in available_actions:
            optimal.append(1)
        else:
            optimal.append(0)

        for x in available_actions:

            for y in range(0, l):
                reward = np.random.normal(true_means[x], 1)
                action_count[x] += 1
                estimates[x] = estimates[x] + (reward - estimates[x]) / count[x]        # Updating the estimate based on the reward
                rewards.append(reward)

        median = np.median([estimates[x] for x in available_actions])

        removed_actions = []
        for x in available_actions:
            if estimates[x] < median:
                removed_actions.append(x)       # Storing all arms whose estimate is less than the median

        available_actions -= set(removed_actions)      # Updating the available actions for the next round

        epsilon_l = 3 / 4 * epsilon_l       # Updating the value of epsilon for the next round
        delta_l = delta_l / 2               # Updating the value of delta for the next round

    if optimal_arm in available_actions:
        optimal.append(1)
    else:
        optimal.append(0)

    total_rewards.append(rewards)
    n_optimal.append(optimal)


fig1 = plt.figure().add_subplot(111)

avg_rewards = []
per_optimal = []
total_rewards = []
n_optimal = []

for i in range(0, n_bandit):

    true_means = np.random.normal(0, 1, n_arms)     # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
    estimated_means = np.zeros((1, n_arms))[0]      # Initializing the estimated means to 0
    action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

    time = 0
    e = epsilon
    d = delta

    optimal_arm = np.argmax(true_means)     # Finding out the index of optimal arn

    median_elimination(estimated_means, action_count)

df_reward = pd.DataFrame(total_rewards)

avg_rewards.append(df_reward.sum(axis=0)/n_bandit)              # Calculating the average reward over all the bandit problems


fig1.plot(np.arange(0, len(avg_rewards[0])), avg_rewards[0], col[0])         # Plotting the average reward vs steps

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

fig1.plot(np.arange(0, len(avg_rewards[0])), smooth(avg_rewards[0],15), col[1])    # Smoothening the curve

fig1.title.set_text(
    r'Average Reward of the 1000-arm bandit Vs Steps (Using Median Elimination)')  # Setting the title of the plot
fig1.set_xlabel('Steps')  # Setting the label for x-axis
fig1.set_ylabel('Average Reward')  # Setting the label for y-axis
fig1.legend((r"$\epsilon=$" + str(epsilon) + r", $\delta=$" + str(delta),
             "Smoothened Curve"), loc='best')  # Adding the legend





