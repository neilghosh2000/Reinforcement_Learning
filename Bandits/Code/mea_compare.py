# Importing the required libraries
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import random

n_bandit = 2000 # No. of bandit problems
n_arms = 10 # No. of arms in each bandit problem
n_pulls = 1000  # No. of arms in each bandit problem
epsilon = 0.6
delta = 0.4
c = 1
e = 0.1
temp = 0.5
col = ['r', 'g', 'b', 'k']


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


def softmax(estimates, count, time, temperature):
    prob = np.exp(np.true_divide(estimates, temperature))  # Finding out the probabilities of eaxh action
    prob = np.true_divide(prob, np.sum(prob))
    action = np.random.choice(a=np.arange(0, n_arms), p=prob)  # Picking an action based on the probabilities

    if action == optimal_arm:
        n_optimal[time] += 1

    reward = np.random.normal(true_means[action], 1)
    global total_reward
    total_reward[time] += reward
    count[action] += 1
    estimates[action] = estimates[action] + (reward - estimates[action]) / count[
        action]  # Updating the estimate based on the reward


def e_greedy(estimate, count, time, eps):
    if eps <= random.uniform(0, 1):             # Greedy action take with 1-e probabilty (Exploit)
        action = np.argmax(estimate)
    else:
        action = random.randint(0, n_arms - 1)           # Non greedy action taken with e probability (Explore)

    if action == optimal_arm:
        n_optimal[time] += 1

    reward = np.random.normal(true_means[action], 1)            # Reward sampled from a Normal Distribution corresponding to arm which is picked
    global total_reward
    total_reward[time] += reward
    count[action] += 1
    estimate[action] = estimate[action] + (reward - estimate[action]) / count[action]       # Updating the estimate based on the reward


def ucb1(estimates, count, time, conf):
    upper_bound = np.zeros((1, n_arms))[0]
    for a in range(0, n_arms):
        upper_bound[a] = estimates[a] + conf*math.sqrt(2*math.log(time + 1)/(count[a]+1))       #

    action = np.argmax(upper_bound)     #Picking the action with max. upper bound

    if action == optimal_arm:
      n_optimal[time] += 1

    reward = np.random.normal(true_means[action], 1)
    global total_reward
    total_reward[time] += reward
    count[action] += 1
    estimates[action] = estimates[action] + (reward - estimates[action]) / count[action]        # Updating the estimate based on the reward


fig1 = plt.figure().add_subplot(111)

total_reward = np.zeros((1, n_pulls))[0]
n_optimal = np.zeros((1, n_pulls))[0]

for i in range(0, n_bandit):
    true_means = np.random.normal(0, 1, n_arms)     # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
    estimated_means = np.zeros((1, n_arms))[0]      # Initializing the estimated means to 0
    action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

    optimal_arm = np.argmax(true_means)             # Finding out the index of optimal arn

    for j in range(0, n_pulls):
        e_greedy(estimated_means, action_count, j, e)

avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems


fig1.plot(range(0, n_pulls), avg_reward, col[2])            # Plotting the average reward vs steps



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



total_reward = np.zeros((1, n_pulls))[0]
n_optimal = np.zeros((1, n_pulls))[0]

for i in range(0, n_bandit):
    true_means = np.random.normal(0, 1, n_arms)     # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
    estimated_means = np.zeros((1, n_arms))[0]      # Initializing the estimated means to 0
    action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

    optimal_arm = np.argmax(true_means)             # Finding out the index of optimal arn

    for j in range(0, n_pulls):
        softmax(estimated_means, action_count, j, temp)

avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems

fig1.plot(range(0, n_pulls), avg_reward, col[3])            # Plotting the average reward vs steps



total_reward = np.zeros((1, n_pulls))[0]
n_optimal = np.zeros((1, n_pulls))[0]

for i in range(0, n_bandit):
    true_means = np.random.normal(0, 1, n_arms)     # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
    estimated_means = np.zeros((1, n_arms))[0]      # Initializing the estimated means to 0
    action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

    optimal_arm = np.argmax(true_means)             # Finding out the index of optimal arn

    for j in range(0, n_pulls):
        ucb1(estimated_means, action_count, j, c)

avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems


fig1.plot(range(0, n_pulls), avg_reward, col[1])           # Plotting the average reward vs steps


fig1.title.set_text(
    r'Average Reward of the 10-arm bandit (MEA vs UCB1 vs Softmax vs $\epsilon$-Greedy)')  # Setting the title of the plot
fig1.set_xlabel('Steps')  # Setting the label for x-axis
fig1.set_ylabel('Average Reward')  # Setting the label for y-axis
fig1.legend(("MEA "+ r"($\epsilon=$" + str(epsilon) + r", $\delta=$" + str(delta) + ")",
             r"$\epsilon$-Greedy " + r"($\epsilon=$" + str(e) + ")", "Softmax " + r"($\tau=$" + str(temp) + ")", "UCB1 (c=" + str(c) + ")"), loc='best')  # Adding the legend


