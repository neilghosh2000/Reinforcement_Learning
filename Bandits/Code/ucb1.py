# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import random

n_bandit = 2000         # No. of bandit problems
n_pulls = 1000          # No. of times an arm can be pulled
n_arms = 10             # No. of arms in each bandit problem
c = [0.1, 1, 2]
col = ['r', 'g', 'b', 'y', 'k']

temp = 0.5
e = 0.1


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


def e_greedy(estimates, count, time, eps):
    if eps <= random.uniform(0, 1):  # Greedy action take with 1-e probabilty (Exploit)
        action = np.argmax(estimates)
    else:
        action = random.randint(0, n_arms - 1)  # Non greedy action taken with e probability (Explore)

    if action == optimal_arm:
        n_optimal[time] += 1

    reward = np.random.normal(true_means[action],
                              1)  # Reward sampled from a Normal Distribution corresponding to arm which is picked
    global total_reward
    total_reward[time] += reward
    count[action] += 1
    estimates[action] = estimates[action] + (reward - estimates[action]) / count[
        action]  # Updating the estimate based on the reward


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


fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for c_i in range(0, len(c)):

    total_reward = np.zeros((1, n_pulls))[0]
    n_optimal = np.zeros((1, n_pulls))[0]

    for i in range(0, n_bandit):
        true_means = np.random.normal(0, 1, n_arms)     # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
        estimated_means = np.zeros((1, n_arms))[0]      # Initializing the estimated means to 0
        action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

        optimal_arm = np.argmax(true_means)             # Finding out the index of optimal arn

        for j in range(0, n_pulls):
            ucb1(estimated_means, action_count, j, c[c_i])

    avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems
    per_optimal = np.true_divide(n_optimal, n_bandit / 100)     # Calculating the % optimal action over all the bandit problems

    fig1.plot(range(0, n_pulls), avg_reward, col[c_i])           # Plotting the average reward vs steps
    fig2.plot(range(0, n_pulls), per_optimal, col[c_i])         # Plotting the % optimal action vs steps

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
per_optimal = np.true_divide(n_optimal, n_bandit / 100)      # Calculating the % optimal action over all the bandit problems

fig1.plot(range(0, n_pulls), avg_reward, col[3])            # Plotting the average reward vs steps
fig2.plot(range(0, n_pulls), per_optimal, col[3])           # Plotting the % optimal action vs steps

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
per_optimal = np.true_divide(n_optimal, n_bandit / 100)      # Calculating the % optimal action over all the bandit problems

fig1.plot(range(0, n_pulls), avg_reward, col[4])            # Plotting the average reward vs steps
fig2.plot(range(0, n_pulls), per_optimal, col[4])           # Plotting the % optimal action vs steps


fig1.title.set_text(r'Average Reward of the 10-arm bandit Vs Steps (Using UCB1)')       # Setting the title of the plot
fig1.set_ylabel('Average Reward')   # Setting the label for y-axis
fig1.set_xlabel('Steps')            # Setting the label for x-axis
fig1.legend((r"c=" + str(c[0]), r"c=" + str(c[1]), r"c=" + str(c[2]), r"$\epsilon=$" + str(e) + " ($\epsilon$-Greedy)",
             r"$\tau=$" + str(temp) + " (Softmax)"), loc='best')          # Adding the legend

fig2.title.set_text(r'$\%$ Optimal Action Vs Steps (Using UCB1)')           # Setting the title of the plot
fig2.set_ylabel(r'$\%$ Optimal Action')         # Setting the label for y-axis
fig2.set_xlabel('Steps')                        # Setting the label for x-axis
fig2.set_ylim(0, 100)                           # Setting the limits for the y-axis
fig2.legend((r"c=" + str(c[0]), r"c=" + str(c[1]), r"c=" + str(c[2]), r"$\epsilon=$" + str(e) + " ($\epsilon$-Greedy)",
             r"$\tau=$" + str(temp) + " (Softmax)"), loc='best')