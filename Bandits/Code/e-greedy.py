# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import random

n_bandit = 2000         # No. of bandit problems
n_pulls = 1000          # No. of times an arm can be pulled
n_arms = 10             # No. of arms in each bandit problem
epsilon = [0, 0.01, 0.05, 0.1]
col = ['r', 'g', 'y', 'b']


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


fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for e in range(0, len(epsilon)):

    total_reward = np.zeros((1, n_pulls))[0]
    n_optimal = np.zeros((1, n_pulls))[0]

    for i in range(0, n_bandit):

        true_means = np.random.normal(0, 1, n_arms)         # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
        estimated_means = np.zeros((1, n_arms))[0]                    # Initializing the estimated means to 0
        action_count = np.zeros((1, n_arms))[0]                   # No. of times an action has been selected

        optimal_arm = np.argmax(true_means)         # Finding out the index of optimal arn

        for j in range(0, n_pulls):
            e_greedy(estimated_means, action_count, j, epsilon[e])

    avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems
    per_optimal = np.true_divide(n_optimal, n_bandit / 100)      # Calculating the % optimal action over all the bandit problems

    fig1.plot(range(0, n_pulls), avg_reward, col[e])            # Plotting the average reward vs steps
    fig2.plot(range(0, n_pulls), per_optimal, col[e])           # Plotting the % optimal action vs steps

    fig1.title.set_text(r'Average Reward of the 10-arm bandit Vs Steps (Using $\epsilon$-Greedy)')      # Setting the title of the plot
    fig1.set_xlabel('Steps')                # Setting the label for x-axis
    fig1.set_ylabel('Average Reward')       # Setting the label for y-axis
    fig1.legend((r"$\epsilon=$" + str(epsilon[0]), r"$\epsilon=$" + str(epsilon[1]), r"$\epsilon=$" + str(epsilon[2]),
                 r"$\epsilon=$" + str(epsilon[3])), loc='best')         # Adding the legend


    fig2.title.set_text(r'$\%$ Optimal Action Vs Steps (Using $\epsilon$-Greedy)')          # Setting the title of the plot
    fig2.set_xlabel('Steps')                    # Setting the label for x-axis
    fig2.set_ylabel(r'$\%$ Optimal Action')     # Setting the label for y-axis
    fig2.set_ylim(0, 100)                       # Setting the limits for the y-axis
    fig2.legend((r"$\epsilon=$" + str(epsilon[0]), r"$\epsilon=$" + str(epsilon[1]), r"$\epsilon=$" + str(epsilon[2]),
                 r"$\epsilon=$" + str(epsilon[3])), loc='best')         # Adding the legend


