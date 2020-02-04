# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt

n_bandit = 2000 #No. of bandit problems
n_pulls = 1000 # No. of times an arm can be pulled
n_arms = 10 # No. of arms in each bandit problem
temp = [0.25, 0.5, 0.75, 1]
col = ['r', 'g', 'y', 'b']


def softmax(estimates, count, time, temperature):
    prob = np.exp(np.true_divide(estimates, temperature))  #Finding out the probabilities of eaxh action
    prob = np.true_divide(prob, np.sum(prob))
    action = np.random.choice(a=np.arange(0, n_arms), p=prob) #Picking an action based on the probabilities

    if action == optimal_arm:
        n_optimal[time] += 1

    reward = np.random.normal(true_means[action], 1)
    global total_reward
    total_reward[time] += reward
    count[action] += 1
    estimates[action] = estimates[action] + (reward - estimates[action]) / count[action]        # Updating the estimate based on the reward


fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

for t in range(0, len(temp)):

    total_reward = np.zeros((1, n_pulls))[0]
    n_optimal = np.zeros((1, n_pulls))[0]

    for i in range(0, n_bandit):
        true_means = np.random.normal(0, 1, n_arms)         # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
        estimated_means = np.zeros((1, n_arms))[0]          # Initializing the estimated means to 0
        action_count = np.zeros((1, n_arms))[0]         # No. of times an action has been selected

        optimal_arm = np.argmax(true_means)         # Finding out the index of optimal arn

        for j in range(0, n_pulls):
            softmax(estimated_means, action_count, j, temp[t])

    avg_reward = np.true_divide(total_reward, n_bandit)         # Calculating the average reward over all the bandit problems
    per_optimal = np.true_divide(n_optimal, n_bandit / 100)     # Calculating the % optimal action over all the bandit problems

    fig1.plot(range(0, n_pulls), avg_reward, col[t])            # Plotting the average reward vs steps
    fig2.plot(range(0, n_pulls), per_optimal, col[t])           # Plotting the % optimal action vs steps


    fig1.title.set_text(r'Average Reward of the 10-arm bandit Vs Steps (Using Softmax)')      # Setting the title of the plot
    fig1.set_xlabel('Steps')                # Setting the label for x-axis
    fig1.set_ylabel('Average Reward')       # Setting the label for y-axis
    fig1.legend((r"$\tau=$" + str(temp[0]), r"$\tau=$" + str(temp[1]), r"$\tau=$" + str(temp[2]),
                 r"$\tau=$" + str(temp[3])), loc='best')         # Adding the legend


    fig2.title.set_text(r'$\%$ Optimal Action Vs Steps (Using Softmax)')          # Setting the title of the plot
    fig2.set_xlabel('Steps')                    # Setting the label for x-axis
    fig2.set_ylabel(r'$\%$ Optimal Action')     # Setting the label for y-axis
    fig2.set_ylim(0, 100)                       # Setting the limits for the y-axis
    fig2.legend((r"$\tau=$" + str(temp[0]), r"$\tau=$" + str(temp[1]), r"$\tau=$" + str(temp[2]),
                 r"$\tau=$" + str(temp[3])), loc='best')         # Adding the legend


