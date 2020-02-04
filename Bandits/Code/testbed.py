# Importing the required libraries
import numpy as np

n_bandits = 2000    # No. of bandit problems
n_arms = 10         # No. of arms in each bandit problem


def initialize(n_bandits, n_arms):

    true_mean = list()
    action_count = list()

    for i in range(0, n_bandits):
        curr_means = np.random.normal(0, 1, n_arms)  # Sampling the true means from a Normal Distribution of mean = 0 and variance = 1
        curr_count = np.zeros((1, n_arms))[0]  # No. of times an action has been selected

    true_mean.append(curr_means)
    action_count.append(curr_count)

    return true_mean, action_count      # Returning the list of true means and times an action has been picked for all the bandits

