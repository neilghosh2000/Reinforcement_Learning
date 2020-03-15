#!/usr/bin/env python

import click
import numpy as np
import gym
import math
from gym.envs.registration import register
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def env_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=0.5)


def grad(theta, state, action):
    s_1 = include_bias(state)
    mean = theta.dot(s_1)
    s_1 = s_1[np.newaxis]
    gradient = action - mean
    gradient = gradient.reshape(gradient.shape[0], 1)
    x = gradient.dot(s_1)
    return x / (np.linalg.norm(x) + 1e-8)


def include_bias(x):
    return np.insert(x, 0, 1)


@click.command()
@click.argument("env_id", type=str, default="vishamC")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        from rlpa2 import chakra
        env = gym.make('chakra-v0')
        get_action = env_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'vishamC':
        from rlpa2 import vishamC
        env = gym.make('vishamC-v0')
        get_action = env_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' or 'vishamC' ")

    env.seed(42)

    steps = []
    avg_rewards = []
    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    max_iterations = 500
    batch_size = 25
    alpha = 0.0001
    gamma = 0.9

    for i in range(max_iterations):

        total_rew = 0
        step = 0
        theta_grad = np.zeros((action_dim, obs_dim + 1))

        for j in range(batch_size):
            print(i, j)
            g = 1
            curr_state = env.reset()
            done = False
            # Only render the first trajectory
            # Collect a new trajectory
            traj = []
            ret = 0
            ret_avg = 0
            while not done:
                action = get_action(theta, curr_state, rng=rng)
                next_state, rew, done, _ = env.step(action)
                curr_state = next_state

                if abs(curr_state[0]) > 1 or abs(curr_state[1]) > 1:
                    rew = -5
                    traj.append([curr_state, action, rew])
                    curr_state = env.reset()
                else:
                    traj.append([curr_state, action, rew])
                # env.render(mode='human')
                ret += gamma*rew
                g *= gamma
                total_rew += rew
                ret_avg += ret

            g_t = 1
            step += len(traj)
            ret_avg = ret_avg / (len(traj))
            for k in range(len(traj)):
                curr_state, action, rwd = traj[k]
                adv = ret - ret_avg
                theta_grad += alpha * adv * grad(theta, curr_state, action) * g_t

                g_t *= gamma
                ret -= rwd
                ret /= gamma

        step = step / float(batch_size)
        theta_grad = theta_grad / float(batch_size)
        avg_reward = total_rew / float(batch_size)

        theta = np.add(theta, theta_grad)
        avg_rewards.append(avg_reward)
        steps.append(step)

    fig1 = plt.figure().add_subplot(111)
    fig2 = plt.figure().add_subplot(111)

    fig1.plot(np.arange(max_iterations), avg_rewards)
    fig1.set_ylabel("Average Rewards")
    fig1.set_xlabel("Iterations")

    fig2.plot(np.arange(max_iterations), steps)
    fig2.set_ylabel("Average Steps for Convergence")
    fig2.set_xlabel("Iterations")

    plt.show()
    print(theta)

    x_list, y_list = [], []
    z = []
    for i in range(1, 10):
        print(i)
        for j in range(0, 360, 20):
            x = 0.1*i * math.sin(j*math.pi/180)
            y = 0.1*i * math.cos(j*math.pi/180)
            x_list.append(x)
            y_list.append(y)
            env.state = [x, y]

            done = False
            reward = 0
            g = 1
            ret = 0

            while not done:
                action = env_get_action(theta, env.state, rng)
                action = action / (np.linalg.norm(action)) * 0.025
                next_state, rew, done, _ = env.step(action)

                env.state = next_state

                reward += rew
                env.state = next_state

                if abs(env.state[0]) > 1 or abs(env.state[1]) > 1:
                    env.state = [x, y]

                ret += g * rew
            z.append(ret)

    fig3 = plt.axes(projection='3d')
    fig3.plot(x_list, y_list, z)
    fig3.set_xlabel("X")
    fig3.set_ylabel("Y")
    fig3.set_zlabel("Value Function")
    plt.show()

if __name__ == "__main__":
    main()
