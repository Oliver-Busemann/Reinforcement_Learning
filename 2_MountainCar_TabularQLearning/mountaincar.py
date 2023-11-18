'''
Solve the MountainCar Env with tabular Q-learnring
Steps for Q-learning:

1) Initialize the Q-Table
2) Get tuple: (state, action, reward, new_state)
3) Perform Bellman-update
4) Repeat 2-3 until kovergence/limit

Problem: Observations i.e. states are continuous and not discrete
Solution: Make them discrete by creating bins
'''

import gym
import numpy as np
import pickle
import random
from copy import deepcopy

# Hyperparams
LEARNING_RATE = 0.1
NUM_OBS = 20  # discrete number of states to round the observations to
GAMMA = 0.9  # discount factor
NUM_EPIDOES = 500
EPSILON = 1  # exploration (1) vs exploitation (0)


env = gym.make('MountainCar-v0')#, render_mode='human')

# get the actions space
n_actions = env.action_space.n

# we need to round the obs space to the defined number of values
obs_min = env.observation_space.low
obs_max = env.observation_space.high
n_obs = len(obs_min)

# create concrete bins for the observations
obs_bins = []

# do this for all observations
for i in range(n_obs):
    obs_bin = np.arange(obs_min[i], obs_max[i] + 0.0001, (obs_max[i] - obs_min[i]) / NUM_OBS)
    obs_bins.append(obs_bin)

# now obs_bins has NUM_OBS bins to assign each continuous observation a discrete class

# shape [NUM_OBS, NUM_OBS, n_actions]
shape_q_table = [NUM_OBS for _ in range(n_obs)] + [n_actions]

# build the q table with random values; rewards are -1 in each step so choose the values accordingly
q_table = np.random.uniform(low=-2, high=-1, size=shape_q_table)

best_reward = -float('inf')

env = gym.make('MountainCar-v0')

# play the defined number of episodes
for played_episodes in range(NUM_EPIDOES):

    start = env.reset()
    obs, _ = start

    done = False
    
    episode_reward = 0
    
    # play the current episode
    while not done:
        
        # greedy or random
        prob = random.random()  # 0-1

        # make the observation discrete
        obs_cat = []
        
        for bins, o in zip(obs_bins, obs):
            
            # subtract the obs from the bins and make each number positive to get the closest bin using argmin
            bin = deepcopy(bins) - o
            bin = [abs(b) for b in bin]
            bin = np.argmin(bin)  # this is the current obs rounded
            
            obs_cat.append(bin)
        
        # EPSILON == Prob. of picking a random action
        if prob <= EPSILON:

            action = int(np.random.randint(n_actions))

        # (1 - EPSILON) == Prob. of picking a greedy action
        else:

            # ToDo make it dynamically somehow
            action = np.argmax(q_table[obs_cat[0], obs_cat[1], :])
        
        # take the chose action
        obs_new, reward, done, _, _ = env.step(action)

        # make the new obs discrete too for getting the best Q-Value for it
        obs_next_cat = []
        for bins, o in zip(obs_bins, obs_new):
                
            # subtract the obs from the bins and make each number positive to get the closest bin using argmin
            bin = deepcopy(bins) - o
            bin = [abs(b) for b in bin]
            bin = np.argmin(bin)  # this is the current obs rounded
            
            obs_next_cat.append(bin)

        Q_next_max = max(q_table[obs_next_cat[0], obs_next_cat[1], :])

        # perform bellman-update, i.e. current q value
        q_table[obs_cat[0], obs_cat[1], action] = (1 - LEARNING_RATE) * q_table[obs_cat[0], obs_cat[1], action] + LEARNING_RATE * (reward + GAMMA * Q_next_max)

        episode_reward += reward

        # next step
        obs = obs_new
        
    # after the epoch lower EPSILON so that after the defined number of episodes it becomes 0 (i.e. only greedy actions)
    EPSILON -= (1 / NUM_EPIDOES)
    print(played_episodes)
    if episode_reward > best_reward:
        best_reward = episode_reward


    if played_episodes % 5 == 0:
        print(f'Best reward: {best_reward}; {played_episodes} played.')


# finally save the q-table
with open('q_table.pickle', 'wb') as f:
    pickle.dump(q_table, f)