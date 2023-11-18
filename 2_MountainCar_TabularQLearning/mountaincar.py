'''
Solve the MountainCar Env with tabular Q-learnring
Steps for Q-learning:

1) Initialize the Q-Table
2) Get tuple: (state, action, reward, new_state)
3) Perform Bellman-update
4) Repeat 2-3 until kovergence/limit

Problem: Observations i.e. states are continuous and not discrete
Solution: Make them discrete by rounding
'''

import gym
import numpy as np
import pickle
import random

# Hyperparams
LEARNING_RATE = 0.1
NUM_OBS = 20  # discrete number of states to round the observations to
REWARD_GOAL = -200
GAMMA = 0.9  # discount factor
NUM_EPIDOES = 400
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
    obs_bin = np.arange(obs_min[i], obs_max[i], (obs_max[i] - obs_min[i]) / NUM_OBS)
    obs_bins.append(obs_bin)

# now obs_bins has NUM_OBS bins to assign each continuous observation a discrete class

# shape [NUM_OBS, NUM_OBS, n_actions]
shape_q_table = [NUM_OBS for _ in range(n_obs)] + [n_actions]

# build the q table with random values; rewards are -1 in each step so choose the values accordingly
q_table = np.random.uniform(low=-2, high=-1, size=shape_q_table)

best_reward = -float('inf')

# play the defined number of episodes
for played_episodes in range(NUM_EPIDOES):

    env = gym.make('MountainCar-v0')#, render_mode='human')

    start = env.reset()
    obs, _ = start

    done = False
    
    episode_reward = 0
    
    # play the current episode
    while not done:
        
        # greedy or random
        prob = random.random()  # 0-1

        # random
        if prob <= EPSILON:
            

            obs, reward, done, _, _ = env.step(int(np.random.randint(n_actions)))
            episode_reward += reward

        else:
            # greedy
            pass

    
    if episode_reward > best_reward:
        best_reward = episode_reward


    if played_episodes % 50 == 0:
        print(f'Best reward: {best_reward}; {played_episodes} played.')


# finally save the q-table
with open('q_table.pickle', 'wb') as f:
    pickle.dump(q_table, f)