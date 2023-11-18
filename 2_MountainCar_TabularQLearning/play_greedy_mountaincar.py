# show the learned policy

import gym
import pickle
import numpy as np
from copy import deepcopy

path_q_table = 'q_table.pickle'

with open(path_q_table, 'rb') as f:
    q_table = pickle.load(f)

NUM_OBS = q_table.shape[0]

env = gym.make('MountainCar-v0', render_mode='human')

start = env.reset()

obs, _ = start

done = False

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

total_reward = 0
episodes = 0

while not done:

    # make the observation discrete
    obs_cat = []

    for bins, o in zip(obs_bins, obs):
        
        # subtract the obs from the bins and make each number positive to get the closest bin using argmin
        bin = deepcopy(bins) - o
        bin = [abs(b) for b in bin]
        bin = np.argmin(bin)  # this is the current obs rounded
        
        obs_cat.append(bin)

    best_action = np.argmax(q_table[obs_cat[0], obs_cat[1], :])

    obs, reward, done, _, _ = env.step(best_action)

    total_reward += reward
    episodes += 1

print(f'Finished after {episodes} with {total_reward} reward')

