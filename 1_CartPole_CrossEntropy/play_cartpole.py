import gym
import torch
import pytorch_lightning as pl
from model import *
import numpy as np

path_weights = 'CartPole_Model_Weights.pth'

env = gym.make('CartPole-v1', render_mode='human')  # gui like env.render()

# define model and load weights
net = NN()
net.model.load_state_dict(torch.load(path_weights))

done = False

start = env.reset()
obs, _ = start

total_reward = 0

while not done:

    obs_tensor = torch.tensor(obs).type(torch.float32).unsqueeze(0)  # add batchdim      
    action = net(obs_tensor)
    action_prob = torch.softmax(action, dim=1)
    take_action = action_prob.detach().numpy()[0]
    take_action = np.random.choice(len(take_action), p=take_action)

    obs, reward, done, _, _ = env.step(take_action)
    total_reward += reward

env.close()

print(f'Total Episode Reward: {total_reward}')