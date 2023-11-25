'''
Solve the BreakOut env with Deep Reinforcement Learning (DQN)
Steps:
1) Find the correct shape to crop out (relevent field)
2) Write a function that crops the obs and makes it grayscale (reduces compute needed by 3)
'''

import gym
import cv2
import numpy as np
from collections import deque
import random
import torch
print(torch.cuda.is_available())
import torch.nn as nn
import pytorch_lightning as pl
from utils_breakout import *
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.98
EPISODES = 1 # 30_000
EPSILON = 1
EPSILON_DECAY = EPSILON / EPISODES  # decay linearly
REPLAY_SIZE = 10_000
UPDATE_TARGET = 2_000  # target net loads weights after X steps
NUM_IMAGES = 4  # use X images/observations to see the movement of the ball




env = gym.make("ALE/Breakout-v5", render_mode='human')
num_actions = env.action_space.n
replay_butter = Replay_Buffer(batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE)

# get the image shapes for the models
dummy_obs, _ = env.reset()
dummy_img = crop_roi(dummy_obs)
img_height, img_width = dummy_img.shape

# create the models
model = Net(num_images=NUM_IMAGES, img_width=img_width, img_height=img_height, num_actions=num_actions)
target_model = Net(num_images=NUM_IMAGES, img_width=img_width, img_height=img_height, num_actions=num_actions)
target_model.load_state_dict(model.state_dict())

for episode in range(EPISODES):

    obs, _ = env.reset()
    obs = crop_roi(obs)

    done = False

    while not done:
        action = int(np.random.randint(num_actions))
        obs, reward, done, _, _ = env.step(action)
        obs = crop_roi(obs)

env.close()