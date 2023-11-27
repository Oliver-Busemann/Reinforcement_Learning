'''
Solve the BreakOut env with Deep Reinforcement Learning (DQN)
'''

import gym
import cv2
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
from utils_breakout import *
from tqdm import tqdm
import os


NAME = 'BreakOut_Run_0'

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.98
EPISODES = 30_000
EPSILON = 1
EPSILON_DECAY = EPSILON / EPISODES  # decay linearly from 1 to 0 over all episodes
REPLAY_SIZE = 50_000
UPDATE_TARGET = 300  # target net loads weights after X steps
NUM_IMAGES = 4  # use X images/observations to see the movement of the ball
SAVE_EVERY = 500  # save model weights every 500
LOG_EVERY = 50  # log reward to tensorboard and print in console

device = 'cuda'
assert torch.cuda.is_available() == True

# folder to save the weights to every SAVE_EVERY episode
folder_weights = 'Model_Weights'
os.makedirs(folder_weights, exist_ok=True)

env = gym.make("ALE/Breakout-v5")#, render_mode='human')
num_actions = env.action_space.n
replay_butter = Replay_Buffer(batch_size=BATCH_SIZE, replay_size=REPLAY_SIZE)

# get the image shapes for the models
dummy_obs, _ = env.reset()
dummy_img = crop_roi(dummy_obs)
img_height, img_width = dummy_img.shape  # grayscale so only 2 dims

# create the models
model = Net(num_images=NUM_IMAGES, img_width=img_width, img_height=img_height, num_actions=num_actions)
model.train()

target_model = Net(num_images=NUM_IMAGES, img_width=img_width, img_height=img_height, num_actions=num_actions)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# we only optimize the model; the target model only uses its weights
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss()
logger = Logger(folder=NAME, log_every=LOG_EVERY)

model.to(device)
target_model.to(device)

# play the defined number of episodes
for episode in tqdm(range(EPISODES)):

    episode_reward = 0

    # each observation to train on has NUM_IMAGES images
    observations = deque(maxlen=NUM_IMAGES)

    obs, _ = env.reset()
    obs = crop_roi(obs)

    # append the last obs; obs currently is shape (HxW) we need shape (1xHxW)
    observations.append(np.expand_dims(obs, axis=0))

    done = False

    while not done:

        # if the needed number of obs are there, stack them as an array and append to replay memory
        if len(observations) == NUM_IMAGES:
            # this stacks the observations to shape (NUM_IMAGES, H, W)
            obs_add = np.vstack(observations)

            # pick a random or a greedy action; a greedy action can only be done if we have a full observation to predict
            if random.random() <= EPSILON:

                action = int(np.random.randint(num_actions))

            else:  # greedy
                
                # predict the best action with the target network; preprocess the observation for this
                obs_tensor = obs_add / 255.
                obs_tensor = torch.Tensor(obs_tensor).type(torch.float32).unsqueeze(0).to(device)

                # now its shape (1, NUM_IMAGES, H, W) and can be put in the target model
                with torch.no_grad():
                    # the model pred is shape (1, num_actions) so to get the best action we take the argmax of dim 1
                    action = torch.argmax(target_model(obs_tensor), dim=1)

            # if the needed num_observations existed in the previous step, we can create the tuple (s, a, r, s') and add it to the replay buffer

            # take the action and get the new state and reward
            obs, reward, done, _, _ = env.step(action)
            obs = crop_roi(obs)

            # append the new obs to the image queue
            observations.append(np.expand_dims(obs, axis=0))

            # create s' with this
            new_obs_add = np.vstack(observations)

            save_tuple = (obs_add, action, reward, new_obs_add, done)

            replay_butter.add_tuple(save_tuple)

        # if there are not enough observations for a single state (NUM_IMAGES) then also take a random action
        else:
            action = int(np.random.randint(num_actions))
            obs, reward, done, _, _ = env.step(action)
            obs = crop_roi(obs)
            observations.append(np.expand_dims(obs, axis=0))

        # if the replay buffer has the defined length we start fitting the model with a random batch from it in every step
        if len(replay_butter.buffer) == REPLAY_SIZE:
            
            # the batch is a list with the len of the defined batch_size and has the defined tuples (s, a, r, s')
            batch = replay_butter.get_batch()
            states, actions, rewards, new_states, dones = preprocess_batch(batch, bs=BATCH_SIZE, n=NUM_IMAGES, h=img_height, w=img_width, device=device)

            # now we calculate the q-values for the states with the model and the new_states with the target model (here the weights shouldnt be updated)
            q_values = model(states)

            with torch.no_grad():
                new_q_values = target_model(new_states)

            # catculate the second part of the bellman equantion
            target_q_values = rewards + GAMMA * torch.max(new_q_values, dim=1).values * (1 - dones)

            # calculate the loss; we need the predicted q_values from the chosen actions in each instance
            loss = mse_loss(q_values.gather(dim=1, index=actions.unsqueeze(1).type(torch.int64)), target_q_values.unsqueeze(1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        episode_reward += reward

    # add the current episode reward
    logger.add_episode_reward(episode_reward=episode_reward)
    episode_reward = 0

    # update the target model weights with the model weights
    if episode % UPDATE_TARGET == 0:
        target_model.load_state_dict(model.state_dict())

    # log the metrics
    if episode % LOG_EVERY == 0:
        logger.log_metrics(current_episode=episode)

    # save the model weights and put the avg reward in the name since the last save
    if episode % SAVE_EVERY == 0:
        avg_reward = float(np.array(logger.episode_rewards[-SAVE_EVERY:]).mean())
        avg_reward = round(avg_reward, 2)
        torch.save(model.state_dict(), f'{folder_weights}/{NAME}_AvgReward_{avg_reward}.pth')

    # decay epsilon, so get a bit more greedy
    EPSILON -= EPSILON_DECAY

env.close()
