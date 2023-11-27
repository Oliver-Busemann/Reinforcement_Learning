import cv2
from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# this function takes as input an observation and returns the grayscale cropped version 
def crop_roi(img, WIDTH_START=7, WIDTH_END=153, HEIGHT_START=31, HEIGHT_END=198):

    img = img[HEIGHT_START: HEIGHT_END, WIDTH_START: WIDTH_END, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


# build the replay buffer
class Replay_Buffer():

    def __init__(self, batch_size=32, replay_size=10_000):
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.buffer = deque(maxlen=self.replay_size)

    def add_tuple(self, tuple):
        self.buffer.append(tuple)

    # sample a random batch from the buffer
    def get_batch(self):
        batch = random.sample(self.buffer, k=self.batch_size)
        return batch
    

# this takes as input a batch (list) of tuples (s, a, r, s', done) and returns them in a format to train on
def preprocess_batch(batch, bs, n, h, w, device):

    # create one array for the states and new states
    states = np.empty((bs, n, h, w))
    new_states = np.empty((bs, n, h, w))
    actions = np.empty((bs))
    rewards = np.empty((bs))
    dones = np.empty((bs))

    for i in range(bs):
        # batch[i] is a tuple; [0] then is the state
        states[i, :, :, :] = batch[i][0]
        new_states[i, :, :, :] = batch[i][3]
        actions[i] = batch[i][1]
        rewards[i] = batch[i][2]
        dones[i] = batch[i][4]

    states /= 255.
    new_states /= 255.

    states = torch.Tensor(states).type(torch.float32)
    new_states = torch.Tensor(new_states).type(torch.float32)
    actions = torch.Tensor(actions).type(torch.float32)
    rewards = torch.Tensor(rewards).type(torch.float32)
    dones = torch.Tensor(dones).type(torch.float32)

    states = states.to(device)
    new_states = new_states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)

    return states, actions, rewards, new_states, dones


class CNN(nn.Module):
    def __init__(self, num_images):
        super().__init__()
        self.num_images = num_images

        self.conv_1 = nn.Conv2d(in_channels=self.num_images, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.relu_1 = nn.ReLU()
        self.maxp_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.relu_2 = nn.ReLU()
        self.maxp_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.bn_3 = nn.BatchNorm2d(num_features=128)
        self.relu_3 = nn.ReLU()
        self.maxp_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.maxp_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.maxp_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.maxp_3(x)
        
        x = self.flatten(x)

        return x

class Net(nn.Module):
    def __init__(self, num_images, img_width, img_height, num_actions):
        super().__init__()
        self.cnn = CNN(num_images=num_images)

        with torch.no_grad():
            self.cnn_out_size = self.cnn(torch.zeros((1, num_images, img_height, img_width))).size()[1]

        self.fc_1 = nn.Linear(self.cnn_out_size, num_actions)

    def forward(self, x):

        x = self.cnn(x)
        x = self.fc_1(x)

        return x


class Logger():
    def __init__(self, folder, log_every):
        self.writer = SummaryWriter(folder)
        self.log_every = log_every
        self.episode_rewards = []

    def add_episode_reward(self, episode_reward):
        self.episode_rewards.append(episode_reward)

    def log_metrics(self, current_episode):

        # log the average reward, the min and the max
        current_rewards = np.array(self.episode_rewards[-self.log_every:])
        avg_reward = current_rewards.mean()
        min_reward = current_rewards.min()
        max_reward = current_rewards.max()

        self.writer.add_scalar('Avg_Reward', avg_reward, current_episode)
        self.writer.add_scalar('Min_Reward', min_reward, current_episode)
        self.writer.add_scalar('Max_Reward', max_reward, current_episode)

        print(f'Episode: {current_episode} - Avg_Reward: {avg_reward}; Min_reward: {min_reward}; Max_reward: {max_reward}')
    