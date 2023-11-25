import cv2
from collections import deque
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl


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

class Net(pl.LightningModule):
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
    
    def training_step(self, batch, batch_idx):
        pass