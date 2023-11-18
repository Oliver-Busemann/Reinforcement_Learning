'''Solve the cartpole environment with the crossentropy methode

Steps:
  1) Run num_episodes with current model
  2) For each episodes save the sequences of observations, rewards, actions
  3) Choose the best X% episodes for training data and train the model
  4) Repeat until convergence
'''

from model import *
from data import *
import gym
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")  # deprecationwarning...
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

goal = 475  # reward to solve the env
current_best = 0
satisfied_episodes = 0  # achieve goal multiple epochs in a row

num_episodes = 50
batch_size = 32
epochs = 1
percentile = 0.75  # pick the best 25% of the episodes
best_episodes = int(num_episodes * (1 - percentile))  # for slicing after sorting lists

#env = gym.make('CartPole-v1', render_mode='human')  # gui like env.render()
env = gym.make('CartPole-v1')

# define the model
net = NN()

# create episodes and train the model until goal is satisfied
while current_best < goal:

    observations = []
    actions = []
    total_rewards = []

    # play the defined number of episodes
    for episode in range(num_episodes):
        
        # values to save in each episode
        obs_episode = []
        actions_episode = []
        reward_episode = 0

        start = env.reset()
        obs, _ = start  # cant unpack directly....

        done = False

        while not done:
            #random_action = np.random.randint(0, 2)
            # take an action based on the model prediction from the current observation 
            obs_tensor = torch.tensor(obs).type(torch.float32).unsqueeze(0)  # add batchdim
            
            action = net(obs_tensor)
            
            # action is currently only the logits
            action_prob = torch.softmax(action, dim=1)
            
            take_action = action_prob.detach().numpy()[0]
            

            take_action = np.random.choice(len(take_action), p=take_action)
            #take_action = int(torch.argmax(action_prob, dim=1))
            
            # append the values to the lists
            obs_episode.append(obs)
            actions_episode.append(take_action)

            obs, reward, done, _, _ = env.step(take_action)
            reward_episode += reward

            #env.render()

            if done:
                break

        # append the values from the done episode to the lists
        observations.append(obs_episode)
        actions.append(actions_episode)
        total_rewards.append(reward_episode)

    # after all episodes are finished assign the mean reward
    current_mean = np.array(total_rewards).mean()
    if current_mean > current_best:
        current_best = current_mean


    # show progress
    print(f'MEAN: {current_mean:.2f}')
    print(f'BEST: {current_best:.2f}\n')
    

    # create training dataset and loader from the data
    ds = Data(observations, actions, total_rewards, best_episodes)

    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    trainer = pl.Trainer(accelerator='cpu', max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model=net, train_dataloaders=dl)

print('SOLVED!!!')

# save weights to show performance later
torch.save(net.model.state_dict(), 'CartPole_Model_Weights.pth')

env.close()