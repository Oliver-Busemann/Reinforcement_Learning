import torch
import numpy as np


# this will take as input all observations & actions and create the final training data
class Data(torch.utils.data.Dataset):
    def __init__(self, observations, actions, total_rewards, best_episodes):
        self.observations = observations
        self.actions = actions
        # must be the same number for input-output pairs
        assert len(self.observations) == len(self.actions)

        # to pick best x episodes
        self.best_episodes = best_episodes

        self.total_rewards = total_rewards

        # to sort the lists in a descending order based on the rewards get the indices for that
        sorted_indices = np.argsort(self.total_rewards)[::-1]  # descending order
        
        # now sort all lists like that and get the best ones for training
        self.total_rewards = [self.total_rewards[i] for i in sorted_indices][:self.best_episodes]
        self.actions = [self.actions[i] for i in sorted_indices][:best_episodes]
        self.observations = [self.observations[i] for i in sorted_indices][:self.best_episodes]

        #print(f'Best {self.best_episodes} Episodes: {np.array(self.total_rewards).mean():.2f}')

        # now the actions & obs are a list of lists that have the obs/actions [[a1.1, a1.2, ...], [a2.1, a2.2, ...], []]
        # add them in one list to create training data
        actions_1dim = [a for sublist in self.actions for a in sublist]
        obs_1dim = [o for sublist in self.observations for o in sublist]
        
        self.train_actions = torch.tensor(actions_1dim).type(torch.float32)
        self.train_obs = torch.tensor(obs_1dim).type(torch.float32)

    def __len__(self):
        return self.train_actions.size()[0]
    
    def __getitem__(self, index):
        x = self.train_obs[index]
        x = torch.tensor(x).type(torch.float32)
        y = self.train_actions[index]
        y = torch.tensor(y).type(torch.long)

        return x, y