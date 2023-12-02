from utils_breakout import *
import gym
import torch
import torch.nn as nn
from collections import deque


NUM_IMAGES = 4

path_weights = '/home/olli/Projects/Exercises/Reinforcement_Learning/3_BreakOut_DQN/Model_Weights_BreakOut_Run_2/BreakOut_Run_2_AvgReward_6.86.pth'

device = 'cpu' # 'cuda'
#assert torch.cuda.is_available() == True

observations = deque(maxlen=NUM_IMAGES)

# build the model and load the best weights
env = gym.make('ALE/Breakout-v5', render_mode='human')
num_actions = env.action_space.n
obs, _ = env.reset()
obs = crop_roi(obs)
observations.append(np.expand_dims(obs, axis=0))
img_height, img_width = obs.shape
model = Net(num_images=NUM_IMAGES, img_width=img_width, img_height=img_height, num_actions=num_actions)
model.eval()
model.load_state_dict(torch.load(path_weights))
model.to(device)

done = False

episode_reward = 0

lifes = 5

while not done:

    if len(observations) < NUM_IMAGES:

        action = 1  # int(np.random.randint(num_actions))

    else:

        obs_full = np.vstack(observations)
        obs_full = torch.Tensor(obs_full).type(torch.float32).unsqueeze(0).to(device)
        obs_full /= 255.

        with torch.no_grad():
            q_values = model(obs_full)
            action = int(torch.argmax(q_values, dim=1))

    obs, reward, done, _, info = env.step(action)
    
    if lifes != info['lives']:
        lifes = info['lives']  # change life score

        # empty queue so that action 1 is performed for starting the next life!
        observations = deque(maxlen=NUM_IMAGES)

    obs = crop_roi(obs)
    observations.append(np.expand_dims(obs, axis=0))

    episode_reward += reward

env.close()

print(f'Final_Reward: {episode_reward}')

        