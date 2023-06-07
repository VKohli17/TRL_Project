import gym
from env import Env
from stable_baselines3 import DDPG
from demand_dists import *

import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
import random
random.seed(0)

ts = 5_000

env = Env(p=1, c_h=1, c_s=1, T=ts, W=Binomial(100, 0.1))
# env = gym.make('MountainCarContinuous-v0')


model = DDPG("MlpPolicy", env, 
             learning_starts=32, 
             batch_size=64, 
             tau=0.01, 
             gamma=1, 
             train_freq=(1, "step"), 
             verbose=1, 
             buffer_size=10000,
             seed=42,
             policy_kwargs={"net_arch": {"pi": [4, 4], "qf": [4, 4]}})
model.gradient_steps = 3
# rc = RewardCallback()
try:
    model.learn(reset_num_timesteps=False, total_timesteps=ts)
except KeyboardInterrupt as k:
    pass
print(env.dc)
env.plot_rewards("ddpg.png")
rewards = [x[3] for x in env.data]
print(np.mean(rewards))
print(np.std(rewards))

