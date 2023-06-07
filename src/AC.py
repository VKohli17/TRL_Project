from env import Env
from demand_dists import *

def policy(state):
    threshold = 10
    act = max(0, threshold - state)
    act = min(act, 15)
    return act

env = Env(p=1, c_h=1, c_s=1, T=5000, W=Binomial(100, 0.1))
state = env.reset()
done = False
rewards = []
while not done:
    action = policy(state)
    next_state, reward, done = env.step(action)
    rewards.append(reward)
    state = next_state


import matplotlib.pyplot as plt
plt.plot(rewards[2000:])
plt.savefig('rewards_optimal.png')

