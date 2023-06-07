from demand_dists import *
from env import Env

env = Env(1, 1, 1, 10, W=Gaussian(10, 1))
state = env.reset()

done = False
while not done:
    state, reward, done = env.step(1)
    print(state, reward, done)