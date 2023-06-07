from env import Env
from demand_dists import *
from config import configs
import argparse

def policy(state):
    threshold = env.W.mean
    act = max(0, threshold - state[0])
    return act


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=1)
    args = parser.parse_args()
    config = configs["config" + str(args.config)]

    env = Env(**config)
    state = env.reset()
    done = False
    rewards = []
    while not done:
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        state = next_state
    mean, std = env.plot_rewards("optimal.png")
    print(f"Mean: {mean}, Std: {std}")
