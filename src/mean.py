from env import Env
from demand_dists import *
from config import configs
import argparse


def policy(state, threshold):
    act = max(0, int(threshold - state))
    act = min(act, 15)
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
    W_sum = 0
    W_count = 0

    while not done:
        if W_count > 0:
            threshold = W_sum / W_count
        else:
            threshold = 10
        action = policy(state, threshold)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        W_sum += (state + action - next_state)
        W_count += 1
        state = next_state
    mean, std = env.plot_rewards("mean.png")
    print(f"Mean: {mean}, Std: {std}")
