import gym
from env import Env
from stable_baselines3 import DDPG
from demand_dists import *
from config import configs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=1)
    args = parser.parse_args()
    config = configs["config" + str(args.config)]

    env = Env(**config)
    T = config["T"]
    model = DDPG("MlpPolicy", env, learning_starts=1, batch_size=64, tau=0.005, gamma=1, train_freq=(1, "step"), verbose=1)#, policy_kwargs={"net_arch": {"pi": [8], "qf": [8]}})
    try:
        model.learn(reset_num_timesteps=False, total_timesteps=T)
    except KeyboardInterrupt as k:
        pass
    mean, std = env.plot_rewards("ddpg.png")
    print(f"Mean: {mean}, Std: {std}")
