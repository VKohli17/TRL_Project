import gym
from env import Env
from stable_baselines3 import PPO
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
    model = PPO("MlpPolicy", env, verbose=2, n_steps=1000, n_epochs=10, batch_size=32, policy_kwargs={"net_arch": [{"pi": [8], "vf": [8]}]})
    try:
        model.learn(reset_num_timesteps=False, total_timesteps=T)
    except KeyboardInterrupt as k:
        pass
    mean, std = env.plot_rewards("ppo.png")
    print(f"Mean: {mean}, Std: {std}")