import gym
from gym import spaces
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

class Env(gym.Env):
    def __init__(self, p: float, c_h: float, c_s: float, T: int, W: Callable[[int], int], scaled=False) -> None:
        """
        :param p: procurement cost
        :param c_h: holding cost
        :param c_s: shortage cost
        :param T: time horizon
        :param W: demand function
        """
        super().__init__()

        self.p = p
        self.c_h = c_h
        self.c_s = c_s
        self.T = T
        self.W = W
        self.scaled=scaled

        self.state_high = -100#np.array(self.W.mean + 100*self.W.std)
        self.state_low = 100#min(0 ,self.W.mean - 100*self.W.std)

        # self.action_space = spaces.Discrete(15) # the range of actions is [0,9]
        self.action_space = spaces.Box(low=0, high=15, dtype=np.float32, shape=(1,))
        self.observation_space = spaces.Box(low=np.array([0, int(self.state_high)]), 
                                            high=np.array([1, int(self.state_low)]), 
                                            shape=(2,), 
                                            dtype=np.int64) # the range of the state is unbounded

        self.S = None # state
        self.t = None # time keeeping
        self.dc = 0
        self.data = []


    def scale_state(self, S):
        new_S = S.copy()
        new_S = max(new_S, self.state_low)
        new_S = min(new_S, self.state_high)
        new_S = (new_S - ((self.state_low + self.state_high) / 2)) / (self.state_high - self.state_low)
        if type(new_S) != np.ndarray:
            new_S = np.array([new_S])
        return new_S
    
    def observation(self):
        weekend = int((self.t % 7 == 5) or (self.t % 7 == 6))
        if self.scaled:
            new_S = self.scale_state(self.S)
        else:
            new_S = self.S.copy()
        observation = np.concatenate([new_S, [weekend]])
        return observation
        

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take an action (1 timestep) and return the next state, reward, done and info
        :param action: action
        :return: (state, reward, done, info)
        """
        self.S += int(action)
        demand = self.W(self.t)
        # self.running_mean = (self.running_mean * self.t + demand) / (self.t + 1)
        self.S -= demand
        self.t += 1
        done = (self.t == self.T)
        if done:
            self.dc += 1
        reward = -self.p * action - self.c_h * max(self.S, 0) - self.c_s * max(-self.S, 0)
        # reward += self.p * action
        info = {}
        self.data.append((self.t, self.S, action, reward))
        return self.observation(), float(reward), done, info


    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial state
        :return: initial state
        """
        print("Resetting the environment")
        self.S = np.array([0], dtype=int) # state
        self.t = 0
        return self.observation()
    

    def plot_rewards(self, filename: str) -> None:
        rewards = [x[3] for x in self.data]
        mean = np.mean(rewards)
        std = np.std(rewards)
        plt.clf()
        plt.plot(rewards)
        plt.savefig(filename)
        plt.close()
        return mean, std
    

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from demand_dists import Binomial
    env = Env(p=1, c_h=1, c_s=1, T=5000, W=Binomial(100, 0.1))
    print(np.zeros(1, dtype=int).dtype)
    print(env.observation_space.sample().dtype)
    # It will check your custom environment and output additional warnings if needed
    print(check_env(env))