from env import Env
from demand_dists import *
from config import configs
import argparse
import torch
import torch.nn as nn
torch.manual_seed(42)

class CriticNN(nn.Module):
    def __init__(self, c_s, c_h) -> None:
        super().__init__()
        # a,b,c,d
        self.weights = nn.Parameter(data=torch.tensor([5, 0, 1, 1], dtype=torch.float32))

        # cost params
        self.c_s = c_s
        self.c_h = c_h


    # def forward(self, state):
    #     if state < self.weights[0]:
    #         return self.weights[1] - (state - self.weights[0]) / (self.weights[3] * self.c_s)
    #     else:
    #         return self.weights[1] + (state - self.weights[0]) / (self.weights[2] * self.c_h)

    def forward(self, state):
        return -1 * (torch.sqrt((state - self.weights[0])**2 + 0.01) - 0.1)

    def actor(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            if state < self.weights[0]:
                return (self.weights[0] - state)
            else: 
                return torch.tensor(0.0)

class CriticFA:
    def __init__(self, c_s, c_h, gamma) -> None:
        self.gamma = gamma
        self.critic = CriticNN(c_s, c_h)

        # optimizer
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        v_state = self.critic(state)
        v_next_state = self.critic(next_state)
        td_error = v_state - (reward + self.gamma * v_next_state)
        loss = td_error ** 2
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=1)
    args = parser.parse_args()
    config = configs["config" + str(args.config)]
    env = Env(**config)

    c_h = config["c_h"]
    c_s = config["c_s"]
    algo = CriticFA(c_s, c_h, gamma=1)

    state = env.reset()
    done = False
    rewards = []
    while not done:
        action = algo.critic.actor(state).clone().detach().numpy()
        next_state, reward, done, info = env.step(action)
        algo.update(state, action, reward, next_state)
        state = next_state
    mean, std = env.plot_rewards("Custom_NN_CFA.png")
    print(f"Mean: {mean}, Std: {std}")

    print(algo.critic.weights)

