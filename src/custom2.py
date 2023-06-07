from env import Env
from demand_dists import *
from config import configs
import argparse
import torch
import torch.nn as nn
torch.manual_seed(42)

class CriticNN(nn.Module):
    def __init__(self, obs_size) -> None:
        super().__init__()

        self.part_classifier = nn.Sequential(
            nn.Linear(obs_size, 2),
            nn.Sigmoid(),
        )

        self.critic = nn.Sequential(
            nn.Linear(2*obs_size, 4),
            nn.Linear(4, 1),
            nn.Tanh()
        )
    
    def forward(self, state):
        part = self.part_classifier(state)
        out = self.critic(torch.cat([state, part]))
        return out
    
    def freeze(self):
        for param in self.part_classifier.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.part_classifier.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True



class Algorithm:
    def __init__(self, gamma) -> None:
        self.critic = CriticNN(2)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)
        self.gamma = gamma
        self.min_state=6
        self.max_state=12
    
    def actor(self, state):
        self.min_state = max(min(self.min_state, state[0]), 7)
        self.max_state = min(max(self.max_state, state[1]), 15)
        state = torch.tensor(state, dtype=torch.float32)
        sc = state.clone().detach()
        with torch.no_grad():
            max_V = -float('inf')
            max_state = None
            for s in np.arange(self.min_state, self.max_state+1):
                sc[0] = s
                val = self.critic(sc)
                if val > max_V:
                    max_V = val
                    max_state = s
        if state[0] < max_state:
            return max_state - state
        else:
            return torch.tensor(0.0)
    
    def actor2(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        max_state = state.clone().detach().reshape(1,)
        max_state.requires_grad = True
        self.critic.freeze()
        for _ in range(10):
            v_state = self.critic(max_state)
            grad_state, = torch.autograd.grad(v_state, max_state)
            max_state = max_state - grad_state
        self.critic.unfreeze()
        if state < max_state:
            return max_state - state
        else:
            return torch.tensor(0.0)
    
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
    algo =  Algorithm(gamma = 1)
    gradient_steps = 2
    state = env.reset()
    done = False
    rewards = []
    while not done:
        print(env.t, end='\r')
        action = algo.actor(state).clone().detach().numpy()
        next_state, reward, done, info = env.step(action)
        for _ in range(gradient_steps):
            algo.update(state, action, reward, next_state)
        state = next_state
    mean, std = env.plot_rewards("Custom_NN_CFA.png")
    print(f"Mean: {mean}, Std: {std}")