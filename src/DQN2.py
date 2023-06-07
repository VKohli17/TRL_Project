from collections import deque, namedtuple
import math
from env import Env
import torch
import random
import torch.nn as nn
from config import configs
import argparse
random.seed(42)
torch.manual_seed(42)

Step = namedtuple('Step', ('obs', 'action', 'reward', 'next_obs'))

class Buffer:
    def __init__(self, max_size) -> None:
        self.buffer = deque([], maxlen=max_size)
    
    def add(self, *args):
        self.buffer.append(Step(*args))
    
    def sample(self, size):
        if len(self.buffer) < size:
            return random.sample(self.buffer, len(self.buffer))
        return random.sample(self.buffer, size)


class QNetwork(nn.Module):
    def __init__(self, obs_size, act_size) -> None:
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, act_size)
        )
    
    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, obs_size, act_size, buffer_size, grad_steps, learn_after) -> None:
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.obs_size = obs_size
        self.act_size = act_size
        self.grad_steps = grad_steps
        self.learn_after = learn_after
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps = self.eps_start

        self.buffer = Buffer(buffer_size)

        self.policy_net = QNetwork(obs_size, act_size).to(self.device)
        self.target_net = QNetwork(obs_size, act_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss() #nn.MSELoss()

    def actor(self, obs):
        rand = random.random()
        if rand < self.eps:
            return torch.randint(0, self.act_size, size=(1,1)).to(self.device)
        else:
            return self.policy_net(obs).argmax().view(1,1)


    def learn(self):
        mini_batch = Step(*zip(*self.buffer.sample(self.batch_size)))
        states = torch.cat(mini_batch.obs)
        actions = torch.cat(mini_batch.action)
        rewards = torch.cat(mini_batch.reward)
        next_states = torch.cat(mini_batch.next_obs)

        Q_sa = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            Q_sa_next = self.target_net(next_states).max(1)[0].view(-1,1)
        
        target = rewards + self.gamma * Q_sa_next
        loss = self.criterion(Q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def train(self, env):
        done = False
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
        step = 0
        while not done:
            print(step, end='\r')
            action = self.actor(obs)
            next_obs, reward, done, info = env.step(action.item())
            step += 1
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step / self.eps_decay)
            # self.eps = max(self.eps_end, self.eps * self.eps_decay)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).view(1,1)
            self.buffer.add(obs, action, reward, next_obs)
            obs = next_obs

            if step > self.learn_after:
                for _ in range(self.grad_steps):
                    self.learn()
            
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=int, default=1)
args = parser.parse_args()
config = configs["config" + str(args.config)]
env = Env(**config)

dqn = DQN(
    batch_size=128,
    gamma=0.999,
    eps_start=0.3,
    eps_end=0.05,
    eps_decay=100,
    tau=0.05,
    lr=5e-4,
    obs_size=2,
    act_size=16,
    buffer_size=1000,
    grad_steps=1,
    learn_after=128
)

dqn.train(env)
mean, std = env.plot_rewards("dqn.png")
print(f"Mean: {mean}, Std: {std}")