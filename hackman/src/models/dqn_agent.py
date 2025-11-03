# src/models/dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        samples = random.sample(self.buf, batch_size)
        batch = Transition(*zip(*samples))
        return batch

    def __len__(self):
        return len(self.buf)

class DQNNet(nn.Module):
    def __init__(self, input_dim, hidden=[256,128], action_dim=26):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, lr=1e-4, gamma=0.99, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DQNNet(state_dim).to(self.device)
        self.target = DQNNet(state_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay = ReplayBuffer()
        self.steps = 0

    def select_action(self, state_vec, guessed_mask, eps):
        # state_vec: numpy array
        if random.random() < eps:
            # pick random unseen letter
            unseen = [i for i in range(26) if guessed_mask[i]==0]
            if not unseen:
                return random.randrange(26)
            return random.choice(unseen)
        self.net.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            q = self.net(x).cpu().numpy().squeeze()
            # mask guessed
            q = q.copy()
            q[guessed_mask==1] = -1e9
            return int(q.argmax())

    def store(self, *args):
        self.replay.push(*args)

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        state = torch.FloatTensor(np.stack(batch.state)).to(self.device)
        action = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.stack(batch.next_state)).to(self.device)
        done = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        q_vals = self.net(state).gather(1, action)
        # double DQN style target
        with torch.no_grad():
            next_q = self.target(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_vals, target_q)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.steps += 1
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.target.load_state_dict(self.net.state_dict())
