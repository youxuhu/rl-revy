import copy
import numpy as np
import common

from dezero import Variable
from dezero import optimizers
from dezero import Model
import dezero.functions as F
import dezero.layers as L
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.stack([x[1] for x in data])
        reward = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done

class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.action_size = 2
        self.buffer_size = 10000
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr).setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        else:
            state = state[np.newaxis, :]
            qs = self.qnet(state)
            return int(qs.data.argmax())
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + self.gamma * next_q * (1 - done)

        loss = F.mean_squared_error(q, target)
        
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    agent = DQNAgent()
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    reward_history = []

    for episode in range(episodes):
        agent.epsilon = 0
        state, info  = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminate, truncated, info = env.step(action)

            done = terminate or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)

    env.close()

    episode_returns = np.array(reward_history, dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(episode_returns) + 1), episode_returns, linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
        