import numpy as np
import common
from common.gridworld import GridWorld
import matplotlib.pyplot as plt

from dezero import Model
import dezero.functions as F
import dezero.layers as L
import dezero.optimizers as optimizers


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)
    def get_action(self, state):
        if np.random.rand() < self.epsilon:#探索，小于epsilon就随机，否则就贪婪
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return qs.data.argmax()
        

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1, dtype=np.float64)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = reward + self.gamma * next_q
        qs = self.qnet(state)#前向传播
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT*WIDTH, dtype = np.float64)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


def build_q_dict(agent, env):
    q_dict = {}
    for state in env.states():
        qs = agent.qnet(one_hot(state)).data.flatten()
        for action, q_value in enumerate(qs):
            q_dict[(state, action)] = float(q_value)
    return q_dict


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    loss_history = []

    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)

            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state

        average_loss = total_loss / cnt
        loss_history.append(average_loss)

    plt.plot(loss_history)
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.title('Q-Learning Loss')
    plt.tight_layout()
    plt.show()

    q = build_q_dict(agent, env)
    env.render_q(q)



        

