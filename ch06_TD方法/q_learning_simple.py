from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9  # 折扣因子
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = np.argmax(next_qs)

        target = self.gamma * next_q_max + reward
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
