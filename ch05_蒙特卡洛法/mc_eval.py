from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions) # 策略
        self.V = defaultdict(lambda: 0)# 价值函数
        self.cnts = defaultdict(lambda: 0)# 用于通过增量式实现收益的平均处理值上
        self.memory = []# 持续记录状态、动作、奖励的列表

    def get_action(self, state):
        action_prob = self.pi[state]
        actions = list(action_prob.keys())
        probs = list(action_prob.values())
        return np.random.choice(actions, p = probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()
    
    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action , reward = data
            G = self.gamma *G + reward
            self.cnts[state] += 1
            self.V[state] += (G-self.V[state])/self.cnts[state]


if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()
    episodes = 1000
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)

            if done:
                agent.eval()
                break

            state = next_state
    env.render_v(agent.V)