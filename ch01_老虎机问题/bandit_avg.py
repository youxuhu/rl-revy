import numpy as np
import matplotlib.pyplot as plt


# epsilon-greedy算法实现的Agent类
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)#每一台机器的平均奖励
        self.ns = np.zeros(action_size)#每一台机器被选择的次数
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Qs)) #随机选择一台机器
        else:
            return np.argmax(self.Qs) #选择平均奖励最高的机器
    

#老虎机的实现
class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)#每一台机器的胜率(返回一个数组)
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0



runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []
    
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))
 
    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

#绘制图形
plt.ylabel("Rates")
plt.xlabel("Steps")

plt.plot(avg_rates)
plt.show()