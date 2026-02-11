import numpy as np
import matplotlib.pyplot as plt


class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) #每次玩都会改变胜率
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.Qs = np.zeros(actions)
    
    def update(self, action, reward):
        #使用alpha更新法更新Q值
        self.Qs[action] += self.alpha * (reward - self.Qs[action])
    
    def get_action(self):
        if np.random.rand() <self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)
        
# if __name__ == "__main__":
#     steps = 1000
#     epsilon = 0.1
#     alpha = 0.1

#     bandit = NonStatBandit()
#     agent = AlphaAgent(epsilon, alpha)

#     total_reward = 0
#     total_rewards = []
#     rates = []

#     for step in range(steps):
#         action = agent.get_action()
#         reward = bandit.play(action)
#         agent.update(action, reward)
#         total_reward += reward

#         total_rewards.append(total_reward)
#         rates.append(total_reward / (step + 1))
    
#     print(total_reward)
#     plt.ylabel("Total Reward")
#     plt.xlabel("Steps")
#     plt.plot(total_rewards)
#     plt.show()
    
#     plt.ylabel("Rates")
#     plt.xlabel("Steps")
#     plt.plot(rates)
#     plt.show()


if __name__ == "__main__":
    runs = 200
    steps = 2000
    epsilon = 0.1
    alpha = 0.1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = NonStatBandit()
        agent = AlphaAgent(epsilon, alpha)

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

