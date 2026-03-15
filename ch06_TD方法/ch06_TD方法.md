# 6 ch06_TD方法
对于蒙特卡洛方法，只有在到达回合的终点时才能更新价值函数。因为只有在回合结束时才可以确定收益。
对于连续性任务不可以使用蒙特卡洛方法。即便是对于回合制任务，如果一个回合需要花费大量的时间，那么蒙特卡洛方法也会花费大量的时间。
TD（Temporal Difference）方法不使用环境模型，每次行动都会更新价值函数。

## 6.1 使用TD方法评估策略

### 6.1.1 TD方法的推导
收益的定义是: $G_t=R_t+\gamma G_{t+1}$

价值函数的定义：
$v_\pi(s)=E_\pi[G_t|S_t=s]$         (6.3)
$=E_\pi[R_t+\gamma G_{t+1}|S_t=s]$  (6.4)

- 使用MC的方法可以从6.3导出
- 使用TD的方法可以从6.4导出
TD方法的更新公式为：
$V'_\pi(s)=\sum_{a,s'}\pi(a|s)p(s'|s,a)[r(s,a,s')+\gamma V_\pi(s')]$
这个式子的是当前的价值函数是基于下一个状态的价值函数来更新的。这个过程的特点是需要考虑所有的迁移。

TD方法只使用下一个行动和价值函数来更新当前的价值函数。
- 像dp方法一样，通过"自举"的方式就可以依次更新价值函数。
- 像mc方法一样，TD方法无需了解环境的相关知识，使用采样数据就可以对价值函数进行更新。

接下来使用数学方法来推导TD方法的更新公式。
$v_\pi(s)=E_\pi[R_t+\gamma v_\pi(S_{t+1})|S_t=s]$
根据样本数据近似计算$R_t+\gamma v_\pi(S_{t+1})$的部分
可以得到TD方法的更新式：
$V'_\pi(S_t)=V_\pi(S_t)+\alpha[R_t+\gamma V_\pi(S_{t+1})-V_\pi(S_t)]$
目标是$R_t+\gamma v_\pi(S_{t+1})$，也叫做TD目标。TD方法会朝着TD目标更新当前的价值函数。
### 6.1.2 TD方法和MC方法的比较
相比于MC方法，TD方法的更新频率更高，可以在每一步行动后就更新家住函数。
### 6.1.3 TD方法的实现
```python
from collections import defaultdict
import numpy as np

from common.gridworld import GridWorld

class TDAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state] #目标的价值函数为0
        target = reward + self.gamma * next_V

        self.V[state] += (target -self.V[state]) * self.alpha
    

if __name__ == "__main__":
    env = GridWorld()
    agent = TDAgent()
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state
    env.render_v(agent.V)
```

## 6.2 SARSA
同策略型SARSA来进行策略更新
### 6.2.1 同策略型的SARSA
在这个情况下我们不适用价值函数V来评估策略，我们使用行动价值函数Q来进行策略评估.这样我们可以不以来环境，并且可以直接使用贪婪方法。
$\mu(s)=argmax_aQ_\pi(s,a)$
下面将是TD方法式子中的状态价值函数的更新式：
$Q'_\pi(S_t,A_t)=Q_\pi(S_t,A_t)+\alpha[R_t+\gamma Q_\pi(S_{t+1},A_{t+1})-Q_\pi(S_t,A_t)]$

在同策略型中不可以直接使用贪婪化，这样会导致无法探索。

### 6.2.2 SARSA的实现
```python

from collections import defaultdict, deque
import numpy as np
from common.utils import greedy_probs
from common.gridworld import GridWorld

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_action = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_action)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)#使用deque，deque的使用方法和列表相同，但是如果添加maxlen参数，那么在添加元素是，如果超过了最大长度，那么会先进先出（队列）。

    def get_action(self, state):
        action_probs = self.pi[state] #从pi中选择行动
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()
    
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory)<2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        #下一个Q函数
        next_q = 0 if done else self.Q[(next_state, next_action)]
        #使用TD方法进行更新
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target -self.Q[state, action])*self.alpha
        #策略的改进
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)#到达目标时的调用，因为action和reward都将没有，所以传入None。
                break
            state = next_state
    
    env.render_q(agent.Q)   
```

## 6.3 异策略型的SARSA
### 6.3.1 异策略型和重要性采样
在异策略型中，智能代理拥有两种策略，即行为策略和目标策略。智能代理首先会居于行为策略，通过采取各种行动收集大量数据，然后再通过使用样本数据贪婪的更新目标策略。
需要注意以下两点：
- 如果行为策略和目标策略概率分布相似，则结果会更稳定（方差更小）。考虑到这一点我们使用当前的Q函数的行为策略进行$\epsilon-greedy$更新，对目标策略使用贪婪更新。
- 由于两种策略不同，因此我们使用重要性采样来校正权重$\rho$

考虑更新Q函数的式子：
$Q'_\pi(S_t,A_t)=Q_\pi(S_t,A_t)+\alpha[R_t+\gamma Q_\pi(S_{t+1},A_{t+1})-Q_\pi(S_t,A_t)]$

在异策略型中，我们需要使用重要性采样来校正权重$\rho$，因此更新式子为：
$Q'_\pi(S_t,A_t)=Q_\pi(S_t,A_t)+\alpha\rho[R_t+\gamma Q_\pi(S_{t+1},A_{t+1})-Q_\pi(S_t,A_t)]$
其中$\rho$的定义为：
$\rho=\frac{\pi(A_t|S_t)}{b(A_t|S_t)}$
其中$\pi$是目标策略，$b$是行为策略。

### 6.3.2 异策略型SARSA的实现
```python
from collections import defaultdict, deque
import numpy as np
from common.utils import greedy_probs
from common.gridworld import GridWorld

class SarsaOffPolicy:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_action = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_action)
        self.b = defaultdict(lambda: random_action)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen = 2)
    
    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory)<2:
            return
        
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 0
        else:
            next_q = self.Q[next_state, next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho*(reward + self.gamma * next_q)
        self.Q[state, action] += (target -self.Q[state, action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, action_size=self.action_size)# 对目标使用贪心算法
        self.b[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)



if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaOffPolicy()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state
    
    env.render_q(agent.Q)
```

## 6.4 Q学习
