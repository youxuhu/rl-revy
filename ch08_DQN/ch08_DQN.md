# 8. ch08_DQN
DQN是基于Q学习和神经网络的方法，添加了新的技术，“经验回放”和“目标网络”。
## 8.1. OpenAiGym
简单的使用方法
```

import gymnasium as gym

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state = env.reset()#初始状态
    print(state)

    action_space = env.action_space 
    print(action_space)#行动的维度

    action = 0
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(next_state)
```
### 8.1.2 随机智能代理
使用一个随机智能代理我们可以快速地得到崩坏的结果。
```

import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = 'human')
state = env.reset()
done = False

while not done:
    frame = env.render()

    action = np.random.choice([0, 1])
    next_state, reward, terminate, truncated, info = env.step(action)
    done = terminate or truncated
    
env.close()
```
openai gym 中使用 'observation' 来代替 'state'，并且所有的api都由observation来命名
## 8.2 DQN的核心技术
在Q学习中通常使用估计值来更新估计值，这样会产生变得不稳定的趋势。与此同时再加入神经网络这种表现力强的函数近似，则结果会变得更加的不稳定。

### 8.2.1 经验回放
在Q学习中，每当只能代理对环境采取行动时都会产生数据。
具体来说在某个时刻t得到的$E_t=(s_t, a_t, r_t, s_{t+1})$，我们将$E_t$称为“经验数据”。
但是经验数据是强相关的，也就是说Q学习使用强相关的数据进行计算这样会产生偏差。弥补这个差异的技术包括经验回放。

使用经验回放进行训练的过程非常的简单。首先我们将智能代理的数经验数据$E_t$保存到缓存区中。然后，在更新函数Q的时候，从缓冲区随机的取出一个小批次的经验数据进行训练。
### 8.2.2 经验回放的实现
经验回放的缓冲区中无法存储无限多的数据。所以，我们需要事先决定他的最大容量。我们使用队列可以快速地实现这个目标。
```

from collections import deque
import random
import numpy as np
import common

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
```
在初始化阶段接受buffer_size和batch_size两个参数。buffer_size是缓冲区最大容量，batch_size是每一次小批量训练时的大小。
使用add方法将经验添加到缓冲区，通过len(replay_buffer)可以得到当前缓冲区中经验的数量，通过get_batch方法可以随机的从缓冲区取出一个小批量的经验数据。
完整的代码:
```

from collections import deque
import random
import numpy as np
import common

import gymnasium as gym

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

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

    for episode in range(10):
        state, info = env.reset()
        done = False
        while not done:
            action = 0
            next_state, reward, terminate, truncated, info = env.step(action)
            done = terminate or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

    state, action, reward, next_state, done = replay_buffer.get_batch()#得到小批次
    print(state.shape)
    print(action.shape)
    print(reward.shape)
    print(next_state.shape)
    print(done.shape)

    env.close()


```
### 8.2.3 目标网络
在监督学习中我们要为训练数据添加正确的答案标签。在这种情况下输入正确答案标签不变。以MNIST数据集为例，输入的图像为7，那么该标签总是为7.
在Q学习中我们的更新目标会随着训练的进行而改变。为了弥补这种差异我们使用一种固定的TD目标的方法，叫做目标网络。

目标网络是一个独立的神经网络，结构和参数与Q网络相同。我们使用目标网络来计算TD目标，而不是使用Q网络。
目标网络的参数每隔一段时间就会被Q网络的参数更新一次。
