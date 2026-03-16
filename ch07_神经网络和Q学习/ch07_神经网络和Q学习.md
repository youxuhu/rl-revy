# 7. 神经网络和Q学习
在简单的问题中将Q函数保存为表格是可行的，但是在复杂问题中是不现实的。
为了解决这个问题我们考虑使用简单紧凑的函数来近似Q函数。最有力的方法便是神经网络。
## 7.1 DeZero 介绍

使用variable的自动微分模块，与torch中的张量的操作类似

```
import numpy as np
import common

from dezero import Variable

if __name__ =="__main__":
    x_np = np.array(5.0)
    x = Variable(x_np)

    y = 3 * x **2
    print(y)
    y.backward()
    print(x.grad)
```

### 7.1.2 多维数组和函数

```

import numpy as np
import common

from dezero import Variable
import dezero.functions as F


if __name__ =="__main__":
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    a, b = Variable(a), Variable(b)
    c = F.matmul(a, b)
    print(c)

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = F.matmul(a, b)
    print(c)
    
```
matmul计算了向量内积和矩阵乘法

### 7.1.3 最优化
$y=100(x_1 + x_0^2)^2+(x_0-1)$
使用Dezero来计算上面函数的最小值，
```

import numpy as np
import common

from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 **2) ** 2 + (x0 - 1) ** 2
    return y

if __name__ == "__main__":
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)
```
我们通过这个例子可以快速得到函数的梯度，-2， 400.
我们可以使用梯度下降法来得到最小的点
```

import numpy as np
import common

from dezero import Variable



def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 **2) ** 2 + (x0 - 1) ** 2
    return y


if __name__ == "__main__":
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    lr = 0.001 #学习lv 
    iters = 10000

    for i in range(iters):
        print(x0, x1)
        y = rosenbrock(x0, x1)
        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data

    print(x0, x1)

```
这样我们便可以求道函数的最小值。
## 7.2 线性回归
机器学习使用诗句来解决问题，现在我们来实现最简单的线性回归
### 7.2.1 玩具数据集
实验使用的小数据集被称为玩具数据集
### 7.2.2 线性回归的理论知识
我们的目标是找到拟合数据的直线，为此我们需要减小数据和预测值之间的差，这个差叫做残差。
$L=\frac{1}{N}\sum_{i=1}^N(Wx_i+b-y_i)^2$
这个式子叫做均方差。
### 7.2.3 线性回归的实现
```

import numpy as np
import common

from dezero import Variable
import dezero.functions as F




np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2)/len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print("="*10)
print('W = ', W.data)
print('b = ', b.data)
```
## 7.3 神经网络
### 7.3.1 非线性数据集
### 7.3.2 线性变化和激活函数
```
y = F.matmul(s, W) + b
y = F.line(x, W, b)
```
下面的一个式子叫做线性变换或者仿射变换
用于全连接层
### 7.3.3 神经网络的实现
我们使用真实的数据来实现一个神经网络
```

import numpy as np
import common

from dezero import Variable
import dezero.functions as F

if __name__ == "__main__":
    np.random.seed(0)

    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    #参数初始化
    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.rand(H, O))
    b2 = Variable(np.zeros(O))

    #神经网络的推理
    def predict(x):
        y = F.linear(x, W1, b1)
        y = F.sigmoid(y)
        y = F.linear(y, W2, b2)
        return y
    
    lr = 0.2
    iters = 10000

    #神经网络的训练
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()

        loss.backward()

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data

        if i% 1000 ==0:
            print(loss.data)
    
```
这样我们便成功搭建了一个神经网络，只使用了两个全连接层。
### 7.3.4 层和模型
在dezero中可以使用Linear快速的实现一个层
```
Linear(out_size, nobias = False, dtype=np.float32, in_size=None)
```
out_size是输出大小, in_size是输入大小
Linear的使用大小:
```

import numpy as np
import common

from dezero import Variable
import dezero.functions as F
import dezero.layers as L

if __name__ == "__main__":
    linear = L.Linear(10)#只定义输出大小

    batch_size , input_size = 100, 5
    x = np.random.randn(batch_size, input_size)
    y = linear(x)

    print('y_shape', y.shape)
    print('params shape', linear.W.shape, linear.b.shape)

    for param in linear.params():
        print(param.name, param.shape)
```
这样便成功的使用了linear,在linear中如果不输入input_size那么在x输入时会自动获取对应的input_size
这和pytorch的操作是相同的，我们可以使用model来快速的构建多层的神经网络
```

import numpy as np
import common

from dezero import Model
import dezero.functions as F
import dezero.layers as L

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(x)
        return y

if __name__ == "__main__":
    model = TwoLayerNet(10, 1)
    for param in model.params():
        print(param)
    model.cleargrads()#清除梯度
```
完整的两层神经网络
```

import numpy as np
import common

from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero import Model

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    iters = 10000

    class TwoLayerNet(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)

        def forward(self, x):
            y = F.sigmoid(self.l1(x))
            y = self.l2(y)
            return y

    model = TwoLayerNet(10, 1)
    for i in range(iters):
        y_pred = model.forward(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= p.grad.data * lr
        if i % 1000 == 0:
            print('loss:',loss)
    
```
使用优化器的两层神经网络
```

import numpy as np
import common

from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers


if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.01
    iters = 100000
    class TwoLayerNet(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)
            
        def forward(self, x):
            y = F.sigmoid(self.l1(x))
            y = self.l2(y)
            return y
        
    model = TwoLayerNet(10, 1)
    optimizer = optimizers.Adam(lr)

    optimizer.setup(model)

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print('loss', loss.data)

```


## 7.4 Q学习和神经网络
本章开始Q学习和神经网络的融合
### 7.4.1 神经网络的预处理
将3*4的网格世界转化为onehot编码
```

import numpy as np
import common

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT*WIDTH, dtype = np.float64)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]




if __name__ == "__main__":
    state = (2, 0)
    x = one_hot(state)
    print(x.shape)
    print(x)    
```
我们通过vec[np.newaxis,:]添加了一个竖向的轴

### 7.4.2 表示Q函数的神经网络
典型的两种结构有：
- 状态、行动两个输入，输出Q函数的值
- 状态作为输入，输出多个行动

对于第一种结构有多少个action就需要前向传播多少次，第二个结构将会更加的高效只需要一次前向传播就可以求得所有的行动价值函数Q

我们现在实现第二种结构，只需要使用state作为输入参数

```

import numpy as np
import common

from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers

class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y



def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT*WIDTH, dtype = np.float64)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]




qnet = QNet()
state = (2, 0)
state = one_hot(state)

qs = qnet.forward(state)
print(qs.shape) #(1, 4)
```


### 7.4.3 神经网络和Q学习


$Q(S_t, A_t) + \alpha[R_t+\gamma Q(S_{t+1}, A_{t+1}) -Q(S_t, A_t) ]$

将更新目标当作T，那么T可以被视为正确答案的标签，由于T是标量值，所以可以被视为回归问题。
```

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
```