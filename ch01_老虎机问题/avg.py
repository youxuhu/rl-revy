import numpy as np
np.random.seed(0)#固定的随机种子
# rewards = []
# for n in range(1, 11):
#     reward = np.random.rand()#虚拟的奖励
#     rewards.append(reward)
#     Q = sum(rewards) / n
#     print(Q)

Q = 0
for n in range(1, 11):
    reward = np.random.rand()#虚拟的奖励
    # Q = Q + (reward - Q ) / n
    Q += (reward - Q )/n
    print(Q)
