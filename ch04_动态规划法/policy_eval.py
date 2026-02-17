from common.gridworld import *
from collections import defaultdict


def eval_onestep(pi, V, env, gamma = 0.9):
    
    # 遍历环境中的所有状态
    for state in env.states():  # env.states()返回所有可能的状态
        
        # 处理目标状态（终止状态）
        if state == env.goal_state:  # 如果当前状态是目标状态
            V[state] = 0  # 目标状态的价值始终为0（没有未来收益）
            continue  # 跳过后续计算，直接处理下一个状态
        
        # 获取当前状态下各行动的选择概率
        action_probs = pi[state]  # 例如：{'left': 0.25, 'right': 0.25, 'up': 0.25, 'down': 0.25}
        
        # 初始化当前状态的新价值为0
        new_V = 0

        # 遍历所有可能的行动
        for action, action_prob in action_probs.items():  # action_prob是采取该行动的概率
            # 根据当前状态和选择的行动，确定下一个状态
            next_state = env.next_state(state, action)
            
            # 获取执行该行动后获得的即时奖励
            r = env.reward(state, action, next_state)

            # 贝尔曼期望方程的核心计算：
            # 累加：行动概率 × (即时奖励 + 折扣因子 × 下一个状态的价值)
            # 注意：这里使用旧的V[next_state]值进行同步更新
            new_V += action_prob * (r + gamma * V[next_state])
        
        # 更新当前状态的价值为新计算的值
        V[state] = new_V
    
    # 返回更新后的价值函数
    return V


def policy_eval(pi, V, env, gamma, threshold = 0.001):
    
    # 无限循环，直到价值函数收敛
    while True:
        # 保存旧的价值函数，用于比较变化
        old_V = V.copy()  # 使用copy()避免引用同一个字典对象
        
        # 执行一次迭代更新，得到新的价值函数
        V = eval_onestep(pi, V, env, gamma)
        
        # 初始化最大变化量为0
        delta = 0
        
        # 遍历所有状态，找出价值变化的最大值
        for state in V.keys():
            # 计算当前状态的价值变化绝对值
            t = abs(V[state] - old_V[state])
            
            # 更新最大变化量
            if delta < t:
                delta = t
        
        # 检查收敛条件：如果所有状态的价值变化都小于阈值
        if delta < threshold:
            break  # 停止迭代，已收敛
    
    # 返回收敛后的价值函数
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9

    pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda: 0.0)
    V = policy_eval(pi, V, env, gamma)
    env.render_v(V)