from common.gridworld import *
from policy_iter import greedy_policy
from collections import defaultdict

def value_iter_onestep(V, env, gamma):
    for state in env.states(): # 访问所有的状态
        if state == env.goal_state: # 目的地的价值数总是为0
            V[state] = 0
            continue

        action_values = []
        for action in env.actions(): # 访问所有的行动
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=1e-3, is_render=True):
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy()#更新前的价值函数
        V = value_iter_onestep(V, env, gamma)
        #求更新量的最大值
        delta = 0
        for state in V.keys():
            t = abs(V[state]-old_V[state])
            if delta < t:
                delta = t
        #与阈值比较判断是否收敛
        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)