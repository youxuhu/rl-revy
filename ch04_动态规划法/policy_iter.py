from collections import defaultdict
from policy_eval import policy_eval
from common.gridworld import *
from common.gridworld_render import *

# argmax函数：返回字典中值最大的键
def argmax(d):
    max_value = max(d.values())
    max_keys = 0
    for key, value in d.items():
        if value == max_value:
            max_keys = key
    return max_keys

# 使用argmax函数将价值函数贪婪化
def greedy_policy(V, env, gamma):
    pi = {}
    for state in env.states():
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
        
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=1e-3, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})#初始化的pi是一个均匀随机策略，即在每个状态下选择每个行动的概率都是0.25
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)#调用policy_eval函数评估当前策略pi的价值函数V，得到更新后的价值函数V
        new_pi = greedy_policy(V, env, gamma)#根据价值函数V使用greedy_policy函数生成一个新的策略，该策略在每个状态下选择使得价值函数最大的行动。
        if is_render:
            env.render_v(V, pi)
        
        if new_pi == pi:# 检查更新后的策略是否与之前的策略相同，如果相同则说明策略已经收敛，可以停止迭代
            break
        pi = new_pi
    return pi

if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    pi = policy_iter(env, gamma, is_render=True)
    print(pi)

