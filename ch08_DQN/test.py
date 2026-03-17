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