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