import gymnasium as gym
import random

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
actions = env.action_space.n

episodes = 5
for episode in range(1, episodes+1):
    state, info = env.reset()
    #print(state,info)
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.randrange(0, actions, 1)
        n_state, reward, done, truncated, info = env.step(action)
        score += reward
        #print('n_state:{}, reward:{}, done:{}, truncated:{}, info:{}'.format(n_state, reward, done, truncated, info))
    print('Episode:{} Score:{}'.format(episode, score))
env.close() 