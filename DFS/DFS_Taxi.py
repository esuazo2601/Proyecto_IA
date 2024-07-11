import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.taxi import TaxiEnv
import time

def dfs(env, state, visited, path, passenger_picked, TE):
    # Decode the current state
    taxi_row, taxi_col, passenger, dest = TE.unwrapped.decode(state)

    # Check if the state has already been visited
    if state in visited:
        return None, 0

    # Mark this state as visited
    visited.add(state)

    # If the passenger is picked up and we're at the destination, drop off the passenger
    if passenger_picked and (taxi_row == env.locs[dest][0] and taxi_col == env.locs[dest][1]):
        action = 5  # Dropoff action
        env.unwrapped.s = state  # Restore state
        next_state, _, done, _, _ = env.step(action)
        if done:
            return path + [action], 1

    # If the passenger is not picked up and we're at the passenger's location, pick up the passenger
    if not passenger_picked and passenger < 4 and (taxi_row == env.locs[passenger][0] and taxi_col == env.locs[passenger][1]):
        action = 4  # Pickup action
        env.unwrapped.s = state  # Restore state
        next_state, _, _, _, _ = env.step(action)
        result, reward = dfs(env, next_state, visited, path + [action], True, TE)
        if reward > 0:
            return result, reward

    # Explore all possible actions (0: south, 1: north, 2: east, 3: west)
    for action in range(4):
        env.unwrapped.s = state  # Restore state
        next_state, _, done, _, _ = env.step(action)
        if done:
            return path + [action], 1
        result, reward = dfs(env, next_state, visited, path + [action], passenger_picked, TE)
        if reward > 0:
            return result, reward

    # Backtrack if no valid path found
    visited.remove(state)
    return None, 0

def run_dfs(episodes):
    env = gym.make('Taxi-v3')
    TE = TaxiEnv()

    rewards_per_episode = np.zeros(episodes)
    
    for episode in range(episodes):
        initial_state = env.reset()[0]  # Obtain initial state
        visited = set()
        path, reward = dfs(env, initial_state, visited, [], False, TE)
        
        # # Muestra el camino encontrado para el modo humano
        # if path:
        #     env = gym.make('Taxi-v3', render_mode='human')
        #     env.reset()
        #     env.unwrapped.s = initial_state
        #     for action in path:
        #         env.step(action)
        #         env.render()
        
        rewards_per_episode[episode] = reward

    env.close()
    
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title('Recompensas por episodio usando DFS en Taxi')
    plt.savefig('taxi_dfs.png')

if __name__ == '__main__':
    start = time.time()
    run_dfs(1000)
    end = time.time()
    print("TIME: ",(end-start))

