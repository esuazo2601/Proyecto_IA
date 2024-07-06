import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.taxi import TaxiEnv

def dfs(env, state, visited, path, passenger_picked, TE):
    taxi_row, taxi_col, passenger, dest = TE.unwrapped.decode(state)
    
    if passenger == dest:
        return path, 1  # Return path and reward of 1 for reaching the goal

    if state in visited:
        return None, 0  # If state already visited, return None

    visited.add(state)

    # If passenger not picked up and we are at passenger's location
    if not passenger_picked and passenger < 4 and (taxi_row == env.locs[passenger][0] and taxi_col == env.locs[passenger][1]):
        action = 4  # Pickup action
        next_state, _, _, _, _ = env.step(action)
        return dfs(env, next_state, visited, path + [action], True, TE)

    # If passenger picked up and we are at destination
    if passenger_picked and passenger == 4 and (taxi_row == env.locs[dest][0] and taxi_col == env.locs[dest][1]):
        action = 5  # Dropoff action
        next_state, _, _, _, _ = env.step(action)
        return path + [action], 1  # Return path and reward of 1 for reaching the goal

    possible_moves = [0, 1, 2, 3]  # South, North, East, West
    for action in possible_moves:
        env.unwrapped.s = state  # Restore the state
        next_state, _, _, _, _ = env.step(action)
        subpath, subreward = dfs(env, next_state, visited, path + [action], passenger_picked, TE)
        if subpath is not None:
            return subpath, subreward
    
    visited.remove(state)
    return None, 0  # No valid path found, return 0 reward

def run_dfs(episodes):
    env = gym.make('Taxi-v3', render_mode=None)

    rewards_per_episode = np.zeros(episodes)
    
    for episode in range(episodes):
        initial_state = env.reset()[0]  # Obtain initial state
        TE = TaxiEnv()
        visited = set()
        path, reward = dfs(env, initial_state, visited, [], False, TE)
        
        rewards_per_episode[episode] = reward

    env.close()
    
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title('Recompensas por episodio usando DFS en Taxi')
    plt.savefig('taxi_dfs.png')

if __name__ == '__main__':
    run_dfs(1000)
