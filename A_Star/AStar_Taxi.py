import heapq
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
# Recompensa global de los episodios
TotalRecom = 0

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def a_star(env, start_state):
    global TotalRecom
    taxi_row, taxi_col, passenger, dest = env.unwrapped.decode(start_state)

    # Priority queue para A*
    priority_queue = []
    heapq.heappush(priority_queue, (0, start_state, [], False))

    visited = set()

    while priority_queue:
        current_cost, state, path, passenger_picked = heapq.heappop(priority_queue)
        
        if state in visited:
            continue
        
        visited.add(state)
        
        taxi_row, taxi_col, passenger, dest = env.unwrapped.decode(state)

        # Si el pasajero está en el taxi y se esta en la posición de destino, se deja al pasajero
        if passenger_picked and (taxi_row == env.locs[dest][0] and taxi_col == env.locs[dest][1]):
            action = 5  # Dropoff action
            env.unwrapped.s = state  # Recupera el estado
            next_state, recomp, done, _, _ = env.step(action)
            TotalRecom += recomp
            if done:
                return path + [action], 1

        # Si el pasajero no está en el taxi y se esta en la posición del pasajero, se sube al taxi
        if not passenger_picked and passenger < 4 and (taxi_row == env.locs[passenger][0] and taxi_col == env.locs[passenger][1]):
            action = 4  # Pickup action
            env.unwrapped.s = state  # Restore state
            next_state, recomp1, _, _, _ = env.step(action)
            TotalRecom += recomp1
            heapq.heappush(priority_queue, (current_cost + 1, next_state, path + [action], True))
            continue

        # Explora las posibles acciones (0: south, 1: north, 2: east, 3: west)
        for action in range(4):
            env.unwrapped.s = state  # Recupera el estado
            next_state, recomp2, _, _, _ = env.step(action)
            TotalRecom += recomp2
            next_taxi_row, next_taxi_col, _, _ = TE.unwrapped.decode(next_state)

            if passenger_picked:
                heuristic = manhattan_distance((next_taxi_row, next_taxi_col), env.locs[dest])
            else:
                heuristic = manhattan_distance((next_taxi_row, next_taxi_col), env.locs[passenger])
            
            total_cost = current_cost + 1 + heuristic
            heapq.heappush(priority_queue, (total_cost, next_state, path + [action], passenger_picked))

    return None, 0

def run_a_star(episodes):
    env = gym.make('Taxi-v3')

    rewards_per_episode = np.zeros(episodes)
    
    for episode in range(episodes):
        initial_state = env.reset()[0]  # Obtiene el estado inicial
        path, reward = a_star(env, initial_state)
        
        # # Muestra el camino encontrado para el modo humano (opcional)
        # if path:
        #     env = gym.make('Taxi-v3', render_mode='human')
        #     env.reset()
        #     env.unwrapped.s = initial_state
        #     for action in path:
        #         env.step(action)
        #         env.render()
        
        rewards_per_episode[episode] = reward

    env.close()
    print("Recompensa global", (TotalRecom/episodes))
    
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa')
    plt.title('Recompensas por episodio usando A* en Taxi')
    plt.savefig('taxi_a_star.png')

if __name__ == '__main__':
    start = time.time()
    run_a_star(1000)
    end = time.time()
    print("TIME: ",(end-start))
