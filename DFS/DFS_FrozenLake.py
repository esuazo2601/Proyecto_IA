import gymnasium as gym
from DFS import dfs, get_instructions
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
# Recompensa global de los episodios
TotalRecom = 0 

def run_dfs_on_frozenlake(episodes):
    global TotalRecom
    rewards_per_episode = []
    start_time = time.time()

    for episode in range(episodes):
        # Crear el entorno de FrozenLake
        env = gym.make('FrozenLake-v1', desc=generate_random_map(100), is_slippery=True, render_mode=None)
        env_desc = env.get_wrapper_attr('desc')

        # Extraer la matriz del nivel
        level_matrix = [[cell.decode() for cell in row] for row in env_desc]

        # Encontrar las coordenadas de inicio y meta
        start = None
        end = None
        for row_idx, row in enumerate(level_matrix):
            for col_idx, cell in enumerate(row):
                if cell == 'S':
                    start = (row_idx, col_idx)
                elif cell == 'G':
                    end = (row_idx, col_idx)

        # Ejecución de DFS
        path = dfs(level_matrix, start, end)
        steps = get_instructions(path)

        done = False
        state, info = env.reset()
        score = 0

        while not done:
            for action in steps:
                state, reward, done, truncated, info = env.step(action)
                score += reward
                if reward == 0 and not done:
                    reward = -1
                score += reward
                TotalRecom += reward

                if done:
                    break

        rewards_per_episode.append(score)
        env.close()

    end_time = time.time()

    # Mostrar resultados
    print("TOTAL TIME FROZEN:", (end_time - start_time))
    print("Recompensa global", (TotalRecom/episodes))
    #print("Recompensas por episodio:", rewards_per_episode)

    # Graficar recompensas
    # plt.plot(rewards_per_episode)
    # plt.xlabel('Episodios')
    # plt.ylabel('Recompensas Acumuladas')
    # plt.title('Recompensas Acumuladas en FrozenLake')
    # plt.savefig('frozen_dfs_slippery.png')
    # plt.show()

def main():
    episodes = 100  # Número de iteraciones a ejecutar
    run_dfs_on_frozenlake(episodes)

if __name__ == '__main__':
    main()
