import gymnasium as gym
from DFS import dfs, get_instructions
import time as t
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', desc = generate_random_map(100), is_slippery=True, render_mode=None)
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

# Número de episodios
num_episodes = 100
rewards_per_episode = []

for episode in range(num_episodes):
    done = False
    stats, info = env.reset()
    score = 0

    while not done:
        #env.render()
        for action in steps:
            #t.sleep(0.5)
            n_state, reward, done, truncated, info = env.step(action)
            score += reward
            if done:
                break

    rewards_per_episode.append(score)

env.close()

# Graficar las recompensas acumuladas por episodio
plt.plot(rewards_per_episode)
plt.xlabel('Episodios')
plt.ylabel('Recompensas Acumuladas')
plt.title('Recompensas Acumuladas en FrozenLake')
plt.savefig('frozen_dfs_slippery.png')

if path:
    print("Camino encontrado:", path)
else:
    print("No se encontró camino.")
