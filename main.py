import gymnasium as gym
from DFS import dfs

env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human")
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

if path:
    print("Camino encontrado:", path)
else:
    print("No se encontró camino.")

env.close()