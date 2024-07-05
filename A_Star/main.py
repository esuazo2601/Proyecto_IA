import gymnasium as gym
from A_star import A_star,get_instructions
import time as t
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

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


# Ejecución de A Star
path = A_star(level_matrix, start, end)
steps = get_instructions(path)
#print(steps)

done=False
stats,info = env.reset()
score = 0

rewards = []
episodes = 100
for episode in range(episodes):
    while not done:
        #env.render()
        for action in steps:
            #t.sleep(0.5)
            n_state,reward,done,truncated,info = env.step(action)
            score += reward
    rewards.append(score)
    env.close()

print(rewards)
plt.plot(rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensas Acumuladas')
plt.title('Recompensas Acumuladas en FrozenLake')
plt.savefig('frozen_ASTAR_slippery.png')
plt.show()


if path:
    print("Camino encontrado A STAR:", path)
else:
    print("No se encontró camino.")
