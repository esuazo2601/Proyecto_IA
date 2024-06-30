import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

# Definir el entorno y los parámetros
matrix = [
    [0, 0, 0, 'H', 0],
    [0, 'H', 0, 'H', 0],
    [0, 0, 0, 0, 0],
    [0, 'H', 0, 'H', 0],
    [0, 0, 0, 0, 0]
]

# Parámetros del Q-learning
start = (0, 0)
end = (4, 4)
alpha = 0.9  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 1  # Probabilidad de exploración inicial
min_epsilon = 0.05  # Probabilidad de exploración mínima
epsilon_decay = 0.001  # Tasa de decaimiento de epsilon
episodes = 1000  # Número de episodios

# Inicializar la tabla Q
rows, cols = len(matrix), len(matrix[0])
Q = np.zeros((rows, cols, 4))  # 4 acciones posibles (arriba, abajo, izquierda, derecha)

# Definir las acciones posibles
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Derecha, Izquierda, Abajo, Arriba

# Recompensas de los episodio
rewards_per_episode = np.zeros(episodes)

# Definir la función de recompensa
def get_reward(matrix, state, next_state):
    if matrix[next_state[0]][next_state[1]] == 'H':
        return -100  # Penalización por obstáculo
    elif next_state == end:
        return 100  # Recompensa por llegar al objetivo
    else:
        return -1  # Penalización por movimiento

# Elegir una acción usando epsilon-greedy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # Exploración: elegir una acción aleatoria
    else:
        return np.argmax(Q[state[0], state[1]])  # Explotación: elegir la mejor acción

# Entrenar el agente con Q-learning
for episode in range(episodes):
    state = start
    total_reward = 0  # Recompensa acumulada en cada episodio
    while state != end:
        action = choose_action(state, epsilon)
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        
        if 0 <= next_state[0] < rows and 0 <= next_state[1] < cols:
            reward = get_reward(matrix, state, next_state)
            total_reward += reward
            next_action = np.argmax(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            state = next_state
        else:
            reward = -100  # Penalización por salir del entorno
            total_reward += reward
            Q[state[0], state[1], action] += alpha * (reward - Q[state[0], state[1], action])
            break
    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decaimiento de epsilon
    rewards_per_episode[episode] = total_reward

sum_rewards = np.zeros(episodes)
for t in range(episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake.png')    

# Encontrar el camino usando la tabla Q aprendida
def find_path(matrix, start, end):
    state = start
    path = [start]
    while state != end:
        action = np.argmax(Q[state[0], state[1]])
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        
        if 0 <= next_state[0] < rows and 0 <= next_state[1] < cols and matrix[next_state[0]][next_state[1]] != 'H':
            path.append(next_state)
            state = next_state
        else:
            break
    return path

def get_instructions(path: List[Tuple[int, int]]):
    steps = []
    for i in range(1, len(path)):
        # Obtener el punto actual y el punto anterior
        punto_anterior = path[i-1]
        punto_actual = path[i]
        
        # Comparar las coordenadas para determinar la dirección del movimiento
        if punto_actual[0] < punto_anterior[0]:
            steps.append(3)
        elif punto_actual[0] > punto_anterior[0]:
            steps.append(1)
        elif punto_actual[1] < punto_anterior[1]:
            steps.append(0)
        elif punto_actual[1] > punto_anterior[1]:
            steps.append(2)
    return steps

# Obtener el camino usando Q-learning
path = find_path(matrix, start, end)

# Verificar si se ha encontrado un camino válido
if path[-1] != end:
    print("No se encontró un camino válido.")
else:
    # Obtener las instrucciones a partir del camino
    instructions = get_instructions(path)
    print("Path:", path)
    print("Instructions:", instructions)
