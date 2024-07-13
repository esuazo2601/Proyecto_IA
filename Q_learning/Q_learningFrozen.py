import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import time
# Variable global para registrar la recompensa global
TotalRecom = 0
# Inicializar la tabla Q con ceros
Q = np.zeros((100*100, 4))

# Elegir una acción usando la estrategia epsilon-greedy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # Exploración: elegir una acción aleatoria
    else:
        return np.argmax(Q[state,:])  # Explotación: elegir la mejor acción

def run(episodes):
    global TotalRecom
    env = gym.make('FrozenLake-v1', desc = generate_random_map(size=100), is_slippery=True, render_mode=None)

    alpha = 0.9  # Tasa de aprendizaje
    gamma = 0.9  # Tasa de descuento
    epsilon = 1.0  # Valor inicial de epsilon para la estrategia epsilon-greedy
    epsilon_decay = 0.0001  # Tasa de decaimiento de epsilon
    min_epsilon = 0.1  # Probabilidad mínima de exploración

    rewards_per_episode = np.zeros(episodes)
    goal_state = 9999  # Estado objetivo para una cuadrícula de 100x100


    start = time.time()
    for episode in range(episodes):
        state = env.reset()[0]  # Obtener el estado inicial
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            action = choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Modificar la recompensa según las nuevas reglas
            if next_state == goal_state:
                reward = 1  # Llegó a la meta
                print(f'Episodio {episode} ha encontrado la meta.')
                terminated = True  # Terminar el episodio al llegar a la meta
            elif terminated:
                reward = -1  # Cayó en un agujero
            else:
                reward = -0.1  # Penalización por moverse

            # Penalizar si el agente se queda en el mismo estado
            if next_state == state:
                reward -= 1  # Penalización más alta por quedarse quieto

            total_reward += reward
            TotalRecom += reward

            next_action = np.max(Q[next_state,:])
            Q[state, action] += alpha * (reward + gamma * next_action - Q[state, action])
            state = next_state
        if min_epsilon < episode:
            epsilon = epsilon-epsilon_decay # Decaimiento de epsilon

        rewards_per_episode[episode] = total_reward

    env.close()

    end = time.time()
    print("TOTAL TIME FROZEN", (end-start))
    print("Recompensa global", (TotalRecom/episodes))

    # Calcular recompensas acumuladas para los últimos 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    # Graficar recompensas acumuladas
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas Acumuladas (Últimos 100 Episodios)')
    plt.title('Recompensas Acumuladas en FrozenLake')
    plt.savefig('frozen.png')

if __name__ == '__main__':
    run(20000)
