import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Inicializar la tabla Q con ceros
Q = np.zeros((1000*1000, 4))

# Elegir una acción usando la estrategia epsilon-greedy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(4))  # Exploración: elegir una acción aleatoria
    else:
        return np.argmax(Q[state,:])  # Explotación: elegir la mejor acción

def run(episodes):

    env = gym.make('FrozenLake-v1', desc = generate_random_map(size=1000), is_slippery=False, render_mode=None)

    alpha = 0.9  # Tasa de aprendizaje
    gamma = 0.9  # Tasa de descuento
    epsilon = 1.0  # Valor inicial de epsilon para la estrategia epsilon-greedy
    epsilon_decay = 0.001  # Tasa de decaimiento de epsilon
    min_epsilon = 0.05  # Probabilidad mínima de exploración

    rewards_per_episode = np.zeros(episodes)
    goal_state = 63  # Estado objetivo para una cuadrícula de 8x8

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
                terminated = True  # Terminar el episodio al llegar a la meta
            elif terminated:
                reward = -1  # Cayó en un agujero
            else:
                reward = 0  # Se movió sobre terreno congelado

            total_reward += reward

            next_action = np.max(Q[next_state,:])
            Q[state, action] += alpha * (reward + gamma * next_action - Q[state, action])
            state = next_state
        if min_epsilon < episode:
            epsilon = epsilon-epsilon_decay # Decaimiento de epsilon

        rewards_per_episode[episode] = total_reward

    env.close()

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
