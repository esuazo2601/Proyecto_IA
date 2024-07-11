import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Espacio de observación = 500
# 6 acciones posibles (arriba, abajo, izquierda, derecha, pickup, drop)
# Las recompensas y termino de están definidas en la documentación de gymnasium 
Q = np.zeros((500, 6))  

# Elegir una acción usando epsilon-greedy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(6))  # Exploración: elegir una acción aleatoria
    else:
        return np.argmax(Q[state,:])  # Explotación: elegir la mejor acción

def run(episodes):

    env = gym.make('Taxi-v3', render_mode=None)

    alpha = 0.9 # alpha or learning rate
    gamma = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1 # 1 = 100% random actions
    epsilon_decay = 0.001  # epsilon decay rate
    min_epsilon = 0.05  # Probabilidad de exploración mínima

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        # if episode == 800:
        #     env.close()
        #     env = gym.make('Taxi-v3', render_mode='human')  
        state = env.reset()[0]  # Obtener el estado del diccionario
        terminated = False      # True si el agente cae en una obstaculo
        truncated = False       # True cuando hay > 200 acciones
        total_reward = 0
        
        while(not terminated and not truncated):
            action = choose_action(state, epsilon)

            next_state, reward, terminated, truncated, _  = env.step(action)
            
            total_reward += reward

            next_action = np.max(Q[next_state,:])

            Q[state, action] += alpha * (reward + gamma * next_action - Q[state, action])
            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decaimiento de epsilon
        rewards_per_episode[episode] = total_reward

    env.close()

    # sum_rewards = np.zeros(episodes)
    # for t in range(episodes):
    #     sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    # plt.plot(sum_rewards)
    # plt.savefig('taxi.png')


if __name__ == '__main__':
    start = time.time()
    run(1000)
    end = time.time()
    print("TIME: ",(end-start))