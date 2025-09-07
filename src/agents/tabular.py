import numpy as np
import random
from tqdm import trange
from config import DEBUG

def train_tabular_q(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards = []
    epsilon_history = []
    td_errors = []

    for ep in trange(episodes, desc="🏋️ Training Tabular Q-learning"):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_td_errors = []

        while not done:
            if DEBUG:
              print(env.render())

            if np.random.rand() < epsilon:
              action = env.action_space.sample()
            else:
              action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            old_value = Q[state, action]
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (not done)
            td_error = td_target - old_value
            Q[state, action] += alpha * td_error

            ep_td_errors.append(abs(td_error))

            state = next_state

            #print(f"State: {state}, Action: {action}, Reward: {reward}, Next state: {next_state}\n---\n")

        rewards.append(ep_reward)
        epsilon_history.append(epsilon)
        td_errors.append(np.mean(ep_td_errors))

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("✅ Training complete")

    return Q, rewards, epsilon_history, td_errors
