from tqdm import trange
from collections import deque
import numpy as np
import random
import torch
import config

def evaluate_agent(device, env, policy_net=None, Q=None, tabular=True,  random_start=False):
    episodes=config.EVALUATION_EPISODES
    loop_check_steps=20

    successes, falls, total_rewards, steps_list = 0, 0, [], []

    agent = "DQN" if not tabular else "Tabular Q-learning" 

    if not tabular:
        policy_net.eval()

    for _ in trange(episodes, desc="🧪 Evaluating " + agent):

        state, _ = env.reset()

        done, ep_reward, steps = False, 0, 0
        state_history = deque(maxlen=loop_check_steps)

        while not done:
            if config.DEBUG:
                print(env.render())

            # Select action
            if tabular:
                action = np.argmax(Q[state])
            else:
                state_tensor = torch.tensor([state], device=device)
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

            if config.DEBUG:
                print(f"State: {state}, Action: {action}, Reward: {reward}, Next state: {next_state}\n---\n")

            # Loop detection
            state_history.append(next_state)
            if len(state_history) == loop_check_steps:
              unique_states = len(set(state_history))
              if config.DEBUG and (unique_states < loop_check_steps // 3):  # 1/3 of buffer are unique
                  print("🛑 Loop detected! Ending episode early.")
                  break

            state = next_state

            if reward == -100:
                falls += 1
            if done and next_state == (env.observation_space.n - 1):
                successes += 1

        total_rewards.append(ep_reward)
        steps_list.append(steps)

    if not tabular:
        policy_net.train()

    print(f"✅ Evaluation Results (over {episodes} episodes):")
    print(f"  Success Rate: {successes}/{episodes} ({successes/episodes*100:.1f}%)")
    print(f"  Cliff Falls: {falls}")
    print(f"  Avg Total Reward: {np.mean(total_rewards):.2f}")
    print(f"  Avg Steps per Episode: {np.mean(steps_list):.2f}")