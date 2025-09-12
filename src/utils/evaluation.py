from tqdm import trange
from collections import deque
import numpy as np
import random
import torch
import config

MAX_EVAL_STEPS = 500
MAX_CONSECUTIVE_REPEAT = 10  # episode ends if agent repeats a state this many times in a row

def evaluate_agent(device, env, policy_net=None, Q=None, tabular=True, random_start=False):
    episodes = config.EVALUATION_EPISODES

    successes, falls, total_rewards, steps_list = 0, 0, [], []

    agent = "DQN" if not tabular else "Tabular Q-learning"
    if not tabular:
        policy_net.eval()

    for ep in trange(episodes, desc="🧪 Evaluating " + agent):
        state, _ = env.reset()
        done, ep_reward, steps = False, 0, 0

        last_state = None
        repeat_count = 0

        while not done and steps < MAX_EVAL_STEPS:
            if config.RENDER:
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

            if last_state == next_state:
                repeat_count += 1
            else:
                repeat_count = 0
            last_state = next_state

            # Trigger early stop if stuck in a repeated state
            if repeat_count >= MAX_CONSECUTIVE_REPEAT:
                if config.DEBUG:
                    print(f"🛑 Stuck in state {next_state} for {MAX_CONSECUTIVE_REPEAT} steps. Ending episode early.")
                break

            state = next_state

            if reward == -100:
                falls += 1
            if done and next_state == (env.observation_space.n - 1):
                successes += 1

        total_rewards.append(ep_reward)
        steps_list.append(steps)

        if steps >= MAX_EVAL_STEPS and config.DEBUG:
            print("⏱️ Max steps reached, ending episode.")

    if not tabular:
        policy_net.train()

    print(f"✅ Evaluation Results (over {episodes} episodes):")
    print(f"  Success Rate: {successes}/{episodes} ({successes/episodes*100:.1f}%)")
    print(f"  Cliff Falls: {falls}")
    print(f"  Avg Total Reward: {np.mean(total_rewards):.2f}")
    print(f"  Avg Steps per Episode: {np.mean(steps_list):.2f}")