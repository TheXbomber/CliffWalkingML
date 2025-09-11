import random, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import numpy as np
import torch.optim as optim
from tqdm import trange
import config

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.embedding = nn.Embedding(state_size, 16)

        layers = []
        input_size = 16

        for _ in range(config.DQN_HIDDEN_LAYERS):
            layers.append(nn.Linear(input_size, config.DQN_NODES_PER_LAYER))
            layers.append(nn.ReLU())
            input_size = config.DQN_NODES_PER_LAYER

        layers.append(nn.Linear(input_size, action_size))
        self.net = nn.Sequential(*layers)

        # self.net = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_size)
        # )

    def forward(self, x):
        x = self.embedding(x)
        return self.net(x)

def train_dqn(device, env):
    episodes=config.DQN_EPISODES
    batch_size=config.DQN_BATCH_SIZE
    gamma=config.DQN_GAMMA
    epsilon=config.DQN_EPSILON
    min_epsilon=config.DQN_MIN_EPSILON
    epsilon_decay=config.DQN_EPSILON_DECAY
    target_update_freq=config.DQN_TARGET_UPDATE_FREQ
    grad_update_freq=config.DQN_GRAD_UPDATE_FREQ

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=1000)
    step_count = 0

    rewards, epsilon_history, losses = [], [], []

    for ep in trange(episodes, desc="🏋️ Training DQN"):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_losses = []

        while not done:
            step_count += 1
            s_tensor = torch.tensor([state], device=device)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(s_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if step_count % grad_update_freq == 0 and len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.tensor(states, device=device)
                next_states = torch.tensor(next_states, device=device)
                actions = torch.tensor(actions, device=device)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.bool, device=device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    targets = rewards_batch + gamma * next_q * (~dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_losses.append(loss.item())

        rewards.append(ep_reward)
        epsilon_history.append(epsilon)
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        losses.append(avg_loss)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print("✅ Training complete")

    return policy_net, rewards, epsilon_history, losses