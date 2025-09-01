import gymnasium as gym
from config import *
from agents.tabular import train_tabular_q
from agents.dqn import train_dqn
from utils.plot import plot
from utils.evaluation import evaluate_agent
import sys

# Set the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# Create the environment
env = gym.make("CliffWalking-v1", render_mode="ansi")
n_states = env.observation_space.n
n_actions = env.action_space.n

while True:
    print("SELECT AN ACTION:")
    print("[1] 🏋️ Train")
    print("[2] 🧪 Evaluate")
    print("[3] ⚡ Train & Evaluate")
    print("[0] 🚪 Exit")

    try:
        choice = int(input())
    except ValueError:
        continue
    if DEBUG:
        print(f"Choice: {choice}")

    if choice not in [1, 2, 3, 0]:
        continue
    else:
        match choice:
            case 1:
                while True:
                    print("🤖 SELECT AN AGENT:")
                    print("[1] 📊 Tabular Q-learning")
                    print("[2] 🧮 DQN")
                    print("[0] 🔙 Go Back")
                    
                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env, TAB_EPISODES)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env, DQN_EPISODES)
                            plot(rewards_dqn=rewards_dqn, epsilon_history_dqn=epsilon_history_dqn, losses=losses)
                            break
                        elif choice == 0:
                            break
                        else:
                            print("⚠️ ERROR")
                            sys.exit()

            case 2:
                while True:
                    print("🤖 SELECT AN AGENT:")
                    print("[1] 📊 Tabular Q-learning")
                    print("[2] 🧮 DQN")
                    print("[0] 🔙 Go Back")

                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            try:  
                                evaluate_agent(device, env, policy_net=None, Q=Q, tabular=True, episodes=1000)
                            except NameError:
                                print("⚠️ You need to train tabular Q-learning before evaluating it!")
                                break
                            break
                        elif choice == 2:
                            try:
                                evaluate_agent(device, env, policy_net=policy_net, tabular=False, episodes=1000)
                            except NameError:
                                print("⚠️ You need to train DQN before evaluating it!")
                                break
                            break
                        elif choice == 0:
                            break
                        else:
                            print("⚠️ ERROR")
                            sys.exit()
                
            case 3:
                while True:
                    print("🤖 SELECT AN AGENT:")
                    print("[1] 📊 Tabular Q-learning")
                    print("[2] 🧮 DQN")
                    print("[0] 🔙 Go Back")

                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env, TAB_EPISODES)
                            evaluate_agent(device, env, policy_net=None, Q=Q, tabular=True, episodes=1000)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env, DQN_EPISODES)
                            evaluate_agent(device, env, policy_net=policy_net, tabular=False, episodes=1000)
                            plot(rewards_dqn=rewards_dqn, epsilon_history_dqn=epsilon_history_dqn, losses=losses)
                            break
                        elif choice == 0:
                            break
                        else:
                            print("⚠️ ERROR")
                            sys.exit()

            case 0:
                sys.exit()