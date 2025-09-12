import gymnasium as gym
import config
from agents.tabular import train_tabular_q
from agents.dqn import train_dqn
import sys
import os
from utils.plot import plot
from utils.evaluation import evaluate_agent
from utils.wrappers import RandomStartWrapper

def rebuild_env():
    env = gym.make("CliffWalking-v1", render_mode="ansi", is_slippery=config.SLIPPERY)
    if config.RANDOM_START:
        env = RandomStartWrapper(env)
    return env

def configure_parameters(env):
    while True:
        print("\n⚙️ CURRENT PARAMETERS:")
        print(f"[1] Debug mode = {config.DEBUG}")
        print(f"[2] Random start = {config.RANDOM_START}")
        print(f"[3] Slippery = {config.SLIPPERY}")
        print("--- TABULAR Q-LEARNING ---")
        print(f"[4] Alpha = {config.TAB_ALPHA}")
        print(f"[5] Gamma = {config.TAB_GAMMA}")
        print(f"[6] Epsilon = {config.TAB_EPSILON}")
        print(f"[7] Minimum epsilon = {config.TAB_MIN_EPSILON}")
        print(f"[8] Epsilon decay = {config.TAB_EPSILON_DECAY}")
        print(f"[9] Episodes = {config.TAB_EPISODES}")
        print("--- DQN ---")
        print(f"[10] Number of hidden layers = {config.DQN_HIDDEN_LAYERS}")
        print(f"[11] Number of nodes per layer = {config.DQN_NODES_PER_LAYER}")
        print(f"[12] Batch size = {config.DQN_BATCH_SIZE}")
        print(f"[13] Gamma = {config.DQN_GAMMA}")
        print(f"[14] Epsilon = {config.DQN_EPSILON}")
        print(f"[15] Minimum epsilon = {config.DQN_MIN_EPSILON}")
        print(f"[16] Epsilon decay = {config.DQN_EPSILON_DECAY}")
        print(f"[17] Buffer size = {config.DQN_BUFFER_SIZE}")
        print(f"[18] Episodes = {config.DQN_EPISODES}")
        print(f"[19] Target update frequency = {config.DQN_TARGET_UPDATE_FREQ}")
        print(f"[20] Gradient update frequency = {config.DQN_GRAD_UPDATE_FREQ}")
        print("--- EVALUATION ---")
        print(f"[21] Episodes = {config.EVALUATION_EPISODES}")
        print("\n⚙️ MANAGE CONFIG:")
        print("[100] 💾 Save Configuration")
        print("[200] 📂 Load Configuration\n")
        print("[0] 🔙 Go Back")

        try:
            choice = int(input("CHANGE A PARAMETER OR MANAGE CONFIG: "))
        except ValueError:
            continue

        if choice == 0:
            return env

        if choice not in [100, 200]:
            value = input("Enter new value: ")
            try:
                if choice in [4, 5, 8, 13, 15, 16]:  # float
                    value = float(value)
                elif choice in [6, 7, 9, 10, 11, 12, 14, 17, 18, 19, 20, 21]:  # integer
                    value = int(value)
                elif choice in [1, 2, 3]:  # bool
                    if value.lower() in ["true", "1", "yes", "y"]:
                        value = True
                    elif value.lower() in ["false", "0", "no", "n"]:
                        value = False
                    else:
                        print("⚠️ Invalid boolean value! Use true/false, 1/0, yes/no, y/n.")
                        continue
            except ValueError:
                print("⚠️ Invalid value type!")
                continue

        match choice:
            case 100:
                config.save_config("config.json")
                continue
            case 200:
                config.load_config("config.json")
                continue
            case 1:
                config.DEBUG = value
            case 2:
                config.RANDOM_START = value
                env.close()
                env = rebuild_env()
            case 3:
                config.SLIPPERY = value
                env.close()
                env = rebuild_env()
            case 4:
                config.TAB_ALPHA = value
            case 5:
                config.TAB_GAMMA = value
            case 6:
                config.TAB_EPSILON = value
            case 7:
                config.TAB_MIN_EPSILON = value
            case 8:
                config.TAB_EPSILON_DECAY = value
            case 9:
                config.TAB_EPISODES = value
            case 10:
                config.DQN_HIDDEN_LAYERS = value
            case 11:
                config.DQN_NODES_PER_LAYER = value
            case 12:
                config.DQN_BATCH_SIZE = value
            case 13:
                config.DQN_GAMMA = value
            case 14:
                config.DQN_EPSILON = value
            case 15:
                config.DQN_MIN_EPSILON = value
            case 16:
                config.DQN_EPSILON_DECAY = value
            case 17:
                config.DQN_BUFFER_SIZE = value
            case 18:
                config.DQN_EPISODES = value
            case 19:
                config.DQN_TARGET_UPDATE_FREQ = value
            case 20:
                config.DQN_GRAD_UPDATE_FREQ = value
            case 21:
                config.EVALUATION_EPISODES = value
            case _:
                print("⚠️ Invalid choice")
                continue

        print("✅ Parameter updated!")



# Set the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

if os.path.exists(config.CONFIG_FILE):
    config.load_config(config.CONFIG_FILE)
else:
    print("⚠️ No config file found, using default parameters")

# Create the environment
env = gym.make("CliffWalking-v1", render_mode="ansi", is_slippery=config.SLIPPERY)

if config.RANDOM_START:
    env = RandomStartWrapper(env)

n_states = env.observation_space.n
n_actions = env.action_space.n

while True:
    print("SELECT AN ACTION:")
    print("[1] 🏋️ Train")
    print("[2] 🧪 Evaluate")
    print("[3] ⚡ Train & Evaluate")
    print("[4] ⚙️ Configure Parameters")
    print("[0] 🚪 Exit")

    try:
        choice = int(input())
    except ValueError:
        continue
    if config.DEBUG:
        print(f"Choice: {choice}")

    if choice not in [1, 2, 3, 4, 0]:
        continue
    else:
        match choice:
            case 1:
                while True:
                    print("🤖 SELECT AN AGENT:")
                    print("[1] 📊 Tabular Q-learning")
                    print("[2] 🧮 DQN\n")
                    print("[0] 🔙 Go Back")
                    
                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if config.DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env)
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
                    print("[2] 🧮 DQN\n")
                    print("[0] 🔙 Go Back")

                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if config.DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            try:  
                                evaluate_agent(device, env, policy_net=None, Q=Q, tabular=True)
                            except NameError:
                                print("⚠️ You need to train tabular Q-learning before evaluating it!")
                                break
                            break
                        elif choice == 2:
                            try:
                                evaluate_agent(device, env, policy_net=policy_net, tabular=False)
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
                    print("[2] 🧮 DQN\n")
                    print("[0] 🔙 Go Back")

                    try:
                        choice = int(input())
                    except ValueError:
                        continue
                    if config.DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            evaluate_agent(device, env, policy_net=None, Q=Q, tabular=True)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env)
                            plot(rewards_dqn=rewards_dqn, epsilon_history_dqn=epsilon_history_dqn, losses=losses)
                            evaluate_agent(device, env, policy_net=policy_net, tabular=False)
                            break
                        elif choice == 0:
                            break
                        else:
                            print("⚠️ ERROR")
                            sys.exit()
                
            case 4:
                env = configure_parameters(env)
                continue

            case 0:
                sys.exit()