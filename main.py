import gymnasium as gym
import config
from agents.tabular import train_tabular_q
from agents.dqn import train_dqn
import sys
from utils.plot import plot
from utils.evaluation import evaluate_agent
from utils.wrappers import RandomStartWrapper

def configure_parameters():
    while True:
        print("\n⚙️ CURRENT PARAMETERS:")
        print(f"[1] Debug mode = {config.DEBUG}")
        print(f"[2] Random start = {config.RANDOM_START}")
        print("--- TABULAR Q-LEARNING ---")
        print(f"[3] Alpha = {config.TAB_ALPHA}")
        print(f"[4] Gamma = {config.TAB_GAMMA}")
        print(f"[5] Epsilon = {config.TAB_EPSILON}")
        print(f"[6] Minimum epsilon = {config.TAB_MIN_EPSILON}")
        print(f"[7] Epsilon decay = {config.TAB_EPSILON_DECAY}")
        print(f"[8] Episodes = {config.TAB_EPISODES}")
        print("--- DQN ---")
        print(f"[9] Number of hidden layers = {config.DQN_HIDDEN_LAYERS}")
        print(f"[10] Number of nodes per layer = {config.DQN_NODES_PER_LAYER}")
        print(f"[11] Batch size = {config.DQN_BATCH_SIZE}")
        print(f"[12] Gamma = {config.DQN_GAMMA}")
        print(f"[13] Epsilon = {config.DQN_EPSILON}")
        print(f"[14] Epsilon decay = {config.DQN_EPSILON_DECAY}")
        print(f"[15] Episodes = {config.DQN_EPISODES}")
        print(f"[16] Target update frequency = {config.DQN_TARGET_UPDATE_FREQ}")
        print(f"[17] Gradient update frequency = {config.DQN_GRAD_UPDATE_FREQ}")
        print("[100] 💾 Save Configuration")
        print("[200] 📂 Load Configuration")
        print("[0] 🔙 Go Back")

        try:
            choice = int(input("SELECT A PARAMETER TO CHANGE OR SAVE/LOAD CONFIG: "))
        except ValueError:
            continue

        if choice == 0:
            break

        if choice not in [100, 200]:
            value = input("Enter new value: ")
            try:
                if choice in [3, 4, 7, 12, 14]:  # float
                    value = float(value)
                elif choice in [5, 6, 8, 9, 10, 11, 13, 15, 16, 17]:  # integer
                    value = int(value)
                elif choice in [1, 2]:  # bool
                    value = value.lower() in ["true", "1", "yes", "y"]
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
            case 3:
                config.TAB_ALPHA = value
            case 4:
                config.TAB_GAMMA = value
            case 5:
                config.TAB_EPSILON = value
            case 6:
                config.TAB_MIN_EPSILON = value
            case 7:
                config.TAB_EPSILON_DECAY = value
            case 8:
                config.TAB_EPISODES = value
            case 9:
                config.DQN_HIDDEN_LAYERS = value
            case 10:
                config.DQN_NODES_PER_LAYER = value
            case 11:
                config.DQN_BATCH_SIZE = value
            case 12:
                config.DQN_GAMMA = value
            case 13:
                config.DQN_EPSILON = value
            case 14:
                config.DQN_EPSILON_DECAY = value
            case 15:
                config.DQN_EPISODES = value
            case 16:
                config.DQN_TARGET_UPDATE_FREQ = value
            case 17:
                config.DQN_GRAD_UPDATE_FREQ = value
            case _:
                print("⚠️ Invalid choice")
                continue

        print("✅ Parameter updated!")



# Set the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# Create the environment
env = gym.make("CliffWalking-v1", render_mode="ansi")

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
                    print("[2] 🧮 DQN")
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
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env, config.TAB_EPISODES)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env, config.DQN_EPISODES)
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
                    if config.DEBUG:
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
                    if config.DEBUG:
                        print(f"Choice: {choice}")

                    if choice not in [1, 2, 0]:
                        continue
                    else:
                        if choice == 1:
                            Q, rewards_tabular, epsilon_history_tabular, td_errors = train_tabular_q(env, config.TAB_EPISODES)
                            evaluate_agent(device, env, policy_net=None, Q=Q, tabular=True, episodes=1000)
                            plot(rewards_tabular=rewards_tabular, epsilon_history_tabular=epsilon_history_tabular, td_errors=td_errors)
                            break
                        elif choice == 2:
                            policy_net, rewards_dqn, epsilon_history_dqn, losses = train_dqn(device, env, config.DQN_EPISODES)
                            evaluate_agent(device, env, policy_net=policy_net, tabular=False, episodes=1000)
                            plot(rewards_dqn=rewards_dqn, epsilon_history_dqn=epsilon_history_dqn, losses=losses)
                            break
                        elif choice == 0:
                            break
                        else:
                            print("⚠️ ERROR")
                            sys.exit()
                
            case 4:
                configure_parameters()
                continue

            case 0:
                sys.exit()