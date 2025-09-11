import json
import os

DEBUG = False
RANDOM_START = True
SLIPPERY = True
EVALUATION_EPISODES = 1000

# Tabular Q-learning parameters
TAB_ALPHA = 0.1
TAB_GAMMA = 0.99
TAB_EPSILON = 1
TAB_MIN_EPSILON = 0.01
TAB_EPSILON_DECAY = 0.995
TAB_EPISODES = 500

# DQN parameters
DQN_BATCH_SIZE = 64
DQN_GAMMA = 0.99
DQN_EPSILON = 1
DQN_MIN_EPSILON = 0.01
DQN_EPSILON_DECAY = 0.995
DQN_BUFFER_SIZE = 1000
DQN_EPISODES = 500
DQN_TARGET_UPDATE_FREQ = 10
DQN_GRAD_UPDATE_FREQ = 1
DQN_HIDDEN_LAYERS = 2
DQN_NODES_PER_LAYER = 32

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

def save_config(filename=None):
    filename = filename or CONFIG_FILE
    config = {k: globals()[k] for k in globals()
              if not k.startswith("__") and k.isupper()}
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Configuration saved to {filename}")

def load_config(filename=None):
    global DEBUG, RANDOM_START, SLIPPERY, EVALUATION_EPISODES
    global DQN_HIDDEN_LAYERS, DQN_NODES_PER_LAYER
    global TAB_ALPHA, TAB_GAMMA, TAB_EPSILON, TAB_MIN_EPSILON, TAB_EPSILON_DECAY, TAB_EPISODES
    global DQN_BATCH_SIZE, DQN_GAMMA, DQN_EPSILON, DQN_EPSILON_DECAY, DQN_EPISODES, DQN_TARGET_UPDATE_FREQ, DQN_GRAD_UPDATE_FREQ, DQN_BUFFER_SIZE, DQN_MIN_EPSILON

    filename = filename or CONFIG_FILE
    try:
        with open(filename, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Config file {filename} not found")
        return

    for k, v in config.items():
        if k in globals():
            globals()[k] = v

    print(f"✅ Configuration loaded from {filename}")

if os.path.exists(CONFIG_FILE):
    load_config(CONFIG_FILE)
else:
    print("⚠️ No config file found, using default parameters")