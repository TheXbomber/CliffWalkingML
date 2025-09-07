import matplotlib.pyplot as plt

def plot(rewards_tabular=None, rewards_dqn=None, epsilon_history_tabular=None, epsilon_history_dqn=None, td_errors=None, losses=None):
    if rewards_tabular and epsilon_history_tabular and td_errors:
        # Plot rewards
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(rewards_tabular)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot epsilon
        plt.subplot(1,3,2)
        plt.plot(epsilon_history_tabular)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        # Plot errors
        plt.subplot(1,3,3)
        plt.plot(td_errors)
        plt.title("Avg TD Error per Episode")
        plt.xlabel("Episode")
        plt.ylabel("TD Error")

        plt.tight_layout()
        plt.savefig("plot_tabular.png")
        plt.close()
        print('💾 Plot saved to file "plot_tabular.png"')
        #plt.show()

    if rewards_dqn and epsilon_history_dqn and losses:
        # Plot rewards
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(rewards_dqn)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot epsilon
        plt.subplot(1,3,2)
        plt.plot(epsilon_history_dqn)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        # Plot loss
        plt.subplot(1,3,3)
        plt.plot(losses)
        plt.title("Average Loss per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.savefig("plot_dqn.png")
        print('💾 Plot saved to file "plot_dqn.png"')
        plt.close()
        #plt.show()