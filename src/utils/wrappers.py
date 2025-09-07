import gymnasium as gym
import random

class RandomStartWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)

        # Grid dimensions
        nrow, ncol = self.env.unwrapped.shape

        # Goal = bottom-right corner
        goal_state = nrow * ncol - 1

        # Cliff = bottom row except start (last row, all but first and last col)
        cliff = list(range((nrow - 1) * ncol + 1, nrow * ncol - 1))

        # Valid states = all except cliff and goal
        valid_states = [s for s in range(self.observation_space.n)
                        if s not in cliff and s != goal_state]

        # Pick random state
        state = random.choice(valid_states)
        self.env.unwrapped.s = state

        return state, info
