import gymnasium as gym
import random

class RandomStartWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # do a normal reset first
        state, info = super().reset(**kwargs)

        nrow, ncol = self.env.unwrapped.shape
        goal_state = nrow * ncol - 1
        cliff = list(range((nrow - 1) * ncol + 1, nrow * ncol - 1))
        valid_states = [s for s in range(self.observation_space.n)
                        if s not in cliff and s != goal_state]

        # pick a random start
        random_state = random.choice(valid_states)
        self.env.unwrapped.s = random_state

        # return the randomized state (observation) and info
        return random_state, info
