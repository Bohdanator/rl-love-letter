import numpy as np

class RandomAgent:
    def move(self, observation, action_mask):
        choices = []
        for i, m in enumerate(action_mask):
            if m:
                choices.append(i)
        return np.random.choice(choices)