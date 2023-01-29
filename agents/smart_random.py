import numpy as np

class SmartRandomAgent:
    def move(self, observation, action_mask):
        choices = []
        for i, m in enumerate(action_mask):
            if m:
                if i < 14:
                    choices.append(i)
        return np.random.choice(choices)