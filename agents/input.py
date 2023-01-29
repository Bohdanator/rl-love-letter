import numpy as np

class InputAgent:
    def move(self, observation, action_mask):
        print(observation[-8:])
        choices = []
        for i, m in enumerate(action_mask):
            if m:
                choices.append(i)
        print(choices)
        return int(input())