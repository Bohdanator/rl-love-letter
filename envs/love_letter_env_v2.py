import gym
from gym import spaces
from gym.utils import seeding 

import numpy as np

from game.game import Game

class LoveLetterEnv(gym.Env):

    metadata = {"render_modes": ["human"], "name": "ll_v1", "is_parallelizable": True}

    def __init__(self, other_agent, first=1, render_mode=None, seed=187):
        # actions defined in game.py
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24, ), dtype=np.float64)
        self.reward_range = (-1,1)
        self.render_mode = render_mode
        
        self.other_agent = other_agent
        self.game = Game()
        self.first = 1 - first
        self.np_random, seed = seeding.np_random()
        self.game.reset(self.np_random)
    
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        self.game.print()
    
    def _seed(self, seed=187):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _other_step(self):
        if self.render_mode == "human":
            self.render()
        observation = self.game.observe()
        action_mask = self.game.generate_possible_actions()
        action = self.other_agent.move(observation, action_mask)
        self.game.action(self.game.convert_env_action(action))
    
    def reset(self, seed=None, options=None):
        self._seed(seed)
        self.game.reset(self.np_random)

        if self.first:
            self._other_step()

            if not self.game.running:
                results = self.game.evaluate()
                observation = self.game.observe()
                if self.render_mode == "human":
                    self.render()
                return observation, (1 if results[self.first] else -1), True, {}

        if self.render_mode == "human":
            self.render()
        observation = self.game.observe()
        return observation
    
    def step(self, action):
        game_action = self.game.convert_env_action(action=action)
        if not self.game.action_possible(game_action):
            self.game.kill(self.game.player)
            observation = self.game.observe()
            return observation, -1, True, {}
        
        self.game.action(game_action)
        if not self.game.running:
            results = self.game.evaluate()
            observation = self.game.observe()
            return observation, (1 if results[self.first] else -1), True, {}
        
        self._other_step()
        if not self.game.running:
            results = self.game.evaluate()
            observation = self.game.observe()
            return observation, (1 if results[self.first] else -1), True, {}
        
        observation = self.game.observe()
        return observation, 0, False, {}
        