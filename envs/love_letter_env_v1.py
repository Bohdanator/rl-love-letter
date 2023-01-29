from pettingzoo import AECEnv 
from pettingzoo.utils import agent_selector, wrappers

import gymnasium
from gymnasium.spaces import Box, Discrete, Dict
from gymnasium.utils import seeding

import numpy as np

from game.game import Game

def env(render_mode=None):
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):

    metadata = {"render_modes": ["human"], "name": "ll_v1", "is_parallelizable": True}

    def __init__(self, render_mode=None):
        super().__init__()

        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # actions defined in game.py
        self._action_spaces = {agent: Discrete(15) for agent in self.possible_agents}
        self._observation_spaces = {
            name: Dict(
                {"observation": Box(0, 1, (24, ), dtype=np.float64), "action_mask": Box(0, 1, (15, ), dtype=np.int8)}
            ) for name in self.agents
        }
        self.render_mode = render_mode
        
        self.game = Game()
        self.np_random, seed = seeding.np_random()
    
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        if not self.game.running:
            print("Game over")
            return
        print(self.game)
    
    def seed(self, seed=187):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def observe(self, agent):
        observation = self.game.observe(self.agent_name_mapping[agent])
        action_mask = np.array(self.game.generate_possible_actions()) if agent == self.agent_selection else np.zeros(15)
        return {"observation": observation, "action_mask": action_mask}
    
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.seed(seed)
        self.game.reset(self.np_random)

        if self.render_mode == "human":
            self.render()
    
    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            return self._was_dead_step(action)

        game_action = self.game.convert_env_action(action=action)
        assert self.game.action_possible(game_action)
        
        self.game.action(game_action)
        if not self.game.running:
            results = self.game.evaluate()
            for i, agent in enumerate(self.agents):
                self.rewards[agent] = 1 if results[i] else -1
                self.terminations[agent] = True

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()
        