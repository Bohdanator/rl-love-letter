import gymnasium as gym

from stable_baselines3 import DQN
from envs.love_letter_env_v2 import LoveLetterEnv
from agents.random import RandomAgent
from agents.smart_random import SmartRandomAgent

env = LoveLetterEnv(SmartRandomAgent())

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000, log_interval=4000)
model.save("dqn_loveletter_smarter")