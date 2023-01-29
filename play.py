from stable_baselines3 import DQN 
from envs.love_letter_env_v2 import LoveLetterEnv
from agents.input import InputAgent

env = LoveLetterEnv(InputAgent(), render_mode="human")
agent = "dqn_loveletter_smarter"

model = DQN.load(agent)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()