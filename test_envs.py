from pettingzoo.test import api_test, seed_test
from envs import love_letter_env_v1, love_letter_env_v2
from gym.utils.env_checker import check_env
from agents.random import RandomAgent

env = love_letter_env_v1.env()
api_test(env)
seed_test(love_letter_env_v1.env)
check_env(love_letter_env_v2.LoveLetterEnv(RandomAgent()), skip_render_check=True)