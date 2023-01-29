# Love Letter Reinforcement Learning Environment

This repository contains an [OpenAI Gym](https://github.com/openai/gym) and [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) environment for Reinforcement Learning. There is also a simple tool for training `stable_baselines3` agents on the Gym version. 

Requirements:
- `gym`
- `gymnasium` (for PettingZoo version)
- `pettingzoo`
- `numpy`
- `stable_baselines3`

## Simple tools

`train.py` file shows an example training of a DQN agent on the Gym environment. Changing agent/policy/parameters can be done only by modifying the 

`play.py` file offers the possibility to play a trained agent. It also renders the state before the agent's turn so cheating is possible if beating your PC in a card game is all you ever wanted.

`test_envs.py` tests the environments' consistency with the respective APIs.
