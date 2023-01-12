import gym
import numpy as np
from gym.spaces import Box

# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
class ToyPong(gym.Env):
    n_agent = 1

    def __init__(self, args):
        self.args = args
