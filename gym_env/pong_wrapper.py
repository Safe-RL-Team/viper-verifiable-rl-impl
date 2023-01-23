import collections

import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv

"""
While the oracle trains on pictures provided by the environment,
the extracted policy requires a state vector (x, y, y', y'', ball_x, ball_y)
that is "extracted from the image", but it is not clear how.
We therefore extract everything we can from the ALE ram and estimate the missing quantities.
See https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py#L178
"""


def estimate_speed(last_states):
    if len(last_states) < 2:
        return [0]

    last = None
    speeds = []
    for state in last_states:
        paddle_y = state[0]
        if last is None:
            last = paddle_y
            continue

        speeds.append(paddle_y - last)
        last = paddle_y

    return speeds


def estimate_acceleration(speeds):
    if len(speeds) < 2:
        return [0]

    last = None
    accs = []
    for speed in speeds:
        if last is None:
            last = speed
            continue

        accs.append(speed - last)
        last = speed

    return accs


def estimate_jerk(accs):
    if len(accs) < 2:
        return [0]

    return [accs[-1] - accs[-2]]


def estimate_ball_velocity(last_states):
    if len(last_states) < 2:
        return [[0, 0]]

    last = None
    speeds = []
    for state in last_states:
        ball_x = state[4]
        ball_y = state[5]
        if last is None:
            last = [ball_x, ball_y]
            continue

        speeds.append([ball_x - last[0], ball_y - last[1]])
        last = [ball_x, ball_y]

    return speeds


def ram_to_obs(last_states, ram):
    player_paddle_y = ram[51].astype(np.float32)  # Y coordinate of your paddle
    ball_x = ram[49].astype(np.float32)  # X coordinate of ball
    ball_y = ram[54].astype(np.float32)  # Y coordinate of ball

    speeds = estimate_speed(last_states)
    accs = estimate_acceleration(speeds)
    jerk = estimate_jerk(accs)
    ball_speed = estimate_ball_velocity(last_states)[-1]

    state = [player_paddle_y, speeds[-1], accs[-1], jerk[0], ball_x, ball_y, ball_speed[0], ball_speed[1]]
    return np.array(state, dtype=np.float32)


class PongWrapper(gym.Wrapper):
    def __init__(self, env: VecEnv, return_extracted_obs=False):
        super().__init__(env)
        # List of last states per env
        self._obs_shape = (self.env.num_envs, 8)
        self.extracted_obs_hist = collections.deque(maxlen=10)
        self.return_extracted_obs = return_extracted_obs

    @property
    def observation_space(self) -> spaces.Space:
        if self.return_extracted_obs:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.num_envs, 8), dtype=np.float32)
        return self.env.obs_space

    def get_ram(self):
        return [env.ale.getRAM() for env in self.env.unwrapped.envs]

    def reset(self, **kwargs):
        self.extracted_obs_hist.clear()
        obs = self.env.reset(**kwargs)
        self.update_extracted_state()
        if self.return_extracted_obs:
            return self.get_extracted_obs()
        return obs

    def update_extracted_state(self):
        extracted_states = np.zeros(self._obs_shape, dtype=np.float32)
        for i, env in enumerate(self.env.unwrapped.envs):
            ram = env.ale.getRAM()
            last_env_states = [state[i] for state in self.extracted_obs_hist]
            extracted_states[i] = ram_to_obs(last_env_states, ram)

        self.extracted_obs_hist.append(extracted_states)

    def step(self, action):
        obs_img, reward, done, info = self.env.step(action)
        self.update_extracted_state()
        if self.return_extracted_obs:
            return self.get_extracted_obs(), reward, done, info
        return obs_img, reward, done, info

    def get_extracted_obs(self):
        return self.extracted_obs_hist[-1]
