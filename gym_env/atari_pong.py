import collections
import math

import numpy as np
import gym

# The default ALE environment returns only the picture of the game screen,
# however the extracted policy requires a state vector (x, y, y', y'', ball_x, ball_y)
# that is extracted from the image, but it is not clear how.
# We therefore extract everything we can from the ALE ram and estimate the missing quantities.
# See https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py#L178
class AtariPong(gym.Env):
    def __init__(self, args):
        self.env = gym.make("Pong-v4", obs_type="ram", render_mode="human")

        self._last_states = collections.deque(maxlen=10)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self):
        self._last_states.clear()
        ram = self.env.reset()
        return self._ram_to_obs(ram)

    def _ram_to_obs(self, ram):
        player_paddle_y = ram[51].astype(np.float32)  # Y coordinate of your paddle
        ball_x = ram[49].astype(np.float32)  # X coordinate of ball
        ball_y = ram[54].astype(np.float32)  # Y coordinate of ball

        speed = self._estimate_speed()[-1]
        acc = self._estimate_acceleration()[-1]
        jerk = self._estimate_jerk()[-1]
        ball_speed = self._estimate_ball_velocity()[-1]

        state = [player_paddle_y, speed, acc, jerk, ball_x, ball_y, ball_speed[0], ball_speed[1]]

        return state

    def step(self, action):
        ram, reward, done, info = self.env.step(action)

        next_state = self._ram_to_obs(ram)
        self._last_states.append(next_state)

        return next_state, reward, done, info

    def render(self, mode="human"):
        self.env.render(mode="rgb_array")

    def _estimate_speed(self):
        if len(self._last_states) < 2:
            return [0]

        last = None
        speeds = []
        for state in self._last_states:
            paddle_y = state[0]
            if last is None:
                last = paddle_y
                continue

            speeds.append(paddle_y - last)
            last = paddle_y

        return speeds

    def _estimate_acceleration(self):
        speeds = self._estimate_speed()
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

    def _estimate_jerk(self):
        accs = self._estimate_acceleration()
        if len(accs) < 2:
            return [0]

        return [accs[-1] - accs[-2]]

    def _estimate_ball_velocity(self):
        if len(self._last_states) < 2:
            return [[0, 0]]

        last = None
        speeds = []
        for state in self._last_states:
            ball_x = state[4]
            ball_y = state[5]
            if last is None:
                last = [ball_x, ball_y]
                continue

            speeds.append([ball_x - last[0], ball_y - last[1]])
            last = [ball_x, ball_y]

        return speeds
