import gym
import numpy as np
from gym import register
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

register(
    id='ToyPong-v0',
    entry_point='gym_env.toy_pong:ToyPong',
    kwargs={'args': None}
)

register(
    id='WrappedPong-v0',
    entry_point='gym_env.atari_pong:AtariPong',
    kwargs={'args': None}
)


def make_env(args):
    if args.env_name == "PongNoFrameskip-v4":
        env = make_atari_env(args.env_name, n_envs=args.n_env)
        env = VecFrameStack(env, n_stack=4)
        return env
    if args.env_name == "CartPole-v1":
        return DummyVecEnv([lambda: gym.make(args.env_name) for _ in range(args.n_env)])
    return gym.make(args.env_name)


def is_done(done):
    if type(done) is np.ndarray:
        return done.all()
    else:
        return done
