import gym
from gym import register
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from gym_env.pong_wrapper import PongWrapper

register(
    id='ToyPong-v0',
    entry_point='gym_env.toy_pong:ToyPong',
    kwargs={'args': None}
)


def make_env(args, test_viper=False):
    if args.env_name == "PongNoFrameskip-v4":
        env = make_atari_env("PongNoFrameskip-v4", n_envs=args.n_env)
        if test_viper is True:
            return PongWrapper(env, return_extracted_obs=True)
        return VecFrameStack(PongWrapper(env), n_stack=4)
    if args.env_name == "CartPole-v1":
        return DummyVecEnv([lambda: gym.make(args.env_name) for _ in range(args.n_env)])
    elif args.env_name == "ToyPong-v0":
        return DummyVecEnv([lambda: Monitor(gym.make(args.env_name, args=args)) for _ in range(args.n_env)])

    raise NotImplementedError(f"Environment {args.env_name} not implemented")
