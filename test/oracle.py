import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from gym_env import make_env
from model.paths import get_oracle_path
from train.oracle import get_model


def test_oracle(args):
    env = make_env(args)
    model = get_model(env, args)
    model = model.load(get_oracle_path(args))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, render=args.render)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def is_done(done):
    if type(done) is np.ndarray:
        return done.all()
    else:
        return done
