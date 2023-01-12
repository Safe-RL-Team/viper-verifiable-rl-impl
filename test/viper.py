import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from gym_env import make_env
from model.tree_wrapper import TreeWrapper


def test_viper(args):
    env = make_env(args)
    model = TreeWrapper.load("./log/viper_" + args.env_name + ".joblib")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def is_done(done):
    if type(done) is np.ndarray:
        return done.all()
    else:
        return done
