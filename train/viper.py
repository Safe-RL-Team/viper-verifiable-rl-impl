import warnings

import gym
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from tqdm import tqdm

from gym_env import make_env, is_done
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from test.evaluate import evaluate_policy
from train.oracle import get_model_cls


def train_viper(args):
    # Load stable baselines model from path
    print(f"Training Viper  on {args.env_name}")

    # TODO turn into numpy array
    dataset = []
    policy = None
    policies = []

    for i in tqdm(range(args.steps)):
        beta = 1 - (i / args.steps)
        dataset += sample_trajectory(args, policy, beta)

        clf = DecisionTreeClassifier(max_depth=args.max_depth, max_leaf_nodes=args.max_leaves)
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2] for traj in dataset])
        clf.fit(x, y, sample_weight=weight)

        policies.append(clf)
        policy = clf

    print(f"Viper iteration complete. Dataset size: {len(dataset)}")
    print(f"Performing cross-validation to find the best policy")
    # Cross validate each policy and save the best one
    rewards = []
    env = make_env(args, test_viper=True)
    for i, policy in enumerate(tqdm(policies)):
        mean_reward, std_reward = evaluate_policy(TreeWrapper(policy), env)
        if args.verbose == 2:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")
        rewards.append(mean_reward)

    best_policy = policies[np.argmax(rewards)]
    path = get_viper_path(args)
    print(f"Best policy:\t{np.argmax(rewards)}")
    print(f"Mean reward:\t{np.max(rewards):0.4f}")
    wrapper = TreeWrapper(best_policy)
    wrapper.print_info()
    wrapper.save(path)


def load_oracle_env(args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make_env(args)
        model_cls = get_model_cls(args)
        oracle = model_cls.load(get_oracle_path(args), env=env)
        # SB will add additional wrappers to the env
        env = oracle.env
        return env, oracle


def sample_trajectory(args, policy, beta):
    # We create a new environment for each viper step since
    # vectorized stable baseline environments can only be reset once
    env, oracle = load_oracle_env(args)
    policy = policy or oracle

    trajectory = []
    is_pong = args.env_name == "PongNoFrameskip-v4"

    obs = env.reset()
    while len(trajectory) < args.n_steps:
        active_policy = [policy, oracle][np.random.binomial(1, beta)]
        if isinstance(active_policy, DecisionTreeClassifier):
            if is_pong:
                extracted_obs = env.get_extracted_obs()
                action = active_policy.predict(extracted_obs)
            else:
                action = active_policy.predict(obs)
        else:
            action, _states = active_policy.predict(obs, deterministic=True)

        if not isinstance(active_policy, DecisionTreeClassifier):
            oracle_action = action
        else:
            oracle_action = oracle.predict(obs, deterministic=True)[0]

        obs, reward, done, info = env.step(action)

        if args.render:
            env.render()

        state_loss = get_loss(oracle, obs)
        if is_pong:
            extracted_obs = env.get_extracted_obs()
            trajectory += list(zip(extracted_obs, oracle_action, state_loss))
        else:
            trajectory += list(zip(obs, oracle_action, state_loss))

    return trajectory


# This is the ~l loss from the paper that tries to capture
# how "critical" a state is, i.e. how much of a difference
# it makes to choose the best vs the worst action
def get_loss(model: BaseAlgorithm, obs):
    if isinstance(model, DQN):
        # For q-learners it is the difference between the best and worst q value
        q_values = model.q_net(torch.from_numpy(obs)).detach().numpy()
        return q_values.max(axis=1) - q_values.min(axis=1)
    if isinstance(model, PPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) = -log pi(a|s) TODO copilot says minus??
        action_prob_tensor = model.policy.get_distribution(torch.from_numpy(obs)).distribution.probs
        action_prob = np.log(action_prob_tensor.detach().numpy() + 1e-4)
        return action_prob.max(axis=1) - action_prob.min(axis=1)

    raise NotImplementedError(f"Model type {type(model)} not supported")
