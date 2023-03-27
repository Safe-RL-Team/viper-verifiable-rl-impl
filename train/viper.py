import warnings

import gym
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from tqdm import tqdm

from gym_env import make_env
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from test.evaluate import evaluate_policy
from train.oracle import get_model_cls


def train_viper(args):
    print(f"Training Viper on {args.env_name}")

    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(args.n_iter), disable=args.verbose > 0):
        beta = 1 if i == 0 else 0
        dataset += sample_trajectory(args, policy, beta)

        clf = DecisionTreeClassifier(ccp_alpha=0.0001, criterion="entropy", max_depth=args.max_depth,
                                     max_leaf_nodes=args.max_leaves)
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2] for traj in dataset])

        clf.fit(x, y, sample_weight=weight)

        policies.append(clf)
        policy = clf

        env = make_env(args, test_viper=True)
        mean_reward, std_reward = evaluate_policy(TreeWrapper(policy), env, n_eval_episodes=100)
        if args.verbose == 2:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")
        rewards.append(mean_reward)

    print(f"Viper iteration complete. Dataset size: {len(dataset)}")
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
        model_cls, _ = get_model_cls(args)
        oracle = model_cls.load(get_oracle_path(args), env=env)
        oracle.verbose = args.verbose
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
    n_steps = args.total_timesteps // args.n_iter
    while len(trajectory) < n_steps:
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

        next_obs, reward, done, info = env.step(action)

        if args.render:
            env.render()

        state_loss = get_loss(env, oracle, obs)
        if is_pong:
            extracted_obs = env.get_extracted_obs()
            trajectory += list(zip(extracted_obs, oracle_action, state_loss))
        else:
            trajectory += list(zip(obs, oracle_action, state_loss))

        obs = next_obs

    return trajectory


def get_loss(env, model: BaseAlgorithm, obs):
    """
    This is the ~l loss from the paper that tries to capture
    how "critical" a state is, i.e. how much of a difference
    it makes to choose the best vs the worst action

    Instead of training the decision tree with this loss directly (which is not possible because it is not convex)
    we use it as a weight for the samples in the dataset which in expectation leads to the same result
    """
    if isinstance(model, DQN):
        # For q-learners it is the difference between the best and worst q value
        q_values = model.q_net(torch.from_numpy(obs)).detach().numpy()
        # q_values n_env x n_actions
        return q_values.max(axis=1) - q_values.min(axis=1)
    if isinstance(model, PPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        possible_actions = np.arange(env.action_space.n)

        obs = torch.from_numpy(obs)
        log_probs = []
        for action in possible_actions:
            action = torch.from_numpy(np.array([action])).repeat(obs.shape[0])
            _, log_prob, _ = model.policy.evaluate_actions(obs, action)
            log_probs.append(log_prob.detach().numpy().flatten())

        log_probs = np.array(log_probs).T
        return log_probs.max(axis=1) - log_probs.min(axis=1)

    raise NotImplementedError(f"Model type {type(model)} not supported")
