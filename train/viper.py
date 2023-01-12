import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

from gym_env import make_env, is_done
from model import get_q_difference
from train.oracle import get_model


def sample_trajectory(n_traj, env, policy, oracle, beta):
    trajectory = []
    for _ in range(n_traj):
        obs = env.reset()
        while True:
            active_policy = [policy, oracle][np.random.binomial(1, beta)]

            if isinstance(active_policy, DecisionTreeClassifier):
                action = active_policy.predict(obs)
                print("action", action)
            else:
                action, _states = active_policy.predict(obs, deterministic=True)

            oracle_action = oracle.predict(obs, deterministic=True)
            print("oracle_action", oracle_action)
            obs, reward, done, info = env.step(action)
            q = get_q_difference(oracle, obs)
            trajectory.append((obs[0], oracle_action[0], q))

            if is_done(done):
                env.reset()
                break

    return trajectory


def train_viper(args):
    assert args.n_env == 1, "Use only one env to keep it simple"
    # Load stable baselines model from path
    env = make_env(args)
    model = get_model(env, args)
    oracle = model.load("./log/oracle_" + args.env_name)
    print(f"Training Viper using {type(oracle).__name__} oracle on {args.env_name}")

    dataset = []
    policy = oracle
    policies = []
    for i in range(args.steps):
        beta = 1 - (i / args.steps)
        print(f"Sampling trajectories with beta={beta}")
        dataset += sample_trajectory(args.n_traj, env, policy, oracle, beta)

        l_weights = np.array([traj[2] for traj in dataset])
        l_proba = l_weights / l_weights.sum()
        sampled_indices = np.random.choice(len(dataset), len(dataset), p=l_proba)
        sampled_trajectories = [dataset[i] for i in sampled_indices]

        clf = DecisionTreeClassifier()
        clf.fit([traj[0] for traj in sampled_trajectories], [traj[1] for traj in sampled_trajectories])

        policies.append(clf)
        policy = clf

    print(f"Viper complete. Dataset size: {len(dataset)}")
    # Cross validate each policy and select and save the best one
    mean_scores = []
    for i, policy in enumerate(policies):
        scores = cross_val_score(policy, [traj[0] for traj in dataset], [traj[1] for traj in dataset], cv=5)
        if args.verbose == 2:
            print(f"Policy score: {scores.mean():0.4f} +/- {scores.std():0.4f}")
        mean_scores.append(scores.mean())

    best_policy = policies[np.argmax(mean_scores)]
    path = f"./log/viper_{args.env_name}.joblib"
    print(policy.get_depth())
    print(policy.get_n_leaves())
    print(f"Saving best policy {np.argmax(mean_scores)} with score {max(mean_scores):0.4f} at {path}")
    dump(best_policy, path)
