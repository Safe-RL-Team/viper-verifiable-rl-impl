import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm


# Difference between best and worst q value for a given state
# This is the ~l loss from the paper
def get_q_difference(model: BaseAlgorithm, obs):
    if isinstance(model, DQN):
        q_values = model.q_net(obs)
        return q_values.max() - q_values.min()
    if isinstance(model, PPO):
        action_prob_tensor = model.policy.get_distribution(torch.from_numpy(obs)).distribution.probs
        action_prob = np.log(action_prob_tensor.detach().numpy().reshape(-1) + 1e-4)
        return action_prob.max() - action_prob.min()

    raise NotImplementedError("Only DQN is supported for now")
