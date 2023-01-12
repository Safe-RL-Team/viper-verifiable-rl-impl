from stable_baselines3 import DQN, PPO
from stable_baselines3.ppo import MlpPolicy

from gym_env import make_env, is_done

def train_oracle(args):
    env = make_env(args)
    model = get_model(env, args)
    model.learn(total_timesteps=args.total_timesteps, log_interval=4)
    model_path = "./log/oracle_" + args.env_name
    model.save(model_path)
    print(f"Training complete. Saved model to {model_path}")


# DQN requires specific hyperparameter tuning
# taken from here: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
def get_model(env, args):
    if args.env_name == 'PongNoFrameskip-v4':
        return DQN("CnnPolicy", env, verbose=args.verbose,
                   learning_starts=100000,
                   learning_rate=1e-4,
                   buffer_size=100_000,
                   batch_size=32,
                   target_update_interval=1000,
                   train_freq=4,
                   gradient_steps=1,
                   exploration_fraction=0.1,
                   exploration_final_eps=0.01,
                   optimize_memory_usage=True
                   )

    # n_envs: 8
    # n_timesteps: !!float 1e5
    # policy: 'MlpPolicy'
    # n_steps: 32
    # batch_size: 256
    # gae_lambda: 0.8
    # gamma: 0.98
    # n_epochs: 20
    # ent_coef: 0.0
    # learning_rate: lin_0.001
    # clip_range: lin_0.2
    if args.env_name == 'CartPole-v1':
        return PPO("MlpPolicy", env, verbose=args.verbose,
                   batch_size=256,
                   n_steps=32,
                   n_epochs=20,
                   learning_rate=linear_schedule(0.001),
                   clip_range=linear_schedule(0.2),
                   gae_lambda=0.8,
                   gamma=0.98,
                   ent_coef=0.0)

    return DQN("MlpPolicy", env, verbose=args.verbose)


def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value

    return func
