from stable_baselines3 import DQN, PPO

from gym_env import make_env
from model.paths import get_oracle_path


def train_oracle(args):
    env = make_env(args)
    if args.resume:
        print("Resuming training")
        model = get_model_cls(args).load(get_oracle_path(args), env=env)
    else:
        model = get_model(env, args)

    model.learn(total_timesteps=args.total_timesteps, eval_freq=args.total_timesteps // 2,
                reset_num_timesteps=not args.resume, tb_log_name=args.env_name)
    model_path = get_oracle_path(args)
    model.save(model_path)
    print(f"Training complete. Saved model to {model_path}")


def linear_schedule(initial_value):
    def func(progress):
        return progress * initial_value

    return func


# DQN requires specific hyperparameter tuning
# taken from here: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
ENV_TO_MODEL = {
    'PongNoFrameskip-v4': {
        'model': DQN,
        'kwargs': {
            'policy': 'CnnPolicy',
            'learning_starts': 100000,
            'learning_rate': 1e-4,
            'buffer_size': 1_000_000,
            'batch_size': 32,
            'target_update_interval': 1000,
            'train_freq': 512,  # TODO try to tune this to n-env
            'gradient_steps': 1,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.01,
            'optimize_memory_usage': True,
        },
        # args we resume with
        'kwargs_resume': {
            'exploration_final_eps': 0.01,
            'exploration_initial_eps': 0.1
        }
    },
    'CartPole-v1': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 256,
            'n_steps': 32,
            'n_epochs': 20,
            'learning_rate': linear_schedule(0.001),
            'clip_range': linear_schedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.0
        }
    },
    'ToyPong-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'batch_size': 32,
            'n_epochs': 20,
            'gamma': 0.99,
            'ent_coef': 0.01
        }
    }
}


def get_model_cls(args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    return ENV_TO_MODEL[args.env_name]['model']


def get_model(env, args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    cfg = ENV_TO_MODEL[args.env_name]
    model_kwargs = cfg['kwargs']
    if args.resume:
        model_kwargs.update(cfg['kwargs_resume'])
    return cfg['model'](env=env, verbose=args.verbose, tensorboard_log='./log', **model_kwargs)
