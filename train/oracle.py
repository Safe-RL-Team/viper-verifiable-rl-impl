from stable_baselines3 import DQN, PPO

from gym_env import make_env
from model.paths import get_oracle_path
from train.learning_rates import LinearSchedule, HalfLinearSchedule


def train_oracle(args):
    env = make_env(args)
    if args.resume:
        print("Resuming training")
        cls, policy_kwargs = get_model_cls(args)
        model = cls.load(get_oracle_path(args), env=env)
    else:
        model, policy_kwargs = get_model(env, args)

    log_name = f"{args.log_prefix}{args.env_name}_{args.n_env}env_{kwargs_to_str(policy_kwargs)}"

    model.learn(total_timesteps=args.total_timesteps, eval_freq=args.total_timesteps // 10,
                reset_num_timesteps=not args.resume, tb_log_name=log_name)
    model_path = get_oracle_path(args)
    model.save(model_path)
    model.save(f"./log/{log_name}/model")
    print(f"Training complete. Saved model to {model_path}")


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
            'train_freq': 4,
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
            'learning_rate': LinearSchedule(0.001),
            'clip_range': LinearSchedule(0.2),
            'gae_lambda': 0.8,
            'gamma': 0.98,
            'ent_coef': 0.0
        }
    },
    'ToyPong-v0': {
        'model': PPO,
        'kwargs': {
            'policy': 'MlpPolicy',
            'learning_rate': HalfLinearSchedule(0.0003),
            'policy_kwargs': {
                'net_arch': [64, dict(pi=[128, 64], vf=[64, 64])]
            }
        }
    }
}


def kwargs_to_str(kwargs):
    return '_'.join([f"{k}-{v}" for k, v in kwargs.items() if k not in ['policy', 'policy_kwargs']])


def get_model_cls(args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    return ENV_TO_MODEL[args.env_name]['model'], ENV_TO_MODEL[args.env_name]['kwargs']


def get_model(env, args):
    if args.env_name not in ENV_TO_MODEL:
        raise ValueError(f"Unsupported env: {args.env_name}")

    cfg = ENV_TO_MODEL[args.env_name]
    model_kwargs = cfg['kwargs']
    if args.resume:
        model_kwargs.update(cfg['kwargs_resume'])
    return cfg['model'](env=env, verbose=args.verbose, tensorboard_log='./log', seed=args.seed,
                        **model_kwargs), model_kwargs
