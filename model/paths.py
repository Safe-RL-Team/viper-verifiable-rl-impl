def get_oracle_path(args):
    return "./log/oracle_" + args.env_name


def get_viper_path(args):
    n_leaves = str(args.max_leaves) if args.max_leaves is not None else "all-leaves"
    max_depth = str(args.max_depth) if args.max_depth is not None else "all-depth"
    return "./log/viper_" + args.env_name + "_" + n_leaves + "_" + max_depth + ".joblib"
