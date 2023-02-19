import math
from itertools import pairwise

from z3 import RealVector, And, Implies, Or, sat, If, Solver, Not

from gym_env import make_env
from gym_env.toy_pong import ToyPong
from model.paths import get_viper_path
from model.tree_wrapper import TreeWrapper

from sklearn.tree import _tree


# https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
# Extract rules: https://stackoverflow.com/a/39772170
def verify_correct(args):
    assert args.env_name == "ToyPong-v0", "Only ToyPong-v0 is supported for now"
    env: [ToyPong] = make_env(args).envs[0].unwrapped
    tree = TreeWrapper.load(get_viper_path(args)).tree
    tree_ = tree.tree_
    # Contains the feature that is tested at each node (-1 = leaf, -2 = undefined)
    features = tree_.feature

    """
    Aggregates the rules of the decision tree into a dictionary of partitions.
    {
        'S_0': {
            conditions: [(feature_0, '<=', threshold_0), (feature_1, '>', threshold_1), ...],
            value: 1 # the constant predicted action for this node 
        }
    }
    
    Each leaf of the decision tree defines a polyhedron in the 5 dimensional state space within which
    the predicted action does not change.
    This means that within this partition the dynamics are linear meaning that we can
    approximate a state transition with a vector beta like so: f_pi(s) = f_i(s) beta_i^T s
    
    Therefore for each partition S_i we need to get:
    1. a list of conditions to check if an s is in S_i
    2. the beta vector for the state transitions
    
    How to obtain beta?
    We split the state vector s_t into its components and treat them differently:
    - For the paddle position this is the predicted action for each decision tree leaf node
    - For the ball position and velocity we manually define the environment dynamics
    """

    def recurse(node, path, result):
        if features[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            left_path = path + [(features[node], "<=", threshold)]
            result = recurse(tree_.children_left[node], left_path, result)
            right_path = path + [(features[node], ">", threshold)]
            result = recurse(tree_.children_right[node], right_path, result)
        else:
            result[node] = {
                'conditions': path,
                'value': tree_.value[node][0][0]
            }

        return result

    tree_partitions = recurse(0, [], {})

    # s el of S_i
    def is_state_in_partition(s, partition):
        conditions = partition['conditions']
        return And(
            [And(s[feature] <= threshold) if op == "<=" else And(s[feature] > threshold) for feature, op, threshold in
             conditions])

    t_max = 2 # math.ceil(2 * env.height / env.min_speed)

    # Create vector of state variables for each timestep
    # s = [paddle_x, ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y]
    s = [RealVector('s_t_{}'.format(i), 5) for i in range(t_max)]

    def phi_t(s_t, s_t_1):
        results = []
        for partition in tree_partitions.values():
            # the tree partitions only affect the paddle position
            next_state_is_s_t = And(s_t[0] == s_t[0] + env.paddle_speed * partition['value'])
            results.append(Implies(is_state_in_partition(s_t_1, partition), next_state_is_s_t))

        # Manually add the dynamics of the ball
        collision_left = And(s_t_1[1] < 0)
        collision_left_beta = And([
            s_t[1] == 0,
            # Keep moving in the y direction
            s_t[2] == s_t_1[4] + s_t[2],
            # Make x vel positive
            s_t[3] == abs(s_t_1[3]),
            # Keep y vel
            s_t[4] == s_t_1[4]
        ])
        results.append(Implies(collision_left, collision_left_beta))

        collision_right = And(s_t_1[1] > env.width)
        collision_right_beta = And([
            s_t[1] == env.width,
            # Keep moving in the y direction
            s_t[2] == s_t_1[4] + s_t[2],
            # Make x vel negative
            s_t[3] == -abs(s_t_1[3]),
            # Keep y vel
            s_t[4] == s_t_1[4]
        ])
        results.append(Implies(collision_right, collision_right_beta))

        collision_top = And(s_t_1[2] < 0)
        collision_top_beta = And([
            # Keep moving in the x direction
            s_t[1] == s_t_1[3] + s_t[1],
            s_t[2] == 0,
            # Keep x vel
            s_t[3] == s_t_1[3],
            # Make y vel positive
            s_t[4] == abs(s_t_1[4])
        ])
        results.append(Implies(collision_top, collision_top_beta))

        #
        collision_bottom = And([s_t_1[2] > env.height,
                                # Ball misses the paddle
                                Or([s_t_1[1] < s_t_1[0] - env.paddle_length, s_t_1[1] > s_t_1[0] + env.paddle_length])])
        collision_bottom_beta = And([
            # Keep moving in the x direction
            s_t[1] == s_t_1[3] + s_t[1],
            # Keep moving in the y direction
            s_t[2] == s_t_1[4] + s_t[2],
            # Keep x vel
            s_t[3] == s_t_1[3],
            # Keep y vel
            s_t[4] == s_t[4],
        ])
        results.append(Implies(collision_bottom, collision_bottom_beta))

        # Ball hits the paddle
        collision_bottom_paddle = And([s_t_1[2] > env.height,
                                       And([s_t_1[1] >= s_t_1[0] - env.paddle_length,
                                            s_t_1[1] <= s_t_1[0] + env.paddle_length])])
        collision_bottom_paddle_beta = And([
            # Keep moving in the x direction
            s_t[1] == s_t_1[3] + s_t[1],
            s_t[2] == env.height,
            # Keep x vel
            s_t[3] == s_t_1[3],
            # Make y vel negative
            s_t[4] == -abs(s_t_1[4])
        ])
        results.append(Implies(collision_bottom_paddle, collision_bottom_paddle_beta))
        print("phi_t len", len(results))
        return Or(results)

    # Check if the state is in a partition

    # Assert that we are at a safe state at timestep t
    # i.e. the ball is in the top half of the screen
    y_t_safe = And([And(s_t[2] >= 0, s_t[2] <= env.height / 2) for s_t in s[1:]])
    y_0_safe = And(s[0][2] >= 0, s[0][2] <= env.height / 2)

    # Assert that the state transitions are correct
    phi = [phi_t(s_t, s_t_1) for s_t, s_t_1 in pairwise(s)]

    program = Implies(And([y_0_safe] + phi), y_t_safe)

    # To show that the program is correct, we need to show that its **negation** is unsatisfiable
    # i.e. there is no counterexample to the program
    solver = Solver()
    solver.add(Not(program))
    # If the program is not correct print where the counterexample is
    if solver.check() == sat:
        print("counterexample found!")
        m = solver.model()
        for d in m.decls():
            print("%s = %s" % (d.name(), m[d]))
    else:
        print("the program is correct!")


def abs(x):
    return If(x >= 0, x, -x)

