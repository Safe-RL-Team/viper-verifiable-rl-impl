import math
import time

import numpy as np
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
    wrapper = TreeWrapper.load(get_viper_path(args))
    start = time.time()

    print("Verifying correctness for ToyPong-v0")
    print("Using Tree:")
    wrapper.print_info()
    tree = wrapper.tree
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
            action = np.argmax(tree_.value[node][0])
            if action == 0:
                value = -1  # move left
            elif action == 1:
                value = 1  # move right
            else:
                value = 0  # do nothing

            result[node] = {
                'conditions': path,
                'value': value
            }

        return result

    tree_partitions = recurse(0, [], {})

    # s el of S_i
    def is_state_in_partition(s, partition):
        conditions = partition['conditions']
        return And(
            [And(s[feature] <= threshold) if op == "<=" else And(s[feature] > threshold) for feature, op, threshold in
             conditions])

    t_max = math.ceil(2 * env.height / env.min_speed)

    # Create vector of state variables for each timestep
    # s = [paddle_x, ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y]
    s = [RealVector('s_t_{}'.format(i), 5) for i in range(t_max)]

    def phi_t(s_now, s_next):
        controller = []
        for partition in tree_partitions.values():
            # the tree partitions only affect the paddle position
            next_paddle = clamp(s_now[0] + env.paddle_speed * partition['value'], 0, env.width)
            next_state_is_s_t = And(s_next[0] == next_paddle)
            controller.append(Implies(is_state_in_partition(s_now, partition), next_state_is_s_t))

        # Manually add the dynamics of the ball
        system = []
        # No collision, i.e. the ball is within the box
        no_collision = And([s_now[1] >= 0, s_now[1] <= env.width, s_now[2] >= 0, s_now[2] <= env.height])
        no_collision_beta = And([
            # Keep moving in the x direction
            s_next[1] == s_now[3] + s_now[1],
            # Keep moving in the y direction
            s_next[2] == s_now[4] + s_now[2],
            # Keep x vel
            s_next[3] == s_now[3],
            # Keep y vel
            s_next[4] == s_now[4],
        ])
        system.append(Implies(no_collision, no_collision_beta))

        collision_left = And(s_now[1] < 0)
        collision_left_beta = And([
            s_next[1] == 0,
            # Keep moving in the y direction
            s_next[2] == s_now[4] + s_now[2],
            # Make x vel positive
            s_next[3] == abs(s_now[3]),
            # Keep y vel
            s_next[4] == s_now[4]
        ])
        system.append(Implies(collision_left, collision_left_beta))

        collision_right = And(s_now[1] > env.width)
        collision_right_beta = And([
            s_next[1] == env.width,
            # Keep moving in the y direction
            s_next[2] == s_now[4] + s_now[2],
            # Make x vel negative
            s_next[3] == -abs(s_now[3]),
            # Keep y vel
            s_next[4] == s_now[4]
        ])
        system.append(Implies(collision_right, collision_right_beta))

        collision_top = s_now[2] < 0
        collision_top_beta = And([
            # Keep moving in the x direction
            s_next[1] == s_now[3] + s_now[1],
            s_next[2] == 0,
            # Keep x vel
            s_next[3] == s_now[3],
            # Make y vel positive
            s_next[4] == -1 * s_now[4]
        ])
        system.append(Implies(collision_top, collision_top_beta))

        # Ball passes through the bottom of the screen
        collision_bottom = And([s_now[2] > env.height,
                                # Ball misses the paddle
                                Or([s_now[1] < s_now[0] - env.paddle_length, s_now[1] > s_now[0] + env.paddle_length])])
        system.append(Implies(collision_bottom, no_collision_beta))
        #
        # Ball hits the paddle
        collision_bottom_paddle = And([s_now[2] > env.height,
                                       And([s_now[1] >= s_now[0] - env.paddle_length,
                                            s_now[1] <= s_now[0] + env.paddle_length])])
        collision_bottom_paddle_beta = And([
            # Keep moving in the x direction
            s_next[1] == s_now[3] + s_now[1],
            s_next[2] == env.height,
            # Keep x vel
            s_next[3] == s_now[3],
            # Make y vel negative
            s_next[4] == -abs(s_now[4])
        ])
        system.append(Implies(collision_bottom_paddle, collision_bottom_paddle_beta))
        return And(And(system + controller))

    # Check if the state is in a partition

    # Assert that we are at a safe state at timestep t
    # i.e. the ball is in the top half of the screen
    y_t_safe = Or([And(s_t[2] >= 0, s_t[2] <= env.height / 2) for s_t in s[1:]])
    y_0_safe = And(s[0][2] >= 0, s[0][2] <= env.height / 2, s[0][1] >= 0, s[0][1] <= env.width,
                   # s[0][0] >= 0, s[0][0] <= env.width)
                   s[0][0] == env.width / 2,
                   s[0][1] == env.width / 2
                   )

    # Assert that the state transitions are correct
    phi = [phi_t(s_t_1, s_t) for s_t_1, s_t in pairwise(s)]
    vel_constraint = [
        And([abs(s_t[3]) <= env.max_speed, abs(s_t[3]) >= env.min_speed, abs(s_t[4]) <= env.max_speed,
             abs(s_t[4]) >= env.min_speed]) for
        s_t in s]

    program = Implies(And([y_0_safe] + phi + vel_constraint), y_t_safe)
    # To show that the program is correct, we need to show that its **negation** is unsatisfiable
    # i.e. there is no counterexample to the program
    solver = Solver()
    solver.add(Not(program))

    # If the program is not correct print where the counterexample is
    if solver.check() == sat:
        print("counterexample found!")
        model = solver.model()
        debug_counter_example(model, env)
    else:
        print("the program is correct!")

    end = time.time()
    print(f"Completed in: {end - start:.2f}s")


def abs(x):
    return If(x >= 0, x, -x)


def clamp(x, min, max):
    return If(x < min, min, If(x > max, max, x))


KEY_TO_LABEL = {
    0: 'paddle_x',
    1: 'ball_pos_x',
    2: 'ball_pos_y',
    3: 'ball_vel_x',
    4: 'ball_vel_y'
}


def debug_counter_example(model, env):
    import re
    import ast
    from pprint import pprint

    def get_time(decl_name):
        m = re.search('s_t_([0-9]+?)_', decl_name)
        return m.group(1)

    def get_decls(model):
        states = {}
        for d in model.decls():
            name = d.name()
            value = model[d]
            time = int(get_time(d.name()))
            if time not in states:
                states[time] = {}

            state_key = int(name[-1])
            value_str = value.as_decimal(3)
            if value_str[-1] == "?":
                value_str = value_str[:-1]
            value = ast.literal_eval(value_str)

            states[time][KEY_TO_LABEL[state_key]] = value

        return states

    states = get_decls(model)
    sorted_state_items = list(sorted(states.items(), key=lambda x: x[0]))
    pprint(sorted_state_items)
    print(f"""
    self.ball_vel_x = {states[0]['ball_vel_x']}
    self.ball_vel_y = {states[0]['ball_vel_y']}
    self.ball_pos_y = {states[0]['ball_pos_y']}
    self.ball_pos_x = {states[0]['ball_pos_x']}
    self.paddle_x = {states[0]['paddle_x']}
    """)

    def render_example():
        states = get_decls(model)
        for state in sorted(states.items(), key=lambda x: x[0]):
            env.paddle_x = state[1]['paddle_x']
            env.ball_pos_x = state[1]['ball_pos_x']
            env.ball_pos_y = state[1]['ball_pos_y']
            env.ball_vel_x = state[1]['ball_vel_x']
            env.ball_vel_y = state[1]['ball_vel_y']
            if env.ball_pos_y > env.height:
                print("Ball passed through the bottom of the screen")
                break
            env.render()
            time.sleep(1)

    render_example()
