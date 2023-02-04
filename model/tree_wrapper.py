from typing import Optional, Tuple

import numpy as np
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier


# Wrapper around our extracted decision tree, mostly so that we can use the sb policy evaluator
class TreeWrapper:
    def __init__(self, tree: DecisionTreeClassifier):
        self.tree = tree

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.tree.predict(observation), None

    @classmethod
    def load(cls, path: str):
        clf = load(path)
        return TreeWrapper(clf)

    def save(self, path: str):
        print(f"Saving to\t{path}")
        dump(self.tree, path)

    def print_info(self):
        print(f"Max depth:\t{self.tree.get_depth()}")
        print(f"# Leaves:\t{self.tree.get_n_leaves()}")
