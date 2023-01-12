from typing import Optional, Tuple

import numpy as np
from joblib import load
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
