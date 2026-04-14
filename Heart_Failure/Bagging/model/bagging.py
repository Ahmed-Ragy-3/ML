import numpy as np
import pandas as pd
from typing import List, Optional

from Heart_Failure.Decision_Tree.model.decision_tree import DecisionTree

class Bagging:
    def __init__(self, n_estimators: int = 10, random_seed: int = 42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.trees: List[DecisionTree] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.trees = []
        rng = np.random.RandomState(self.random_seed)

        for i in range(self.n_estimators):
            # Bootstrap sample (sample with replacement)
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[indices].reset_index(drop=True)
            y_sample = y.iloc[indices].reset_index(drop=True)

            tree = DecisionTree(max_depth=5, min_samples_split=2)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Collect predictions from each tree: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote for each sample
        predictions = []
        for sample_preds in all_preds.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            majority = values[np.argmax(counts)]
            predictions.append(majority)

        return np.array(predictions)