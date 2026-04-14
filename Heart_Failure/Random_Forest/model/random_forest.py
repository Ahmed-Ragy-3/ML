import numpy as np
import pandas as pd
from typing import List, Optional

from Heart_Failure.Random_Forest.model.random_forest_tree import RandomForestTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,   # None → sqrt(n_features) per tree
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_seed = random_seed
        self.trees: List[RandomForestTree] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.trees = []
        rng = np.random.RandomState(self.random_seed)

        for i in range(self.n_estimators):
            # Bootstrap sample (sample with replacement)
            indices = rng.choice(len(x), size=len(x), replace=True)
            x_sample = x.iloc[indices].reset_index(drop=True)
            y_sample = y.iloc[indices].reset_index(drop=True)

            tree = RandomForestTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.randint(0, 2**31 - 1),  # unique seed per tree
            )
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # Shape: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(x) for tree in self.trees])

        # Majority vote across trees for each sample
        predictions = []
        for sample_preds in all_preds.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            majority = values[np.argmax(counts)]
            predictions.append(majority)

        return np.array(predictions)
