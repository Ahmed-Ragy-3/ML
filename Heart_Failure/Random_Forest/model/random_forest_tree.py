from typing import Tuple, Union, List, Optional, cast
import pandas as pd
import numpy as np
from Decision_Tree.model.node import Node, InternalNode, LeafNode


class RandomForestTree:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.root: Optional[Node] = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features          # None → use sqrt(n_features)
        self.random_state = random_state
        self._rng: Optional[np.random.RandomState] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._rng = np.random.RandomState(self.random_state)
        y = y.astype(int)
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._traverse(row, self.root) for _, row in X.iterrows()])

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        # Stopping conditions
        if (
            len(y.unique()) == 1
            or len(y) < self.min_samples_split
            or depth >= self.max_depth
        ):
            return LeafNode(value=self._most_common_label(y))

        # Randomly sample a subset of features at this node
        all_features = list(X.columns)
        k = self._resolve_max_features(len(all_features))
        candidate_features = self._rng.choice(all_features, size=k, replace=False)

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in candidate_features:
            thresholds = self._get_thresholds(X[feature])

            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self._split_dataset(X, y, feature, threshold)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gain = self._information_gain(y.values, y_l.values, y_r.values)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        # No valid split found → leaf
        if best_feature is None:
            return LeafNode(value=self._most_common_label(y))

        X_l, X_r, y_l, y_r = self._split_dataset(X, y, best_feature, best_threshold)

        left_child = self._build_tree(X_l, y_l, depth + 1)
        right_child = self._build_tree(X_r, y_r, depth + 1)

        return InternalNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return max(1, int(np.sqrt(n_features)))
        return max(1, min(self.max_features, n_features))

    def _split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str,
        threshold: Union[int, float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _get_thresholds(self, feature_values: pd.Series) -> List[float]:
        unique_values = np.sort(feature_values.unique())
        return [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]

    def _entropy(self, y: np.ndarray) -> float:
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _information_gain(
        self,
        parent_y: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray,
    ) -> float:
        total = len(parent_y)
        w_left = len(left_y) / total
        w_right = len(right_y) / total
        return self._entropy(parent_y) - (
            w_left * self._entropy(left_y) + w_right * self._entropy(right_y)
        )

    def _most_common_label(self, y: pd.Series) -> int:
        return y.value_counts().idxmax()

    def _traverse(self, row: pd.Series, node: Node) -> int:
        if isinstance(node, LeafNode):
            return node.value

        node = cast(InternalNode, node)

        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        else:
            return self._traverse(row, node.right)
