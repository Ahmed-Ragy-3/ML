from typing import Tuple, Union, List, Optional, cast
import pandas as pd
import numpy as np
from Decision_Tree.model.node import Node, InternalNode, LeafNode

class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.root: Optional[Node] = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        y = y.astype(int)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        if (
            len(y.unique()) == 1
            or len(y) < self.min_samples_split
            or depth >= self.max_depth
        ):
            return LeafNode(value=self._most_common_label(y))

        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in X.columns:
            thresholds = self.get_thresholds(X[feature])

            for threshold in thresholds:
                X_l, X_r, y_l, y_r = self.split_dataset(X, y, feature, threshold)

                if len(y_l) == 0 or len(y_r) == 0:
                    continue

                gain = self.information_gain(y.values, y_l.values, y_r.values)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        # no valid split → leaf
        if best_feature is None:
            return LeafNode(value=self._most_common_label(y))

        # split
        X_l, X_r, y_l, y_r = self.split_dataset(X, y, best_feature, best_threshold)

        # recursion
        left_child = self._build_tree(X_l, y_l, depth + 1)
        right_child = self._build_tree(X_r, y_r, depth + 1)

        return InternalNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature: str,
        threshold: Union[int, float]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        left_mask = X[feature] <= threshold
        right_mask = X[feature] > threshold

        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def get_thresholds(self, feature_values: pd.Series) -> List[float]:
        unique_values = np.sort(feature_values.unique())

        return [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]


    def entropy(self, y: np.ndarray) -> float:
        probs = np.bincount(y) / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def information_gain(self, parent_y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray) -> float:

        total = len(parent_y)
        w_left = len(left_y) / total
        w_right = len(right_y) / total

        return self.entropy(parent_y) - (w_left * self.entropy(left_y) + w_right * self.entropy(right_y))

    def _most_common_label(self, y: pd.Series) -> int:
        return y.value_counts().idxmax()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._traverse(row, self.root) for _, row in X.iterrows()])

    def _traverse(self, row: pd.Series, node: Node) -> int:
        if isinstance(node, LeafNode):
            return node.value

        node = cast(InternalNode, node)

        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        else:
            return self._traverse(row, node.right)