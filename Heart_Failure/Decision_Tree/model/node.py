
class Node:
    pass


class InternalNode(Node):
    def __init__(self, feature: str, threshold: float, left: "Node", right: "Node"):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class LeafNode(Node):
    def __init__(self, value: int):
        self.value = value
