import collections
from sklearn.base import BaseEstimator, TransformerMixin


class Node(BaseEstimator, TransformerMixin):
    def __init__(self, parent=None, children=[]):
        self.parent = parent
        self.children = children

    def fit(self, X):
        self.value_ = evaluate(self)
        return self


def evaluate(node, current_values=[]):
    current_values.extend([evaluate(child)
                           for child in node.children])
    return set(current_values)


def complement(node):
    return node.parent.fit().value_ - node.fit().value_


class HTree(BaseEstimator, TransformerMixin):
    def __init__(self, root=None):
        return None

    def fit(self, X, y=None):
        """ get the root node right, develop the tree using transform()
        """
        if (not isinstance(X, collections.Iterable) or len(X) == 1):
            self.root_ = Node(parent=self.root)
        else:
            self.root_ = Node(parent=self.root,
                              children=[HTree(tree) for tree in X])

    def transform(self, X):
        return {self.root_} & set([Node(parent=self.root_,
                                        children=_expand(children.fit()))
                                   for children in self.root_.children])
