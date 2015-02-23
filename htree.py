import collections
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
import numpy as np


class Node(object):
    def __init__(self, parent=None, children=[], value=None):
        self.parent = parent
        self.children = children
        self.value = value

    def evaluate(self):
        return [val for val in _get_node_values(self)]

    def add_children(self, children):
        """ add one or more children
        """
        # print "adding children {} to {}".format(children, self)
        if isinstance(children, collections.Iterable):
            self.children.extend(children)
        else:
            self.children.append(children)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_siblings(self):
        try:
            return self.parent.children
        except AttributeError:
            return None

    def complement(self):
        if self.parent is None:
            return None
        return [child for child in self.parent.children
                if child is not self]

    def set_value(self, value):
        self.value_ = value
        return self

    def _set_level(self, level):
        self.level_ = level
        return self

    def get_descendants(self):
        """ return all descendant with their relative level in the tree
        """
        desc = [node for node in _get_node_list(self)
                if node[0] is not self]
        desc.sort(key=lambda x: x[1])
        return desc

    def get_ancestors(self):
        """ return ancestor nodes with their relative level in the tree
        """
        node = self
        ancestors = list()
        level = 0
        while True:
            try:
                level -= 1
                node = node.get_parent()
                if node is None:
                    raise StopIteration
                ancestors.append((node, level))
            except StopIteration:
                return ancestors


def _get_node_values(node):
    try:
        if not node.children:
            raise StopIteration
        for child_node in node.children:
            for element in _get_node_values(child_node):
                yield element
    except StopIteration:
        yield node.value


def _get_node_list(node, level=0):
    """ returns list of nodes and their level in the tree (root level = 0)
    """
    yield (node, level)
    try:
        if not node.children:
            raise StopIteration
        for child_node in node.children:
            for element in _get_node_list(child_node, level=level + 1):
                yield (element[0], element[1])
    except StopIteration:
        return


class HTree(object):
    def __init__(self, tree_list, root=None):
        """ hierarchical tree object with 'parent'
        """
        self.root = root
        self.tree_list = tree_list
        self._fit()

    def _fit(self):
        """ scan a nested list to form a tree
        """
        if (isinstance(self.tree_list, collections.Iterable)
                and len(self.tree_list) > 1):
            root_ = Node(parent=self.root,
                         children=[])
            child_nodes = [HTree(t, root=root_)._fit().root_
                           for t in self.tree_list]
            root_.add_children(child_nodes)
            self.root_ = root_
        else:
            self.root_ = Node(parent=self.root, value=self.tree_list,
                              children=[])
        self._update()
        return self

    def _update(self):
        """ update node values setting its level and value
        """
        for (node, node_level) in _get_node_list(self.root_):
            node.set_value(node.evaluate())
            node._set_level(node_level)
        return self

    def get_nodes(self):
        """ useful method to get the entire list of nodes and connections

        This method extracts the nodes with their relationships, one can then
        explore the tree by starting to interrogate the node root_ for its
        children, etc.
        """
        return [node for (node, _) in _get_node_list(self.root_)]

    def get_depth(self):
        if not hasattr(self, 'depth_'):
            self.depth_ = max([lev for (_, lev) in
                               _get_node_list(self.root_)])
        return self.depth_


class HierarchicalKMeans(KMeans):
    def __init__(self, tree_depth=3, n_clusters_per_level=8, max_iter=None,
                 n_init=10, init='k-means++', precompute_distances=False,
                 tol=1e-4, n_jobs=1, random_state=None):
        self.tree_depth = tree_depth
        self.n_clusters_per_level = n_clusters_per_level
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.precompute_distances = precompute_distances
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X):
        kmeans_params = self.get_params()
        del kmeans_params["tree_depth"]
        del kmeans_params["n_clusters_per_level"]
        kmeans_params["n_clusters"] = self.n_clusters_per_level
        print kmeans_params
        self.kmeans = KMeans(**kmeans_params)
        return self

    def transform(self, X):
        self.labels_ = self.kmeans.fit(X)


def construct_tree(arity=8, depth=3, obj=True, rng=None):
    while True:
        try:
            rng = check_random_state(rng)
            nodelist = hierarchical_tree(arity=arity, depth=depth, rng=rng)
            tree = HTree(nodelist)
            # randomness could result in twice the same ID, repeat if needed
            if len(np.unique(tree.root_.value_)) == len(tree.root_.value_):
                raise StopIteration
        except StopIteration:
            if obj:
                return tree
            else:
                return nodelist


def hierarchical_tree(arity=8, depth=3, rng=None):
    rng = check_random_state(rng)
    if depth == 0:
        # We can attribute a random label / ID. This may clash, although very
        # improbable. Fingers crossed !
        return rng.randint(0, 2 ** 31 - 1)
    else:
        return [hierarchical_tree(arity=arity, depth=depth - 1, rng=rng)
                for _ in range(arity)]


def _check_htree(htree):
    if hasattr(htree, '__iter__'):
        return HTree(htree)
    elif isinstance(htree, HTree):
        return htree
    else:
        raise TypeError('object can not be converted to an HTree object')

