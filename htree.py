import collections
from sklearn.cluster import KMeans


class Node(object):
    def __init__(self, parent=None, children=[], value=None):
        self.parent = parent
        self.children = children
        self.value = value

    def evaluate(self):
        return set([child.evaluate()
                    if child.children
                    else child.value
                    for child in self.children])

    def add_children(self, children):
        """ add one or more children
        """
        print "adding children {} to {}".format(children, self)
        if isinstance(children, collections.Iterable):
            self.children.extend(children)
        else:
            self.children.append(children)

    def complement(self):
        return self.parent.evaluate - self.evaluate

    def set_value(self, value):
        self.value = value


class HTree(object):
    def __init__(self, root=None, tree_list=None):
        self.root = root
        self.tree_list = tree_list
        self.expand()

    def expand(self):
        """ get the root node right, develop the tree using transform()
        """
        if (not isinstance(self.tree_list, collections.Iterable)
                or len(self.tree_list) == 1):
            self.root_ = Node(parent=self.root, value=self.tree_list,
                              children=[])
        else:
            self.root_ = Node(parent=self.root)

            children = [HTree(root=self.root_, tree_list=tree).root_
                        for tree in self.tree_list]
            self.root_.add_children(children)
        return self


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
