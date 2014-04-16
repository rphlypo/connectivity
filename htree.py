import collections


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
