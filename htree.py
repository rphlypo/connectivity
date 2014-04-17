import collections


class Node(object):
    def __init__(self, parent=None, children=[], value=None):
        self.parent = parent
        self.children = children
        self.value = value

    def evaluate(self):
        return list(_get_node_values(self))

    def add_children(self, children):
        """ add one or more children
        """
        print "adding children {} to {}".format(children, self)
        if isinstance(children, collections.Iterable):
            self.children.extend(children)
        else:
            self.children.append(children)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def complement(self):
        return self.parent.evaluate - self.evaluate

    def set_value(self, value):
        self.value = value


def _get_node_values(node):
    try:
        if not node.children:
            raise StopIteration
        for child_node in node.children:
            for element in _get_node_values(child_node):
                yield element
    except StopIteration:
        yield node.value


class HTree(object):
    def __init__(self, root=None):
        """ hierarchical tree object with 'parent'
        """
        self.root = root

    def tree(self, tree_list):
        """ scan a list of lists to form a tree
        """
        if isinstance(tree_list, collections.Iterable) and len(tree_list) > 1:
            root_ = Node(parent=self.root,
                         children=[])
            child_nodes = [HTree(root=root_).tree(t).root_
                           for t in tree_list]
            root_.add_children(child_nodes)
            self.root_ = root_
        else:
            self.root_ = Node(parent=self.root, value=tree_list,
                              children=[])
        return self
