import numpy as np


class Node(object):
    def __init__(self, neighbours=None, label=None, degree=0):
        self.neighbours = neighbours
        self.label = label
        self.degree = degree

    def connect(self, nodes):
        if self.neighbours is None:
            self.neighbours = []
        if hasattr(nodes, '__iter__'):
            self.degree += len(nodes)
            self.neighbours.extend(nodes)
        else:
            self.degree += 1
            self.neighbours.append(nodes)
        return self


    def deconnect(self, nodes):
        if self.neighbours is None:
            self.neighbours = []
        if hasattr(nodes, '__iter__'):
            intersect = {self.neighbours}.intersection({nodes})
            self.degree -= len(intersect)
            for node in intersect:
                self.neighbours.remove(node)
                node.neighbours.remove(self)
        else:

            self.degree += 1
            self.neighbours.append(nodes)
        return self

def Barabasi_Albert(m0, m, N):
#   if m > m0:
#       raise ValueError('m must be smaller than or equal to m0')
#   # initial graph
#   Graph = [Node() for _ in range(m0)]
#   for (ix, node) in enumerate(Graph):
#       node.connect(Graph[ix + 1:])
#   degrees = np.array([node.degree for node in Graph])
#   cum_degrees = np.float(np.cumsum(degrees)) / np.sum(degrees)
    K = np.eye(N, dtype=np.bool)
    K[np.ix_(np.arange(m0), np.arange(m0))] = True
    for ix in np.arange(m0, N):
        selected = np.zeros((ix,), dtype=np.bool)
        for conn in np.arange(m):
            free = np.logical_not(selected)
            p = np.array(np.sum(K[..., free], axis=0), dtype=np.float)
            cdf = np.cumsum(p) / np.sum(p)
            r = np.random.uniform()
            link = np.where(np.logical_and(r < cdf,
                                           np.logical_not(r >= cdf)))
            K[ix, free[link]] = True
            K[free[link], ix] = True
            selected[free[link]] = True
    rp = np.random.permutation(N)
    return K[np.ix_(rp, rp)]
