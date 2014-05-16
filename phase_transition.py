import numpy as np
import maplotlib.pyplot as plt
import scipy.linalg


import covariance_learn


def _get_designed_mx(a, b):
    err_str = "{var} should be contained in ({min_bound}, {max_bound})"
    if np.abs(a) >= 1.:
        raise ValueError(err_str.format("a", -1, 1))
    if np.abs(b) >= 1. - a ** 2:
        raise ValueError(err_str.format("b", a ** 2 - 1, 1 - a ** 2))
    return np.array([[1, a, b, 0],
                     [a, 1, 0, 0],
                     [b, 0, 1, a],
                     [0, 0, a, 1]], dtype=np.float)


def _eval_on_grid():
    a_ = np.linspace(0, 1, 11)
    b_ = np.linspace(0, 1, 11)
    Z = np.random.normal(size=(100, 4))
    tree = [[0, 1], [2, 3]]
        result = np.ones((a.size(), b.size())) * np.nan
    for (ix_a, a) in enumerate(a_):
        for (ix_b, b) in enumerate(b_):
            try:
                M = _get_designed_mx(a, b)
                eigvals, eigvecs = scipy.linalg.eigh(M)
                M = eigvecs.dot(np.diag(1 / np.sqrt(eigvals))).dot(eigvecs.T)
                X = Z.dot(M)
                Y = M
                alpha_star_hgl, LL_hgl = \
                    covariance_learn._cross_val(X, method='hgl', htree=tree,
                                                alpha_tol=1e-4, n_iter=25,
                                                train_size=.2, test_size=.5)
                score_hgl = covariance_learn.HierarchicalGraphLasso(
                    tree, alpha=alpha_star_hgl).fit(Y).score(Y)
                alpha_star_gl, LL_gl = \
                    covariance_learn._cross_val(X, method='gl',
                                                alpha_tol=1e-4, n_iter=25,
                                                train_size=.2, test_size=.5)
                score_gl = covariance_learn.GraphLasso(
                    alpha=alpha_star_gl).fit(Y).score(Y)

            except ValueError:
                continue

