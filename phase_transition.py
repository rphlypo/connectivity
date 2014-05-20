import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


import covariance_learn


def _get_mx(a, b):
    err_str = "{var} should be contained in ({l_bound}, {u_bound})"
    if np.abs(a) >= 1.:
        raise ValueError(err_str.format(var="a", l_bound=-1, u_bound=1))
    if np.abs(b) >= 1. - a ** 2:
        raise ValueError(err_str.format(var="b", l_bound=a ** 2 - 1,
                                        u_bound=1 - a ** 2))
    return np.array([[1, a, b, 0],
                     [a, 1, 0, 0],
                     [b, 0, 1, a],
                     [0, 0, a, 1]], dtype=np.float)


def eval_grid(a_vec=None, b_vec=None, **kwargs):
    if a_vec is None:
        a_vec = np.linspace(0, 1, 11)
    if b_vec is None:
        b_vec = np.linspace(0, 1, 11)
    m = a_vec.size
    n = b_vec.size
    Z = np.random.normal(size=(100, 4))
    tree = [[0, 1], [2, 3]]
    result = [eval_point(a, b, Z, tree, **kwargs)
              for a in a_vec for b in b_vec]
    return (np.array(zip(*result)[0]).reshape(m, n),
            np.array(zip(*result)[1]).reshape(m, n))


def eval_point(a, b, Z, tree, alpha_tol=1e-2, n_jobs=1, verbose=0, n_iter=10):
    try:
        print "evaluating point({}, {})".format(a, b)
        Theta = _get_mx(a, b)
        eigvals, eigvecs = scipy.linalg.eigh(Theta)
        M = eigvecs.dot(np.diag(1 / np.sqrt(eigvals))).dot(eigvecs.T)
        X = Z.dot(M)
        alpha_star_hgl, LL_hgl = \
            covariance_learn.cross_val(X, model_prec=Theta,
                                       method='hgl', htree=tree,
                                       alpha_tol=alpha_tol, n_iter=n_iter,
                                       train_size=.1, test_size=.5,
                                       n_jobs=n_jobs, verbose=verbose,
                                       score_norm="KL")
        alpha_star_gl, LL_gl = \
            covariance_learn.cross_val(X, model_prec=Theta,
                                       method='gl',
                                       alpha_tol=alpha_tol, n_iter=n_iter,
                                       train_size=.1, test_size=.5,
                                       n_jobs=n_jobs, verbose=verbose,
                                       score_norm="KL")
        print "\thgl: {}, gl: {}".format(LL_hgl[-1], LL_gl[-1])
        return LL_hgl[-1], LL_gl[-1]
    except ValueError:
        print "\tinvalid point"
        return np.nan, np.nan


def plot_grid(result, ix):
    f = result[ix]
    f_abs_max = np.nanmax(np.abs(f))

    fig = plt.figure()
    ax = plt.imshow(f, origin='lower', cmap=plt.cm.RdBu_r)  #,
                    #  interpolation='nearest')
    plt.clim(-f_abs_max, f_abs_max)
    cset = plt.contour(f, np.linspace(-f_abs_max, f_abs_max, 10), linewidths=2,
                       cmap=plt.cm.Set2)
    plt.clabel(cset, inline=True, fmt='%1.4f', fontsize=10)
    plt.colorbar(ax)
    plt.show()
    return fig


if __name__ == "__main__":
    result = eval_grid(n_jobs=10, n_iter=10)
    fig = plot_grid(-result, 0)
    fig.set_title("hierarchical graphical lasso")
