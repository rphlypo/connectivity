import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


import covariance_learn
reload(covariance_learn)


def _get_mx(a, b, mx_type='small'):
    if mx_type == 'small':
        err_str = "{var} should be contained in ({l_bound}, {u_bound})"
        if np.abs(a) >= 1.:
            raise ValueError(err_str.format(var="a", l_bound=-1, u_bound=1))
        abs_bound_b = 2 * np.sqrt((1 - a) * (3 - a - np.sqrt(8 * (1 - a))))
        if np.abs(b) >= abs_bound_b:
            raise ValueError(err_str.format(var="b", l_bound=-abs_bound_b,
                                            u_bound=abs_bound_b))
        # TODO make this a block form such that the second level is sparse
        return np.array([[1, a, b, b / 2],
                         [a, 1, b / 2, 0],
                         [b, b / 2, 1, a],
                         [b / 2, 0, a, 1]], dtype=np.float)
    elif mx_type in {'gael', 'ronald'}:
        mx = np.eye(8, 8)
        mx.flat[8::18] = a
        mx.flat[1::18] = a

        if mx_type == 'gael':
            mx[:-2:2, -1] = b
            mx[-1, :-2:2] = b
        elif mx_type == 'ronald':
            mx[:-2:2, -2] = b
            mx[:-2:2, -1] = b / 2
            mx[1:-2:2, -2] = b / 2
            mx[-2, :-2:2] = b
            mx[-1, :-2:2] = b / 2
            mx[-2, 1:-2:2] = b / 2

    elif mx_type == 'smith':
        # communities
        c = 4
        #
        mx = np.identity(c ** 2)
        # AR model like within community
        for k in np.arange(c):
            mx[k * c + 1:(k + 1) * c, k * c] = a
            mx[k * c, k * c + 1:(k + 1) * c] = a
        #
        mx[c::c, 0] = b
        mx[0, c::c] = b

        plt.matshow(mx)
    eigvals = scipy.linalg.eigvalsh(mx)
    if np.any(eigvals < 1e-9):
        raise ValueError('Your combination (a, b) yields ' +
                         'an ill-conditioned matrix ' +
                         '(lambda_min={})'.format(np.min(eigvals)))
    return mx


def eval_grid(a_vec=None, b_vec=None, mx_type="small", **kwargs):
    """ evaluate the performance of (hierarchical) graph lasso

    arguments
    ---------
    a_vec   : np.ndarray of dim (m,) or list
            specifies the a-entries in the matrix

    b_vec   : np.ndarray of dim (n,) or list
            specifies the b-entries in the matrix

    returns
    -------
    evaluation on the grid a (x) b of the different trees
    """
    if a_vec is None:
        a_vec = np.linspace(0, 1, 11)
    if b_vec is None:
        b_vec = np.linspace(0, 1, 11)
    m = a_vec.size
    n = b_vec.size
    rs = np.random.randint(2 ** 32 - 1)
    if mx_type == "small":
        dim = 4
    elif mx_type in {'gael', 'ronald'}:
        dim = 8
    elif mx_type == 'smith':
        dim = 16
    Z = np.random.normal(size=(100 * dim, dim))
    result = [eval_point(a, b, Z, mx_type=mx_type, random_state=rs,  **kwargs)
              for a in a_vec for b in b_vec]
    return [np.array(zip(*result)[k]).reshape((n, m), order='F')
            for k in range(len(zip(*result)))]


def eval_point(a, b, Z, alpha_tol=1e-2, n_jobs=1, verbose=0, n_iter=10,
               score_norm="KL", CV_norm="ell0", ips_flag=True, mx_type='small',
               random_state=None):
    train_size = .03
    try:
        print "evaluating point ({}, {})".format(a, b)
        Theta = _get_mx(a, b, mx_type=mx_type)
        eigvals, eigvecs = scipy.linalg.eigh(Theta)
        M = eigvecs.dot(np.diag(1. / np.sqrt(eigvals))).dot(eigvecs.T)
        if score_norm != "ell0":
            sqrt_p = np.sqrt(Theta.shape[0])
            Theta = sqrt_p * M
        X = Z.dot(M)
        if mx_type == "small":
            tree1 = [[0, 1], [2, 3]]
            alpha_star_hgl1, LL_hgl1 = \
                covariance_learn.cross_val(X, model_prec=Theta,
                                           method='hgl', htree=tree1,
                                           alpha_tol=alpha_tol, n_iter=n_iter,
                                           train_size=train_size, test_size=.5,
                                           n_jobs=n_jobs, verbose=verbose,
                                           score_norm=score_norm,
                                           CV_norm=CV_norm,
                                           random_state=random_state,
                                           ips_flag=ips_flag)
            tree2 = [[0, 2], [1, 3]]
            alpha_star_hgl2, LL_hgl2 = \
                covariance_learn.cross_val(X, model_prec=Theta,
                                           method='hgl', htree=tree2,
                                           alpha_tol=alpha_tol, n_iter=n_iter,
                                           train_size=train_size, test_size=.5,
                                           n_jobs=n_jobs, verbose=verbose,
                                           score_norm=score_norm,
                                           CV_norm=CV_norm,
                                           random_state=random_state,
                                           ips_flag=ips_flag)
            tree3 = [[0, 3], [1, 2]]
            alpha_star_hgl3, LL_hgl3 = \
                covariance_learn.cross_val(X, model_prec=Theta,
                                           method='hgl', htree=tree3,
                                           alpha_tol=alpha_tol, n_iter=n_iter,
                                           train_size=train_size, test_size=.5,
                                           n_jobs=n_jobs, verbose=verbose,
                                           score_norm=score_norm,
                                           CV_norm=CV_norm,
                                           random_state=random_state,
                                           ips_flag=ips_flag)
        else:
            if mx_type in {'gael', 'ronald'}:
                tree = [[0, 1], [2, 3], [4, 5], [6, 7]]
            else:
                tree = [range(k * 4, (k + 1) * 4) for k in range(4)]
            alpha_star_hgl, LL_hgl = \
                covariance_learn.cross_val(X, model_prec=Theta,
                                           method='hgl', htree=tree,
                                           alpha_tol=alpha_tol, n_iter=n_iter,
                                           train_size=train_size, test_size=.5,
                                           n_jobs=n_jobs, verbose=verbose,
                                           score_norm=score_norm,
                                           CV_norm=CV_norm,
                                           random_state=random_state,
                                           ips_flag=ips_flag)

        alpha_star_gl, LL_gl = \
            covariance_learn.cross_val(X, model_prec=Theta,
                                       method='gl',
                                       alpha_tol=alpha_tol, n_iter=n_iter,
                                       train_size=train_size, test_size=.5,
                                       n_jobs=n_jobs, verbose=verbose,
                                       score_norm=score_norm,
                                       CV_norm=CV_norm,
                                       random_state=random_state,
                                       ips_flag=ips_flag)
        if mx_type == "small":
            print "\thgl1: {}, hgl2: {}, hgl3: {}, gl: {}".format(LL_hgl1[-1],
                                                                  LL_hgl2[-1],
                                                                  LL_hgl3[-1],
                                                                  LL_gl[-1])
            return LL_hgl1[-1], LL_hgl2[-1], LL_hgl3[-1], LL_gl[-1]
        else:
            print "\thgl (alpha={}): {}\n\t gl (alpha={}): {}".format(
                alpha_star_hgl, LL_hgl[-1], alpha_star_gl, LL_gl[-1])
            return LL_hgl[-1], LL_gl[-1]

    except ValueError:
        print "\tinvalid point"
        if mx_type == "small":
            return np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan


def plot_grid(result, extent, ix=[0, -1]):
    if hasattr(ix, '__iter__'):
        f = result[ix[0]] - result[ix[1]]
        f_max = np.nanmax(np.abs(f))
        f_min = -f_max
        cmap = plt.cm.RdBu_r
        cmap2 = plt.cm.copper_r
        title = 'LogLik(true model | data model) [hgl -- gl]'
    else:
        f = result[ix]
        f_max = np.nanmax(f)
        f_min = np.nanmin(f)
        cmap = plt.cm.autumn
        cmap2 = plt.cm.gray
        title = 'LogLik(true model | data model)'

    plt.figure()
    ax = plt.matshow(f, origin='lower', cmap=cmap,
                     interpolation='nearest', extent=extent,
                     fignum=False)
    plt.clim(f_min, f_max)
    cset = plt.contour(f, np.linspace(f_min, f_max, 10), linewidths=2,
                       cmap=cmap2, extent=extent)
    plt.clabel(cset, inline=True, fmt='%1.4f', fontsize=10)
    plt.colorbar(ax)
    plt.title(title)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.show()


def plot_winner_takes_all(result, extent):
    result_ = np.concatenate([res[..., np.newaxis] for res in result], axis=2)
    # add the minimum in first position so for ties -1 is returned
    # add the minimum of trees in second position so tree-ties return 0
    if len(result) > 2:
        result_ = np.concatenate(
            [np.min(result_, axis=2)[..., np.newaxis],
             np.min(result_[..., :-1], axis=2)[..., np.newaxis],
             result_], axis=2)
        f_ix = np.argmax(result_, axis=2) - 1
        f_ix[np.isnan(np.sum(result_, axis=2))] = -2
    else:
        result_ = np.concatenate(
            [np.min(result_, axis=2)[..., np.newaxis],
             result_], axis=2)
        f_ix = np.argmax(result_, axis=2)
        f_ix[np.isnan(np.sum(result_, axis=2))] = -1
    plt.figure()
    ax = plt.matshow(f_ix, origin='lower', cmap=plt.cm.jet, extent=extent,
                     fignum=False)
    plt.clim((-2, 4))
    cset = plt.contour(f_ix, np.linspace(-1.5, 4.5, 7), linewidths=2,
                       cmap=plt.cm.spring, extent=extent)
    plt.clabel(cset, inline=True, fmt='%1.4f', fontsize=10)
    plt.colorbar(ax)
    plt.title('winner takes all')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.show()


if __name__ == "__main__":
    N_JOBS = 1  # 20
    K = 1
    N_ITER = K * N_JOBS
    a_vec = np.linspace(0, 1, 101)
    b_vec = np.linspace(0, 1, 101)
    extent = [(3 * a_vec[0] - a_vec[1]) / 2.,
              (3 * a_vec[-1] - a_vec[-2]) / 2.,
              (3 * b_vec[0] - b_vec[1]) / 2.,
              (3 * b_vec[-1] - b_vec[-2]) / 2.]
    # "score_norm is None" implies log-likelihood of exact model in estimated
    # model ("KL" returns the *negative* of the symmetrised Kullback-Leibler
    # divergence)
    result = eval_grid(n_jobs=N_JOBS, n_iter=N_ITER, a_vec=a_vec, b_vec=b_vec,
                       score_norm='ell0', CV_norm='ell0', ips_flag=False,
                       mx_type='smith')
    result = [-res for res in result]
    fig1 = plot_winner_takes_all(result,  extent)
    fig2 = plot_grid(result, extent, ix=[0, -1])
    fig3 = plot_grid(result, extent, ix=0)
    # TODO : divide not by numel in covariance learn hgl, but by the number of
    # non-zero elements ?
