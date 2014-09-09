import compute_hierarchical_parcellation as chp
import covariance_learn as covl
from joblib import Parallel, delayed
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import infer_penalty as ipen
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def CVpath(subj_dir, cov_learner=None, session=1, n_alphas=51, n_iter=3,
           use_init=True):
    if cov_learner is None:
        cov_learner = covl.GraphLasso(
            alpha=.1,
            base_estimator=EmpiricalCovariance(assume_centered=True))
    X = ipen._get_train_ix(subj_dir, n_iter=n_iter)
    ipen.alpha_max(X)
    score_path = np.zeros((n_alphas,))
    alpha_path = np.logspace(0, -2, n_alphas)
    X_train = [x for x in X if x["session"] == session]
    X_test = np.concatenate([x["data"][x["test_ix"], :]
                             for x in X if x["session"] == 3 - session],
                            axis=0)
    train_ixs = X_train[0]["train_ix"]
    max_alphas = X_train[0]["alpha_max"]
    n_train = len(X_train[0]["train_ix"])
    if use_init:
        Z = [np.identity(X[0]["data"].shape[1]) for _ in range(n_train)]
        U = [np.identity(X[0]["data"].shape[1]) for _ in range(n_train)]
        inits = zip(*([Z, U]))
    else:
        inits = [None] * n_train
    iterators = zip(train_ixs, max_alphas, inits)
    for (ix, alpha) in enumerate(alpha_path):
        results = Parallel(n_jobs=n_iter)(
            delayed(fit_model)(
                np.concatenate([X_train[0]["data"][train_ix, :],
                                X_train[1]["data"][train_ix, :]],
                               axis=0),
                X_test,
                alpha * alpha_max,
                cov_learner,
                use_init=init)
            for train_ix, alpha_max, init in iterators)
        res = zip(*results)
        inits = res[1]
        score_path[ix] = np.mean(res[0])
        iterators = zip(train_ixs, max_alphas, inits)
    return alpha_path, score_path


def fit_model(X_train, X_test, alpha, cov_learner, use_init=None):
    msg = 'fitted model with alpha = {}'.format(alpha)
    cov_learner.__setattr__('alpha', alpha)
    if use_init is not None:
        cov_learner.__setattr__('Zinit', use_init[0])
        cov_learner.__setattr__('Uinit', use_init[1])
    cov_learner.fit(X_train)
    if use_init is not None:
        S = cov_learner.base_estimator.fit(X_train).covariance_
        X = cov_learner.precision_
        Z = cov_learner.auxiliary_prec_
        U = X - Z + (scipy.linalg.inv(X) - S) / cov_learner.rho
        inits = (Z, U)
    else:
        inits = None
    msg += '\ngraphical lasso has set rho to {}'.format(cov_learner.rho_[-1])
    score = cov_learner.score(X_test)
    msg += '\nscore = {} ({} iterations)'.format(score,
                                                 len(cov_learner.f_vals_))
    print msg
    return score, inits


def fista_graph_lasso(X, alpha, L=.01, eta=1.1, Theta_init=None,
                      precision=1e-2, Hessian=False, max_iter=1e2,
                      base_estimator=EmpiricalCovariance(
                          assume_centered=True), fista=False):
    # incorporate the Hessian ...
    t = 1.
    covX = base_estimator.fit(X).covariance_
    p = covX.shape[0]
    penalty_mask = np.ones((p, p)) - np.identity(p)
    alpha = alpha * penalty_mask
    if Theta_init is None:
        Theta = np.identity(p)
    else:
        Theta = Theta_init
    S = covl._cov_2_corr(covX)
    print 'alpha_max = {}'.format(alpha_max(S))
    alpha *= alpha_max(S)
    G = grad_f_graphlasso(Theta, S)
    Z = pL_graphlasso(Theta, alpha, G, L)

#    if Hessian:
#        sqrtS = sqrtm(S)
#        max_eval = np.max(scipy.linalg.schur(
#            sqrtS.dot(Theta.dot(sqrtS)))[0])
#        L = max(L, max_eval)

    f_vals_ = [f_graphlasso(Theta, S) + g_graphlasso(Theta, alpha)]
    print 'F = {} (t = {})'.format(f_vals_[-1], t)

    n_iter = 0
    while True:
        try:
            n_iter += 1
            if not Hessian:
                Theta_ = scipy.linalg.eigh(Theta)
            else:
                Theta_ = Theta
            Z_old = Z.copy()
            G = grad_f_graphlasso(Theta_, S, Hessian=Hessian)
            Z = pL_graphlasso(Theta, alpha, G, L)
            fTheta = f_graphlasso(Theta_, S)
            fZ = f_graphlasso(Z, S)
            while fZ > Q_graphlasso(Z, Theta, G, L, alpha, fTheta,
                                    penalty_mask):
                L *= eta
#                if Hessian:
#                    max_eval = np.max(scipy.linalg.schur(
#                        sqrtS.dot(Theta.dot(sqrtS)))[0])
#                    L = max(L, max_eval)
                Z = pL_graphlasso(Theta, alpha, G, L)
                fZ = f_graphlasso(Z, S)
            if fista:
                t_old = t
                t = 0.5 + np.sqrt(1 + 4 * t ** 2) / 2
                Theta = Z + (t_old - 1.) / t * (Z - Z_old)
            else:
                Theta = Z
            f_vals_.append(f_graphlasso(Theta, S) +
                           g_graphlasso(Theta, alpha))
            print 'F = {} (t = {}, L = {})'.format(f_vals_[-1], t, L)
            if 1. / t < precision or n_iter >= max_iter:
                raise StopIteration
            if len(f_vals_) > 1 and f_vals_[-2] - f_vals_[-1] < precision:
                raise StopIteration
        except StopIteration:
            return Theta, f_vals_


def f_graphlasso(Theta, S):
    if not isinstance(Theta, np.ndarray):
        f = -np.sum(np.log(Theta[0]))
        f += np.diag(Theta[1].T.dot(S).dot(Theta[1])).dot(Theta[0])
        return f
    else:
        return -np.linalg.slogdet(Theta)[1] + np.sum(S * Theta)


def g_graphlasso(Theta, alpha):
    return np.sum(np.abs(Theta * alpha))


def grad_f_graphlasso(Theta, S, Hessian=False):
    if not isinstance(Theta, np.ndarray):
        if not Hessian:
            return S - Theta[1].dot(np.diag(1. / Theta[0])).dot(Theta[1].T)
        else:
            Theta = Theta[1].dot(np.diag(Theta[0])).dot(Theta[1])
            return grad_f_graphlasso(Theta, S, Hessian=Hessian)
    else:
        if not Hessian:
            return S - scipy.linalg.inv(Theta)
        else:
            return Theta.dot(S.dot(Theta)) - Theta


def pL_graphlasso(Theta, alpha, grad, L):
    tmp = Theta - grad / L
    soft_th = np.abs(tmp) - alpha / L
    return np.sign(tmp) * soft_th * (soft_th > 0)


def Q_graphlasso(Z, Theta, grad, L, alpha, f_Theta, pen_mask=None):
    if pen_mask is None:
        pen_mask = np.ones(Theta.shape)
    Q = f_Theta + np.sum((Z - Theta) * grad)
    Q += L / 2 * np.linalg.norm(Theta - Z)
    Q += g_graphlasso(Z, alpha)
    return Q


def sqrtm(X):
    evals, evecs = scipy.linalg.eigh(X)
    return evecs.dot(np.diag(np.sqrt(evals)).dot(evecs.T))


def _check_mx(X, props='pos_sym_2d_sq'):
    msg = ''
    if 'pos' in props and np.min(np.diag(scipy.linalg.schur(X)[0])) < 0.:
        msg += 'matrix is not positive definite'
    if 'sym' in props:
        try:
            np.assert_array_almost_equal(X, X.T)
        except AssertionError:
            if msg:
                msg += ', '
            msg += 'matrix is not symmetric'
    if '2d' in props and X.ndim != 2:
        if msg:
            msg += ', '
        msg += 'matrix has to many dimensions'
    if 'sq' in props and not np.allclose(X.shape - X.shape[0],
                                         np.zeros((X.ndim,))):
        if msg:
            msg += ', '
        msg += 'matrix is not square'
    if msg:
        raise ValueError(msg)


def alpha_max(X):
    p = X.shape[0]
    X[::p + 1] = 0.
    return np.max(np.abs(X[:]))


if __name__ == '__main__':
    import cProfile
    import pstats

    gl = covl.GraphLasso(
        alpha=.4,
        # base_estimator=LedoitWolf(assume_centered=True),
        base_estimator=EmpiricalCovariance(assume_centered=True),
        mu=2., tol=1e-4, rho=1.)

    pr1 = cProfile.Profile()
    subj_dir = chp.subject_dirs[0]
    pr1.enable()
    alpha_path, score_path = CVpath(subj_dir, n_alphas=11, n_iter=3)
    pr1.disable()
    with open("results1.txt", 'w') as s1:
        sortby = 'cumulative'
        ps = pstats.Stats(pr1, stream=s1).sort_stats(sortby)
        ps.print_stats()

    plt.semilogx(alpha_path, score_path)

    pr2 = cProfile.Profile()
    pr2.enable()
    alpha_path, score_path = CVpath(subj_dir, n_alphas=11, n_iter=3,
                                    use_init=False)
    pr2.disable()
    with open("results2.txt", 'w') as s2:
        sortby = 'cumulative'
        ps = pstats.Stats(pr2, stream=s2).sort_stats(sortby)
        ps.print_stats()

    plt.semilogx(alpha_path, score_path)
    plt.show()
