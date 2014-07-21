import compute_hierarchical_parcellation as chp
import covariance_learn as covl
from joblib import Parallel, delayed
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
import infer_penalty as ipen
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


gl = covl.GraphLasso(alpha=.4, base_estimator=LedoitWolf(assume_centered=True),
                     mu=2., tol=1e-3, rho=1.)


def CVpath(subj_dir, cov_learner=gl, session=1, n_alphas=51, n_iter=3,
           use_init=True):
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
    msg += '\ngraphical lasso has set rho to {}'.format(cov_learner.rho)
    score = cov_learner.score(X_test)
    msg += '\nscore = {}'.format(score)
    print msg
    return score, inits


def fista_graph_lasso(X, alpha, L=.1, eta=1.1, Theta_init=None,
                      precision=1e-6):
    t = 1.
    empcov = EmpiricalCovariance(assume_centered=True)
    covX = empcov.fit(X).covariance_
    p = covX.shape[0]
    penalty_mask = np.ones((p, p)) - np.identity(p)
    alpha = alpha * penalty_mask
    if Theta_init is None:
        Theta = np.identity(p)
    else:
        Theta = Theta_init
    S = covl._cov_2_corr(covX)
    G = grad_f_graphlasso(Theta, S)
    Z = pL_graphlasso(Theta, alpha, G, L)

    while True:
        try:
            Theta_ = scipy.linalg.eigh(Theta)
            Z_old = Z.copy()
            G = grad_f_graphlasso(Theta_, S)
            Z = pL_graphlasso(Theta, alpha, G, L)
            fTheta = f_graphlasso(Theta_, S)
            fZ = f_graphlasso(Z, S)
            while fZ > Q_graphlasso(Z, Theta_, G, L, alpha, fTheta,
                                    penalty_mask):
                L *= eta
                Z = pL_graphlasso(Theta, alpha, G, L)
                fZ = f_graphlasso(Z, S)
                t_old = t
                t = 0.5 + np.sqrt(1 + 4 * t ** 2) / 2
            Theta = Z + (t_old - 1) / t * (Z - Z_old)
            if t < precision:
                raise StopIteration
        except StopIteration:
            return Theta


def f_graphlasso(Theta, S):
    if not isinstance(Theta, np.ndarray):
        f = -np.sum(np.log(Theta[0]))
        f += np.diag(Theta[1].T.dot(S).dot(Theta[1])).dot(Theta[0])
        return f
    else:
        return -np.linalg.slogdet(Theta)[1] + np.sum(S * Theta)


def g_graphlasso(Theta, alpha):
    return np.sum(np.abs(Theta * alpha))


def grad_f_graphlasso(Theta, S):
    if not isinstance(Theta, np.ndarray):
        return S - Theta[1].dot(np.diag(1. / Theta[0])).dot(Theta[1].T)
    else:
        return S - scipy.linalg.inv(Theta)


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


if __name__ == '__main__':
    import cProfile
    import pstats
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
