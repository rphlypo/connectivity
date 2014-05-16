import logging
import sys
import numpy as np
import sklearn.utils.extmath
import copy
import numbers


from htree import HTree
from scipy import linalg
from sklearn.base import clone
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))
fast_logdet = sklearn.utils.extmath.fast_logdet


class GraphLasso(EmpiricalCovariance):
    """ the estimator class for GraphLasso based on ADMM

    arguments
    ---------
        alpha: scalar or postive matrix
            the penalisation parameter

        tol: scalar
            tolerance to declare convergence

        max_iter: unsigned int
            maximum number of iterations until convergence

        verbose: unsigned int
            set to 0 for no verbosity
            see logger.setLevel for more information

        base_estimator: instance of covariance estimator class
            this estimator will be used to estimate the covariance from the
            data both in fit and score

        scale_2_corr: boolean
            whether correlation or covariance is to be used

        rho: positive scalar
            ressemblance enforcing penalty between split variables
    """
    def __init__(self, alpha, tol=1e-6, max_iter=100, verbose=0,
                 base_estimator=None,
                 scale_2_corr=True, rho=1., mu=None, score_norm=None):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.scale_2_corr = scale_2_corr
        self.rho = rho
        self.mu = mu
        self.score_norm = score_norm
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        S = self._X_to_cov(X)
        precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_gl(S, self.alpha, rho=self.rho, tol=self.tol,
                     max_iter=self.max_iter)

        self.precision_ = precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self

    def _X_to_cov(self, X):
        if self.base_estimator is None:
            self.base_estimator_ = EmpiricalCovariance(assume_centered=True)
        else:
            self.base_estimator_ = clone(self.base_estimator)
        logger.setLevel(self.verbose)
        S = self.base_estimator_.fit(X).covariance_
        if self.scale_2_corr:
            S = _cov_2_corr(S)
        return S

    def score(self, X_test, y=None):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used, present for API consistence purpose.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        # compute empirical covariance of the test set
        if self.score_norm is not None:
            return self._error_norm(X_test, norm=self.score_norm)
        else:
            test_cov = self.base_estimator_.fit(X_test).covariance_
            if self.scale_2_corr:
                test_cov = _cov_2_corr(test_cov)
            # compute log likelihood
            return log_likelihood(self.precision_, test_cov)

    def _error_norm(self, X_test, norm="Fro"):
        """Computes the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm).

        Parameters
        ----------
        comp_cov : array-like, shape = [n_features, n_features]
            The covariance to compare with.

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'Fro' (default): sqrt(trace(A.T.dot(A)))
            - 'spectral': sqrt(max(eigenvalues(A.T.dot(A)))
            - 'geodesic':
                sum(log(eigenvalues(model_precision.dot(test_covariance))))
            - 'invFro': sqrt(trace(B.T.dot(B)))
            - 'KL':
                (-log(det(test_covariance)) - log(det(model_precision))) / 2
            where A is the error ``(test_covariance - model_covariance)``
            and   B is the error ``(test_precision - model_precision)``
        Returns
        -------
        A distance measuring the divergence between the model and the test set

        """
        test_cov = self.base_estimator_.fit(X_test).covariance_
        if self.scale_2_corr:
            test_cov = _cov_2_corr(test_cov)
        # compute the error norm
        if norm == "frobenius":
            # compute the error
            error = test_cov - self.covariance_
            error_norm = np.sqrt(np.sum(error ** 2))
        elif norm == "spectral":
            # compute the error
            error = test_cov - self.covariance_
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
            error_norm = np.sqrt(squared_norm)
        elif norm == "geodesic":
            eigvals = linalg.eigvals(self.covariance_, test_cov)
            error_norm = np.sum(np.log(eigvals) ** 2) ** (1. / 2)
        elif norm == "invFro":
            error = linalg.inv(test_cov) - self.precision_
            error_norm = np.sqrt(np.sum(error ** 2))
        elif norm == "KL":
            # test_cov is the target model
            # self.precision_
            error_norm = np.linalg.slogdet(test_cov)[1]
            error_norm += np.linalg.slogdet(self.precision_)[1]
            error_norm -= self.precision_.shape[0]
            error_norm += np.linalg.trace(linalg.inv(
                test_cov.dot(self.precision_)))
            error_norm /= 2.
        else:
            raise NotImplementedError(
                "Only the following norms are implemented:\n"
                "spectral, Frobenius, inverse Frobenius, and geodesic")
        return error_norm


class IPS(GraphLasso):
    """ the estimator class for GraphLasso based on ADMM
    """
    def __init__(self, support, tol=1e-6, max_iter=100, verbose=0,
                 base_estimator=None,
                 scale_2_corr=True, rho=1., mu=None, score_norm=None):
        self.support = support
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.scale_2_corr = True
        self.rho = rho
        self.mu = mu
        self.score_norm = score_norm
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        S = self._X_to_cov(X)
        precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_ips(S, self.support, rho=self.rho, tol=self.tol,
                      max_iter=self.max_iter)

        self.precision_ = precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self


class HierarchicalGraphLasso(GraphLasso):
    def __init__(self, htree, alpha, tol=1e-6, max_iter=1e4, verbose=0,
                 base_estimator=None, scale_2_corr=True, rho=1., mu=None,
                 score_norm=None, n_jobs=1):
        """ hierarchical version of graph lasso with ell1-2 penalty

        extra arguments
        ---------------
        htree: an instance of HTree
            defines the hierarchical structure over which the objective
            function is to be optimised
        """
        self.htree = htree
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.scale_2_corr = True
        self.rho = rho
        self.mu = mu
        self.score_norm = score_norm
        self.n_jobs = n_jobs
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        S = self._X_to_cov(X)
        precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_hgl(S, self.htree, self.alpha, rho=self.rho, tol=self.tol,
                      mu=self.mu, max_iter=self.max_iter)

        self.precision_ = precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self


def _admm_gl(S, alpha, rho=1., tau_inc=2., tau_decr=2., mu=None, tol=1e-6,
             max_iter=100, Xinit=None):
    p = S.shape[0]
    Z = (1 + rho) * np.identity(p)
    U = np.zeros((p, p))
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    if isinstance(alpha, numbers.Number):
        alpha = alpha * (np.ones((p, p)) - np.identity(p))
    r_ = list()
    s_ = list()
    f_vals_ = list()
    r_.append(linalg.norm(X - Z) / (p ** 2))
    s_.append(np.inf)
    f_vals_.append(_pen_neg_log_likelihood(X, S, alpha))
    iter_count = 0
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            eigvals, eigvecs = linalg.eigh(rho * (Z - U) - S)
            eigvals_ = np.diag(eigvals + (eigvals ** 2 + 4 * rho) ** (1. / 2))
            X = eigvecs.dot(eigvals_.dot(eigvecs.T)) / (2 * rho)
            # proximal operator for Z: soft thresholding
            Z = np.sign(X + U) * np.max(
                np.reshape(np.concatenate((np.abs(X + U) - alpha / rho,
                                           np.zeros((p, p))), axis=1),
                           (p, p, -1), order="F"), axis=2)
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / (p ** 2))
            s_.append(linalg.norm(Z - Z_old) / (p ** 2))
            f_vals_.append(_pen_neg_log_likelihood(X, S, alpha))

            if mu is not None:
                Y = U * rho  # this is the unscaled Y
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
                U = Y / rho  # newly scaled dual variable
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return Z, r_, s_, f_vals_


def _admm_ips(S, support, rho=1., tau_inc=2., tau_decr=2., mu=None, tol=1e-6,
              max_iter=100, Xinit=None):
    """
    returns:
    -------
    Z       : numpy.ndarray
        the split variable with correct support

    r_      : list of floats
        normalised norm of difference between split variables

    s_      : list of floats
        convergence of the variable Z in normalised norm
        normalisation is based on division by the number of elements
    """
    p = S.shape[0]
    dof = np.count_nonzero(support)
    Z = (1 + rho) * np.identity(p)
    U = np.zeros((p, p))
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    r_ = list()
    s_ = list()
    f_vals_ = list()
    r_.append(linalg.norm(X - Z) / dof)
    s_.append(np.inf)
    f_vals_.append(_pen_neg_log_likelihood(X, S))
    iter_count = 0
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            eigvals, eigvecs = linalg.eigh(rho * (Z - U) - S)
            eigvals_ = np.diag(eigvals + (eigvals ** 2 + 4 * rho) ** (1. / 2))
            X = eigvecs.dot(eigvals_.dot(eigvecs.T)) / (2 * rho)
            # proximal operator for Z: projection on support
            Z = support * (X + U)
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / (p ** 2))
            s_.append(linalg.norm(Z - Z_old) / dof)
            f_vals_.append(_pen_neg_log_likelihood(X, S))

            if mu is not None:
                Y = U * rho  # this is the unscaled Y
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
                U = Y / rho  # newly scaled dual variable
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return Z, r_, s_, f_vals_


def _admm_hgl(S, htree, alpha, rho=1., tau_inc=1.1, tau_decr=1.1, mu=None,
              tol=1e-6, max_iter=1e2, Xinit=None):
    """
    returns:
    -------
    Z       : numpy.ndarray
        the split variable with correct support

    r_      : list of floats
        normalised norm of difference between split variables

    s_      : list of floats
        convergence of the variable Z in normalised norm
        normalisation is based on division by the number of elements
    """
    # TODO foresee option to give a list of alphas, one for each level
    p = S.shape[0]
    Z = (1. + rho) / rho * np.identity(p)
    U = np.zeros((p, p))
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    r_ = list()
    s_ = list()
    f_vals_ = list()
    r_.append(linalg.norm(X - Z))
    s_.append(np.inf)
    iter_count = 0
    # if a (nested) list is given, create the tree
    if hasattr(htree, '__iter__'):
        tree_list = copy.deepcopy(htree)
        htree = HTree()
        htree.tree(tree_list)
        # {htree}._update() is ok for small trees, otherwise use on-the-fly
        # evaluation with {node}._get_node_values() at each node call
        htree._update()
    # this returns an ordered list from leaves to root nodes
    nodes_levels = htree.root_.get_descendants()
    nodes_levels.sort(key=lambda x: x[1])
    nodes_levels.reverse()
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            eigvals, eigvecs = linalg.eigh(rho * (Z - U) - S)
            eigvals_ = np.diag(eigvals + (eigvals ** 2 + 4 * rho) ** (1. / 2))
            X = eigvecs.dot(eigvals_.dot(eigvecs.T)) / (2 * rho)
            # smooth functional score
            f_vals_.append(-np.linalg.slogdet(X)[1] + np.sum(X * S))
            # proximal operator for Z: projection on support
            Z = U + X
            # TODO for a given level we could evaluate all in parallel!
            for (node, level) in nodes_levels:
                if node.complement() is None:
                    continue
                ix = node.evaluate()
                for node_c in node.complement():
                    ixc = node_c.evaluate()
                    B = Z[np.ix_(ix, ixc)]
                    alpha_ = alpha * np.sqrt(np.size(B))
                    f_vals_[-1] = f_vals_[-1] + \
                        alpha_ * np.linalg.norm(X[np.ix_(ix, ixc)])
                    if np.linalg.norm(B):
                        # needs np.linalg.norm(B) / np.sqrt(np.size(B)) and not
                        # np.linalg.norm(B) so as to be independent of size !
                        multiplier = (1. - alpha_ / (rho * np.linalg.norm(B)))
                        multiplier = max(0., multiplier)
                        B *= multiplier
                        Z[np.ix_(ix, ixc)] = B
                        Z[np.ix_(ixc, ix)] = B.T
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / np.sqrt(p ** 2))
            s_.append(linalg.norm(Z - Z_old))

            if mu is not None:
                Y = U * rho  # this is the unscaled Y
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
                U = Y / rho  # newly scaled dual variable
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return Z, r_, s_, f_vals_


def _check_convergence(X, Z, Z_old, U, rho, tol_abs=1e-12, tol_rel=1e-6):
    p = np.size(U)
    n = np.size(X)
    tol_primal = np.sqrt(p) * tol_abs + tol_rel * max([np.linalg.norm(X),
                                                       np.linalg.norm(Z)])
    tol_dual = np.sqrt(n) * tol_abs / rho + tol_rel * np.linalg.norm(U)
    return (np.linalg.norm(X - Z) < tol_primal and
            np.linalg.norm(Z - Z_old) < tol_dual)


def _pen_neg_log_likelihood(X, S, A=None):
    log_likelihood = - np.linalg.slogdet(X)[1] + np.sum((X * S).flat)
    if A is not None:
        log_likelihood += np.sum((X * A).flat)
    return log_likelihood


def log_likelihood(precision, covariance):
    p = precision.shape[0]
    log_likelihood_ = np.linalg.slogdet(precision)[1]
    log_likelihood_ -= np.sum(precision * covariance)
    log_likelihood_ -= p * np.log(2 * np.pi)
    return log_likelihood_ / 2.


def _cov_2_corr(covariance):
    p = covariance.shape[0]
    scale = np.diag(covariance.flat[::p + 1] ** (-1. / 2))
    correlation = scale.dot(covariance.dot(scale))
    # guarantee symmetry
    return (correlation + correlation.T) / 2.


def cross_val(X, method='gl', alpha_tol=1e-4,
              n_iter=100, train_size=.1, test_size=.5,
              model_prec=None, verbose=0, n_jobs=1,
              **kwargs):
    from sklearn import cross_validation
    from joblib import Parallel, delayed
    # logging.ERROR is at level 40
    # logging.WARNING is at level 30, everything below is low priority
    # logging.INFO is at level 20, verbose 10
    # logging.DEBUG is at level 10, verbose 20
    logger.setLevel(logging.WARNING - verbose)
    bs = cross_validation.Bootstrap(X.shape[0], n_iter=n_iter,
                                    train_size=train_size,
                                    test_size=test_size)
    if method == 'gl':
        cov_learner = GraphLasso
    elif method == 'hgl':
        cov_learner = HierarchicalGraphLasso
    elif method == 'ips':
        cov_learner = IPS
    if model_prec is not None:
        sqrt_p = np.sqrt(model_prec.shape[0])
        eigvals, eigvecs = linalg.eigh(model_prec)
        model_prec = eigvecs.dot(np.diag(1. / np.sqrt(eigvals))).dot(eigvecs.T)
        model_prec *= sqrt_p

    # alpha_max ?
    alphas = np.linspace(0., 1., 5)
    LL = np.zeros((5,))
    LL_ = list()
    for (ix, alpha) in enumerate(alphas):
        cov_learner_ = cov_learner(alpha=alpha, **kwargs)
        res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
            X, train_ix, test_ix, model_prec, cov_learner_)
            for train_ix, test_ix in bs)
        LL[ix] = np.mean(np.array(res_))
    LL_.append(LL[2])
    while True:
        try:
            max_ix = min(max(np.argmax(LL), 1), 3)
            LL[0] = LL[max_ix - 1]
            LL[4] = LL[max_ix + 1]
            LL[2] = LL[max_ix]
            LL[1] = LL[3] = 0.
            alphas = np.linspace(alphas[max_ix - 1], alphas[max_ix + 1], 5)
            if alphas[-1] - alphas[0] < alpha_tol:
                raise StopIteration
            logger.info("refining alpha grid to interval [{}, {}]".format(
                alphas[0], alphas[-1]))
            for (ix, alpha) in enumerate(alphas[[1, 3]]):
                cov_learner_ = cov_learner(alpha=alpha, **kwargs)
                res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
                    X, train_ix, test_ix, model_prec, cov_learner_)
                    for train_ix, test_ix in bs)
                ix_ = 2 * ix + 1
                LL[ix_] = np.mean(np.array(res_))
            LL_.append(LL[2])
        except StopIteration:
            return alphas[2], LL_


def _eval_cov_learner(X, train_ix, test_ix, model_prec, cov_learner):
    X_train = X[train_ix, ...]
    if model_prec is None:
        X_test = X[test_ix, ...]
    else:
        X_test = model_prec
    score = cov_learner.fit(X_train).score(X_test)
    return score
