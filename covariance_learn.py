import logging
reload(logging)
# import sys
import numpy as np
import copy
import numbers

from scipy.ndimage.measurements import mean as label_mean
from scipy.special import gamma as gamma_func
from scipy import linalg
from sklearn.base import clone
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
import sklearn.utils.extmath
from functools import partial
from joblib import Parallel, delayed

from htree import HTree
from htree import _check_htree

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
fh = logging.FileHandler('cvl.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

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

    def __init__(self, alpha, tol=1e-6, max_iter=1e4, verbose=0,
                 base_estimator=EmpiricalCovariance(assume_centered=True),
                 scale_2_corr=True, rho=1., mu=None,
                 score_norm='loglikelihood'):
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
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_, rho_ =\
            _admm_gl(S, self.alpha, rho=self.rho, tol=self.tol,
                     max_iter=self.max_iter, mu=self.mu, **kwargs)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        self.rho_ = rho_
        return self

    def _X_to_cov(self, X):
        self.base_estimator_ = clone(self.base_estimator)
        logger.setLevel(self.verbose)
        S = self.base_estimator_.fit(X).covariance_
        if self.scale_2_corr:
            S = _cov_2_corr(S)
        return S

    def score(self, X_test, y=None):
        """Computes the log-likelihood or an error_norm

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X_test is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y : not used, present for API consistency purposes.

        Returns
        -------
        res : float
            log-likelihood or error norm

        """
        # compute empirical covariance of the test set
        if (self.score_norm != 'loglikelihood' and
                self.score_norm is not None):
            return self._error_norm(X_test, norm=self.score_norm)
        else:
            test_cov = self.base_estimator_.fit(X_test).covariance_
            if self.scale_2_corr:
                test_cov = _cov_2_corr(test_cov)
            # compute log likelihood
            return log_likelihood(self.precision_, test_cov)

    def _error_norm(self, X_test, norm="Fro", **kwargs):
        """Computes an error between a covariance and its estimator

        Parameters
        ----------
        X_test : array_like, shape = [n_samples, n_features]
            Data for testing the method, could be the model itself

        norm : str
            The type of norm used to compute the error. Available error types:
            - 'Fro' (default): sqrt(trace(A.T.dot(A)))
            - 'spectral': sqrt(max(eigenvalues(A.T.dot(A)))
            - 'geodesic':
                sum(log(eigenvalues(model_precision.dot(test_covariance))))
            - 'invFro': sqrt(trace(B.T.dot(B)))
            - 'KL': actually Jensen's divergence (symmetrised KL)
            - 'bregman': (-log(det(Theta.dot(S))) + trace(Theta.dot(S)) - p)/2
            - 'ell0': ||B||_0 =  sum(XOR(test_precision, model_precision))/2
                related to accuracy
            where A is the error ``(test_covariance - model_covariance)``
            and   B is the error ``(test_precision - model_precision)``

        keyword arguments can be passed to the different error computations

        Returns
        -------
        A distance measuring the divergence between the model and the test set

        """
        if norm != "ell0":
            test_cov = self.base_estimator_.fit(X_test).covariance_
            if self.scale_2_corr:
                test_cov = _cov_2_corr(test_cov)

        # compute the error norm
        if norm == "frobenius":
            error = test_cov - self.covariance_
            error_norm = np.sqrt(np.sum(error ** 2))
        elif norm == "spectral":
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
            # self.precision_ is the trained data model
            # KL is symmetrised (Jeffreys divergence)
            error_norm = -self.precision_.shape[0]
            error_norm += np.trace(linalg.inv(
                test_cov.dot(self.precision_))) / 2.
            error_norm += np.trace(
                test_cov.dot(self.precision_)) / 2.
        elif norm == "ell0":
            # X_test acts as a mask
            error_norm = self._support_recovery_norm(X_test, **kwargs)
        elif norm == "bregman":
            test_mx = test_cov.dot(self.precision_)
            # negative log-det bregman divergence
            error_norm = - np.linalg.slogdet(test_mx)[1]
            error_norm += np.sum(test_mx) - test_mx.shape[0]
            error_norm /= 2.
        else:
            raise NotImplementedError(
                "Only the following norms are implemented:\n"
                "spectral, Frobenius, inverse Frobenius, geodesic, KL, ell0, "
                "bregman")
        return error_norm

    def _support_recovery_norm(self, X_test, relative=False):
        """ accuracy related error pseudo-norm

        Parameters
        ----------
        X_test : positive-definite, symmetric numpy.ndarray of shape (p, p)
            the target precision matrix

        relative: boolean
            whether the error is given as a percentage or as an absolute
            number of counts


        Returns
        -------
        ell0 pseudo-norm between X_test and the estimator

        """
        if relative:
            p = X_test.shape[0]
            c = p * (p - 1)
        else:
            c = 2.
        return np.sum(np.logical_xor(
            np.abs(self.auxiliary_prec_) > machine_eps(0),
            np.abs(X_test) > machine_eps(0))) / c


class IPS(GraphLasso):

    """ the estimator class for GraphLasso based on ADMM
    """

    def __init__(self, support, tol=1e-6, max_iter=100, verbose=0,
                 base_estimator=EmpiricalCovariance(assume_centered=True),
                 scale_2_corr=True, rho=1., mu=None,
                 score_norm='loglikelihood'):
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
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_, rho_ =\
            _admm_ips(S, self.support, rho=self.rho, tol=self.tol,
                      max_iter=self.max_iter, **kwargs)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        self.rho_ = rho_
        return self


class HierarchicalGraphLasso(GraphLasso):

    def __init__(self, htree, alpha, tol=1e-6, max_iter=1e4, verbose=0,
                 base_estimator=EmpiricalCovariance(assume_centered=True),
                 scale_2_corr=True, rho=1., mu=None,
                 score_norm='loglikelihood', n_jobs=1, alpha_func=None):
        """ hierarchical version of graph lasso with ell1-2 penalty

        arguments (complimentary to GraphLasso)
        ---------
        htree   : embedded lists or HTree object
            specifies data organisation in 'communities'

        alpha_func : a functional taking alpha and level as arguments
            this function makes it possible to adapt 'alpha' to the level
            of evaluation in the tree

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
        self.alpha_func = alpha_func
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        S = self._X_to_cov(X)

        self.htree_ = _check_htree(self.htree)
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_, rho_ =\
            _admm_hgl2(S, self.htree_, self.alpha, rho=self.rho, tol=self.tol,
                       mu=self.mu, max_iter=self.max_iter,
                       alpha_func=self.alpha_func, **kwargs)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        self.rho_ = rho_
        return self


def _admm_gl(S, alpha, rho=1., tau_inc=2., tau_decr=2., mu=None, tol=1e-6,
             max_iter=100, Xinit=None, Zinit=None, Uinit=None):
    p = S.shape[0]
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    if Zinit is None:
        Z = np.identity(p)
    else:
        Z = Zinit
    if Uinit is None:
        U = X - Z
    else:
        U = Uinit
    if isinstance(alpha, numbers.Number):
        alpha = alpha * (np.ones((p, p)) - np.identity(p))
    r_ = list()
    s_ = list()
    f_vals_ = list()
    rho_ = [rho]
    iter_count = 0
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            X, eig_vals = _update_X(S, Z, -U, rho)
            func_val = -np.sum(np.log(eig_vals)) + np.sum(S * X)
            func_val += np.sum(alpha * np.abs(X))
            # proximal operator for Z: soft thresholding
            tmp = np.abs(X - U) - alpha / rho
            Z = np.sign(X - U) * tmp * (tmp > 0.)
#           Z = np.sign(X + U) * np.max(
#               np.reshape(np.concatenate((np.abs(X + U) - alpha / rho,
#                                          np.zeros((p, p))), axis=1),
#                          (p, p, -1), order="F"), axis=2)
            # update scaled dual variable
            U = U + Z - X
            r_.append(linalg.norm(X - Z) / (p ** 2))
            s_.append(linalg.norm(Z - Z_old) / (p ** 2))
            f_vals_.append(func_val)

            if mu is not None:
                rho = _update_rho(U, rho, r_[-1], s_[-1],
                                  mu, tau_inc, tau_decr)
            rho_.append(rho)
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return X, Z, r_, s_, f_vals_, rho_


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
    r_.append(linalg.norm(X - Z))
    s_.append(np.inf)
        normalisation is based on division by the number of elements
    """
    p = S.shape[0]
    dof = np.count_nonzero(support)
    Z = (1 + rho) / rho * np.identity(p)
    U = np.zeros((p, p))
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    r_ = list()
    s_ = list()
    f_vals_ = list()
    rho_ = [rho]
    r_.append(linalg.norm(X - Z) / dof)
    s_.append(np.inf)
    f_vals_.append(_pen_neg_log_likelihood(X, S))
    iter_count = 0
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            X, _ = _update_X(S, Z, U, rho)
            # proximal operator for Z: projection on support
            Z = support * (X + U)
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / (p ** 2))
            s_.append(linalg.norm(Z - Z_old) / dof)
            func_val = -np.linalg.slogdet(support * X)[1] + \
                np.sum(S * X * support)
            f_vals_.append(func_val)

            if mu is not None:
                rho = _update_rho(U, rho, r_[-1], s_[-1],
                                  mu, tau_inc, tau_decr)
                rho_.append(rho)
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return X, Z, r_, s_, f_vals_, rho_


def _admm_hgl2(S, htree, alpha, rho=1., tau_inc=1.1, tau_decr=1.1, mu=None,
               tol=1e-6, max_iter=1e2, Xinit=None, alpha_func=None):
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
    Z = (1. + rho) / rho * np.identity(p)
    U = np.zeros((p, p))
    if Xinit is None:
        X = np.identity(p)
    else:
        X = Xinit
    r_ = list()
    s_ = list()
    f_vals_ = list()
    rho_ = [rho]
    iter_count = 0
    # this returns an ordered list from leaves to root nodes
    nodes_levels = htree.root_.get_descendants()
    max_level = htree.get_depth()
    # all leave node values, do not sort (would break data representation)
    node_list = np.array(htree.root_.value_)

    if alpha_func is None:
        # alpha_func = lambda alpha, level: alpha
        alpha_func = partial(_alpha_func, h=.5, max_level=max_level)

    Labels = np.zeros((p, p, max_level), dtype=np.int)
    for level in np.arange(max_level):
        label = 0
        # filter nodes at a given level (0-th layer is 1st level!)
        node_set = [node for (node, lev) in nodes_levels
                    if lev == level + 1]
        for (ix1, node1) in enumerate(node_set[:-1]):
            for node2 in node_set[ix1 + 1:]:
                label += 1
                # find the index of the nodes w.r.t. order at "root_"
                ix = [np.where(node_list == v)[0][0]
                      for v in node1.value_]
                ixc = [np.where(node_list == v)[0][0]
                       for v in node2.value_]
                Labels[np.ix_(ix, ixc, [level])] = label
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            X, eig_vals = _update_X(S, Z, U, rho)
            # smooth functional score
            func_val = -np.sum(np.log(eig_vals)) + np.sum(X * S)
            # proximal operator for Z: block norm soft thresholding
            Z = U + X

            for level in np.arange(max_level, 0, -1):
                # initialise alpha for given level
                alpha_ = alpha_func(alpha, level)
                # print "alpha(level = {}) = {}".format(level, alpha_)
                if alpha_ < machine_eps(0):
                    continue
                logger.info("alpha(level = {}) = {}".format(level, alpha_))
                # get all nodes at specified level
                L = Labels[..., level - 1]
                multipliers = np.zeros((len(np.unique(L)) - 1,))
                norms_ = np.sqrt(label_mean(Z ** 2, labels=L,
                                            index=np.unique(L[L > 0])))
                Xnorms = np.sqrt(label_mean(X ** 2, labels=L,
                                            index=np.unique(L[L > 0])))
                # might need some 'limit'-behaviour, i.e., eps / eps = 1
                # tmp = rho * norms_ - alpha_
                # multipliers[tmp > 0] = tmp[tmp > 0] / (tmp[tmp > 0] + alpha)
                multipliers[norms_ > 0] = \
                    1. - alpha_ / (rho * norms_[norms_ > 0])
                multipliers = (multipliers > 0) * multipliers
                # the next line is necessary to maintain diagonal blocks
                multipliers = np.concatenate((np.array([1.]), multipliers))
                Z = multipliers[L + L.T] * Z
                func_val += 2 * alpha_ * np.sum(Xnorms)
            f_vals_.append(func_val)
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / np.sqrt(p ** 2))
            s_.append(linalg.norm(Z - Z_old) / np.sqrt(p ** 2))
            if mu is not None:
                rho = _update_rho(U, rho, r_[-1], s_[-1],
                                  mu, tau_inc, tau_decr)
                rho_.append(rho)
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return X, Z, r_, s_, f_vals_, rho_


def _update_X(S, Z, U, rho):
    eig_vals, eig_vecs = linalg.eig(rho * (Z - U) - S)
    eig_vals = np.real(eig_vals / 2.)
    eig_vecs = np.real(eig_vecs)
    eig_vals = (eig_vals + (eig_vals ** 2 + rho) ** (1. / 2)) / rho
    return (eig_vecs * eig_vals).dot(eig_vecs.T), eig_vals


def _check_convergence(X, Z, Z_old, U, rho, tol_abs=1e-12, tol_rel=1e-6):
    p = np.size(U)
    n = np.size(X)
    tol_primal = np.sqrt(p) * tol_abs + tol_rel * max([np.linalg.norm(X),
                                                       np.linalg.norm(Z)])
    tol_dual = np.sqrt(n) * tol_abs / rho + tol_rel * np.linalg.norm(U)
    return (np.linalg.norm(X - Z) < tol_primal and
            np.linalg.norm(Z - Z_old) < tol_dual)


def _update_rho(U, rho, r, s, mu, tau_inc=2., tau_decr=2.):
    if r > mu * s:
        rho *= tau_inc
        U /= tau_inc
    elif s > mu * r:
        rho /= tau_decr
        U *= tau_decr
    # U is changed inplace, no need for returning it
    return rho


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
    scale = np.atleast_2d(np.sqrt(covariance.flat[::p + 1]))
    correlation = covariance / scale / scale.T
    # guarantee symmetry
    return (correlation + correlation.T) / 2.


def cross_val(X, y=None, X_test=None, y_test=None, method='gl', alpha_tol=0.01,
              verbose=0, n_jobs=1, ips_flag=False, htree=None, score_norm=None,
              h=None, base_estimator=None, tol=1e-4, **kwargs):
    """
    if one has a theoretical optimal covariance or precision matrix
    rather than testing data, one may use the next trick

        Suppose the theoretical optimum (covariance matrix) is Theta

        >>> eig_vals, eig_vecs = scipy.linalg.eig(Theta)
        >>> X_test = np.sqrt(eig_vals[:, np.newaxis]) * eigvecs.T

        One sees that X_test.T.dot(X_test) = Theta, if needed one
        could scale

        >>> X_test *= sqrt(p)

        for which X_test.T.dot(X_test) / p = Theta

        use the function _cov_2_data

        One should pay attention, though, since the estimator should be
        EmpiricalCovariance with `assume_centered` = `True`, since any biased
        estimator will use n=p as the number of samples and hence heavily
        bias the solution.
    """
    # logging.ERROR is at level 40
    # logging.WARNING is at level 30, everything below is low priority
    # logging.INFO is at level 20, verbose 10
    # logging.DEBUG is at level 10, verbose 20
    # logger.setLevel(logging.WARNING - verbose)

    # kwargs contains n_train, n_test, n_retest, random_state, n_iter
    shuffle_split = _get_indices(X, y, X_test, y_test, **kwargs)
    logger.debug("prepared indices")

    # selecting the method for learning the covariance method
    if method == 'gl':
        cov_learner = GraphLasso
    elif method == 'hgl':
        cov_learner = HierarchicalGraphLasso
    elif method == 'ips':
        cov_learner = IPS
    logger.debug("chose learner : {}".format(cov_learner))
    logger.debug("chose base_estimator : {}".format(base_estimator))

    # get max_level ifi htree is given
    if htree is not None:
        max_level = HTree(htree).get_depth()

    # initialisation
    if base_estimator is None:
        base_estimator = EmpiricalCovariance(assume_centered=True)
        print 'Chose EmpiricalCovariance as base_estimator'
    alphas = np.linspace(0., 1., 5)
    h_ = np.zeros((5,))
    score = -np.ones((5,)) * np.inf  # scores at -infinity, lower bound
    score_ = list()
    logger.debug("ended initialisations")
    while True:
        try:
            logger.info("refining alpha grid to interval [{}, {}]".format(
                alphas[0], alphas[-1]))
            for (ix, alpha) in enumerate(alphas):
                if not np.isinf(score[ix]):
                    logger.info("alpha[{}] = {}, already computed".format(
                        ix, alpha))
                    continue
                logger.info("computing for alpha[{}] = {}".format(ix, alpha))
                if method == 'gl':
                    cov_learner_ = cov_learner(alpha=alpha,
                                               score_norm=score_norm,
                                               tol=tol, mu=10.,
                                               base_estimator=base_estimator)
                    logger.debug("start optimisation")
                    res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
                        X, X_test, cov_learner_, train_ix, test_ix, retest_ix,
                        ips_flag)
                        for train_ix, test_ix, retest_ix in shuffle_split)
                    score[ix] = np.mean(np.array(res_))
                elif h is not None:
                    afunc = partial(_alpha_func, h=h, max_level=max_level)
                    cov_learner_ = cov_learner(alpha=alpha,
                                               htree=htree,
                                               alpha_func=afunc,
                                               score_norm=score_norm,
                                               tol=tol, mu=10.,
                                               base_estimator=base_estimator)
                    logger.debug("start optimisation")
                    res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
                        X, X_test, cov_learner_, train_ix, test_ix, retest_ix,
                        ips_flag)
                        for train_ix, test_ix, retest_ix in shuffle_split)
                    score[ix] = np.mean(np.array(res_))
                else:
                    logger.debug("start optimisation (h)")
                    h_opt, score_h = _compute_hopt(
                        alpha, score_norm, X, X_test, shuffle_split,
                        htree, ips_flag=ips_flag, n_jobs=n_jobs, tol=tol,
                        base_estimator=base_estimator)
                    h_[ix] = h_opt
                    score[ix] = score_h
            # TODO: better strategy when on boundary ?
            max_ix = min(max(np.argmax(score), 1), 3)
            score_.append(np.max(score))
            alpha_opt = alphas[np.argmax(score)]
            logger.info("score @ alpha = {} : {}".format(alphas, score))
            logger.info("maximum score @ alpha = {}".format(alpha_opt))
            score[0] = score[max_ix - 1]
            score[4] = score[max_ix + 1]
            score[2] = score[max_ix]
            score[1] = -np.inf
            score[3] = -np.inf
            alphas = np.linspace(alphas[max_ix - 1], alphas[max_ix + 1], 5)
            if method == 'hgl' and h is None:
                h_opt = h_[np.argmax(score)]
            if alphas[4] - alphas[0] <= alpha_tol:
                raise StopIteration
        except StopIteration:
            if method == 'gl' or h is not None:
                return alpha_opt, score_
            else:
                return alpha_opt, score_, h_opt


def _compute_hopt(alpha, score_norm, X, X_test, shuffle_split, htree,
                  h_tol=0.01, n_jobs=1, ips_flag=True, tol=1e-16,
                  base_estimator=EmpiricalCovariance(assume_centered=True)):
    """ grid search for the optimal parameter 'h' in hierarchical GraphLasso

    Arguments:
    ----------
    alpha   : float
        the penalisation parameter, we are doing a line search for h given
        alpha

    score_norm : string
        any of the norms supported by the covariance learner

    X   : numpy.ndarray of shape (n, p)
        the training data

    X_test : numpy.ndarray of shape (m, p)
        the testing data

    shuffle_split : list of couples of tuples
        each tuple is an index set, the first of the couple is used for
        indexing data in X, the second in X_test

    htree : a hierarchical tree object, or compatible list
        the hierarchical tree topology

    htol : float, 0. < htol <= 1.
        iterations stop when this tolerance is reached for h

    n_jobs : integer
        number of (embarassingly) parallel jobs

    ips_flag : boolean
        whether maximum likelihood is on the full model or merely support
        set matching (ips_flag=True)

    Returns:
    --------
    h_opt : float, 0. <= h_opt <= 1.
        optimal parameter found during line search


    score : float
        the (mean) score associated with h_opt and alpha
    """
    # init
    scoreh = -np.ones((5,)) * np.inf
    hs = np.linspace(0.,1., 5)
    max_level = HTree(htree).get_depth()
    while True:
        try:
            if alpha > 0.:
                logger.info("\trefining h-grid to interval " +
                            "[{}, {}]".format(hs[0], hs[-1]))
            for (ixh, h) in enumerate(hs):
                if ixh in [0, 2, 4] and not np.isinf(scoreh[ixh]):
                    continue
                cov_learner_h = HierarchicalGraphLasso(
                    htree, alpha, score_norm=score_norm, mu=10.,
                    alpha_func=partial(_alpha_func, h=h,
                                       max_level=max_level),
                    tol=tol, base_estimator=base_estimator)
                res_h = Parallel(n_jobs=n_jobs)(
                    delayed(_eval_cov_learner)(
                        X, X_test, cov_learner_h, train_ix, test_ix,
                        retest_ix, ips_flag)
                    for train_ix, test_ix, retest_ix in shuffle_split)
                scoreh[ixh] = np.mean(np.array(res_h))
                logger.info("\t\tscore (alpha = {}, h = {}) : {}".format(
                            alpha, h, scoreh[ixh]))
                if alpha == 0.:
                    return 0., scoreh[ixh]
            max_ixh = min(max(np.argmax(scoreh), 1), 3)
            h_opt = hs[np.argmax(scoreh)]
            scoreh[0] = scoreh[max_ixh - 1]
            scoreh[4] = scoreh[max_ixh + 1]
            scoreh[2] = scoreh[max_ixh]
            scoreh[1] = -np.inf
            scoreh[3] = -np.inf
            hs = np.linspace(hs[max_ixh - 1],
                             hs[max_ixh + 1], 5)
            if hs[4] - hs[0] <= h_tol:
                raise StopIteration
        except StopIteration:
            return h_opt, np.max(scoreh)


def _get_indices(X, y, X_test, y_test, n_iter=10,
                 train_size=.2, test_size=.2, retest_size=None,
                 random_state=None, **kwargs):
    """ get the shuffle split indices for training, testing, and retesting
    """
    from sklearn import cross_validation
    # depending on the presence of the X_test variable, train and test
    # should either be taken in X only, or in X and X_test, respectively.
    if X_test is None:
        test_size_ = test_size
        train_size_ = train_size
        retest_size_ = None
        # no unnecessary data copy here --> by data reference !
        X_test = X
    else:
        test_size_ = None
        train_size_ = test_size
    # using stratified shuffle split if y is given, shuffle split otherwise
    # training data
    if y is not None:
        shuffle_split = cross_validation.StratifiedShuffleSplit(
            y, n_iter=n_iter, train_size=train_size, test_size=test_size_,
            random_state=random_state)
    else:
        shuffle_split = cross_validation.ShuffleSplit(
            X.shape[0], n_iter=n_iter, train_size=train_size,
            test_size=test_size_, random_state=random_state)
    # (re)testing data different from training data
    if y_test is not None:
        shuffle_split_test = cross_validation.StratifiedShuffleSplit(
            y_test, n_iter=n_iter, test_size=retest_size,
            train_size=train_size_, random_state=random_state)
    elif test_size_ is None:
        shuffle_split_test = cross_validation.ShuffleSplit(
            X_test.shape[0], n_iter=n_iter, test_size=retest_size,
            train_size=train_size_, random_state=random_state)

    # allow for a unique call to retrieve train and test indices
    # test[0] corresponds to test
    # test[1] corresponds to re-test
    if test_size_ is None and retest_size is not None:
        shuffle_split = [(train[0], test[0], test[1]) for train, test in
                         zip(shuffle_split, shuffle_split_test)]
    elif test_size_ is None:
        shuffle_split = [(train[0], test[0], None) for train, test in
                         zip(shuffle_split, shuffle_split_test)]
    else:
        shuffle_split = [(train, test, None) for train, test in shuffle_split]
    return shuffle_split


def _eval_cov_learner(X, X_test, cov_learner,
                      train_ix, test_ix, retest_ix=None, ips_flag=True):
    X_train_ = X[train_ix, ...]
    X_test_ = X_test[test_ix, ...]
    if retest_ix is not None:
        X_retest_ = X_test[retest_ix, ...]
    # learn a sparse covariance model
    cov_learner_ = clone(cov_learner)
    if cov_learner_.alpha > 0.:
        alpha_max_ = alpha_max(X_train_,
                               base_estimator=cov_learner.base_estimator)
        cov_learner_.__setattr__('alpha', cov_learner_.alpha * alpha_max_)
    if not ips_flag:
        logger.debug("start learning precision matrix")
        score = cov_learner_.fit(X_train_).score(X_test_)
    elif cov_learner.score_norm != "ell0":
        # dual split variable contains exact zeros!
        logger.info("\t\tlearning precision matrix")
        logger.debug("required tolerance = %f" % cov_learner_.tol)
        aux_prec = cov_learner_.fit(X_train_).auxiliary_prec_
        mask = np.abs(aux_prec) >= machine_eps(1.)
        logger.info("\t\tstarting and scoring IPS estimate")
        ips = IPS(support=mask, score_norm=cov_learner_.score_norm,
                  base_estimator=cov_learner.base_estimator,
                  tol=cov_learner.tol)
        # on the mask, fit and score the testing dataset in the likelihood sense
        if retest_ix is not None:
            score = ips.fit(X_test_).score(X_retest_)
        else:
            score = ips.fit(X_train_).score(X_test_)
    else:
        raise ValueError('ell0 scoring in CV_loop and IPS are incompatible')

    # make score maximal at optimum
    if cov_learner_.score_norm not in {'loglikelihood', None}:
        score *= -1.
    return score


def alpha_max(X, base_estimator=EmpiricalCovariance(assume_centered=True)):
    _check_estimator(base_estimator)
    _check_2D_array(X)
    C = _cov_2_corr(base_estimator.fit(X).covariance_)
    C.flat[::C.shape[0] + 1] = 0.
    return np.max(np.abs(C))


def _alpha_func(alpha, lev, h=1., max_level=1.):
    if h > machine_eps(0):
        g1 = gamma_func(max_level - lev + h)
        g2 = gamma_func(max_level - lev + 1)
        g3 = gamma_func(h)
        return alpha * g1 / (g2 * g3)
    elif hasattr(lev, '__iter__'):
        return alpha * np.array([lev_ == max_level for lev_ in lev],
                                dtype=np.float)
    else:
        return alpha * np.float(lev == max_level)


def ric(mx, mask=None):
    """ Ravikumar Irrepresentability Condition for a correlation mx

    arguments:
    ---------
        mx  : the matrix on which the ric is to be computed (precision matrix)

        mask: if mx does not contain exact zeros, use this matrix as a logical
            mask for edge indication (non-zero only where edges are present,
            self-loops must be included)

    returns:
    -------
        the irrepresentability condition
    """
    if mask is None:
        mask = np.abs(mx) > machine_eps(0.)
    mx = linalg.inv(mx)
    Gamma = np.kron(mx, mx)
    edge_set = np.where(np.triu(mask).flat[:])[0]
    non_edge_set = np.where(np.triu(np.logical_not(mask)).flat[:])[0]
    G_ScS = Gamma[np.ix_(non_edge_set, edge_set)]
    G_SS = Gamma[np.ix_(edge_set, edge_set)]
    return np.max(np.sum(np.abs(G_ScS.dot(G_SS)), axis=1))


def _check_estimator(base_estimator):
    if not hasattr(base_estimator, 'get_precision'):
        raise ValueError('Your base_estimator is not a covariance estimator')


def _check_2D_array(X):
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a 'numpy.ndarray' object")
    if X.ndim != 2:
        raise ValueError('X must be a 2-dimensional array')


def cov_2_data(Theta, precision=False, scaled=True):
    eig_vals, eig_vecs = linalg.eig(Theta)
    if precision:
        eig_vals = 1. / eig_vals
    if scaled:
        eig_vals *= eig_vals.size
    return (eig_vecs * np.sqrt(eig_vals)).T


def machine_eps(f):
    import itertools
    return next(2 ** -i for i in itertools.count() if f + 2 ** -(i + 1) == f)
