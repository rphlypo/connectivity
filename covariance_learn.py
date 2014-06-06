import logging
reload(logging)
import sys
import numpy as np
import sklearn.utils.extmath
import copy
import numbers


from scipy.ndimage.measurements import mean as label_mean
from scipy.special import gamma as gamma_func


from htree import HTree
from scipy import linalg
from sklearn.base import clone
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
from functools import partial

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
    def __init__(self, alpha, tol=1e-6, max_iter=1e4, verbose=0,
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
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_gl(S, self.alpha, rho=self.rho, tol=self.tol,
                     max_iter=self.max_iter)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
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
        X_test : array_like, shape = [n_samples, n_features]
            Data for testing the method, could be the model itself

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
            # KL is symmetrised
            error_norm = -self.precision_.shape[0]
            error_norm += np.trace(linalg.inv(
                test_cov.dot(self.precision_))) / 2.
            error_norm += np.trace(
                test_cov.dot(self.precision_)) / 2.
        elif norm == "ell0":
            # X_test acts as a mask
            error_norm = self._support_recovery_norm(X_test)
        else:
            raise NotImplementedError(
                "Only the following norms are implemented:\n"
                "spectral, Frobenius, inverse Frobenius, and geodesic")
        return error_norm

    def _support_recovery_norm(self, X_test):
        return np.sum(np.logical_xor(
            np.abs(self.auxiliary_prec_) > machine_eps(0),
            np.abs(X_test) > machine_eps(0))) / 2


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
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_ips(S, self.support, rho=self.rho, tol=self.tol,
                      max_iter=self.max_iter)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self


class HierarchicalGraphLasso(GraphLasso):
    def __init__(self, htree, alpha, tol=1e-6, max_iter=1e4, verbose=0,
                 base_estimator=None, scale_2_corr=True, rho=1., mu=None,
                 score_norm=None, n_jobs=1, alpha_func=None):
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

        if hasattr(self.htree, '__iter__'):
            self.htree_ = HTree(self.htree).create()
            # {htree}._update() is ok for small trees, otherwise use on-the-fly
            # evaluation with {node}._get_node_values() at each node call
        elif isinstance(self.htree, HTree):
            self.htree_ = self.htree
        else:
            raise TypeError("htree must be an iterable or a HTree object")
        precision_, split_precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_hgl2(S, self.htree_, self.alpha, rho=self.rho, tol=self.tol,
                       mu=self.mu, max_iter=self.max_iter,
                       alpha_func=self.alpha_func)

        self.precision_ = precision_
        self.auxiliary_prec_ = split_precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self

#   def _support_recovery_norm(self, X_test):
#       # this returns an ordered list from leaves to root nodes
#       nodes_levels = self.htree_.root_.get_descendants()
#       nodes_levels.sort(key=lambda x: x[1])
#       nodes_levels.reverse()
#       p = len([node for node in nodes_levels if not node[0].get_children()])
#       mask_ = np.zeros((p, p), dtype=np.bool)

#       error_norm = 0

#       for (node, level) in nodes_levels:
#           if node.complement() is None:
#               continue
#           ix = node.evaluate()
#           for node_c in node.complement():
#               ixc = node_c.evaluate()
#               ixs = np.ix_(ix, ixc)
#               mask_ = np.linalg.norm(X_test[ixs]) > machine_eps(1)
#               data_ = np.linalg.norm(
#                   self.auxiliary_prec_[ixs]) > machine_eps(1)
#               error_norm += np.logical_xor(mask_, data_) * X_test[ixs].size
#       return error_norm


def _admm_gl(S, alpha, rho=1., tau_inc=2., tau_decr=2., mu=None, tol=1e-6,
             max_iter=100, Xinit=None):
    p = S.shape[0]
    Z = (1. + rho) / rho * np.identity(p)
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
    iter_count = 0
    while True:
        try:
            Z_old = Z.copy()
            # closed form optimization for X
            eigvals, eigvecs = linalg.eigh(rho * (Z - U) - S)
            eigvals /= 2
            eigvals = (eigvals + (eigvals ** 2 + rho) ** (1. / 2)) / rho
            X = eigvecs.dot(np.diag(eigvals).dot(eigvecs.T))
            func_val = -np.sum(np.log(eigvals)) + np.sum(S * X)
            # proximal operator for Z: soft thresholding
            tmp = np.abs(X + U) - alpha / rho
            Z = np.sign(X + U) * tmp * (tmp > 0)
#           Z = np.sign(X + U) * np.max(
#               np.reshape(np.concatenate((np.abs(X + U) - alpha / rho,
#                                          np.zeros((p, p))), axis=1),
#                          (p, p, -1), order="F"), axis=2)
            func_val += np.sum(alpha * np.abs(X))
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / (p ** 2))
            s_.append(linalg.norm(Z - Z_old) / (p ** 2))
            f_vals_.append(func_val)

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
            return X, Z, r_, s_, f_vals_


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
            eigvals = (eigvals + (eigvals ** 2 + rho) ** (1. / 2)) / rho
            X = eigvecs.dot(np.diag(eigvals).dot(eigvecs.T))
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
            return X, Z, r_, s_, f_vals_


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
    iter_count = 0
    # this returns an ordered list from leaves to root nodes
    nodes_levels = htree.root_.get_descendants()
    max_level = max([lev for (_, lev) in nodes_levels])
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
            eigvals, eigvecs = linalg.eigh(rho * (Z - U) - S)
            eigvals /= 2
            eigvals = (eigvals + (eigvals ** 2 + rho) ** (1. / 2)) / rho
            X = eigvecs.dot(np.diag(eigvals).dot(eigvecs.T))
            # smooth functional score
            func_val = -np.sum(np.log(eigvals)) + np.sum(X * S)
            # proximal operator for Z: block norm soft thresholding
            Z = U + X
            # TODO for a given level we could evaluate all in parallel!
            # 'mounting' in the tree from leaves to root

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
                multipliers = np.concatenate((np.array([1.]), multipliers))
                Z = multipliers[L + L.T] * Z
                func_val += 2 * alpha_ * np.sum(Xnorms)
            f_vals_.append(func_val)
            # update scaled dual variable
            U = U + X - Z
            r_.append(linalg.norm(X - Z) / np.sqrt(p ** 2))
            s_.append(linalg.norm(Z - Z_old))

            if mu is not None:
                U *= rho  # this is the unscaled Y
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
                U /= rho  # newly scaled dual variable
            iter_count += 1
            if (_check_convergence(X, Z, Z_old, U, rho, tol_abs=tol) or
                    iter_count > max_iter):
                raise StopIteration
        except StopIteration:
            return X, Z, r_, s_, f_vals_


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


def cross_val(X, method='gl', alpha_tol=1e-2,
              n_iter=100, train_size=.1, test_size=.5,
              model_prec=None, verbose=0, n_jobs=1,
              random_state=None, ips_flag=False,
              score_norm="KL", CV_norm=None,
              optim_h=False, **kwargs):
    from sklearn import cross_validation
    from joblib import Parallel, delayed
    # logging.ERROR is at level 40
    # logging.WARNING is at level 30, everything below is low priority
    # logging.INFO is at level 20, verbose 10
    # logging.DEBUG is at level 10, verbose 20
    logger.setLevel(logging.WARNING - verbose)
    bs = cross_validation.Bootstrap(X.shape[0], n_iter=n_iter,
                                    train_size=train_size,
                                    test_size=test_size,
                                    random_state=random_state)
    if method == 'gl':
        cov_learner = GraphLasso
    elif method == 'hgl':
        cov_learner = HierarchicalGraphLasso
        tree = kwargs['htree']
        if hasattr(tree, '__iter__'):
            tree = HTree(tree).create()
        max_level = max([lev for (_, lev) in tree.root_.get_descendants()])
    elif method == 'ips':
        cov_learner = IPS
    if CV_norm is None:
        CV_norm = score_norm
    # alpha_max ?
    alphas = np.linspace(0., 1., 5)
    LL = np.zeros((5,))
    LL_ = list()
#   for (ix, alpha) in enumerate(alphas):
#       cov_learner_ = cov_learner(alpha=alpha, score_norm=CV_norm,
#                                  **kwargs)
#       res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
#           X, train_ix, test_ix, model_prec, cov_learner_, ips_flag)
#           for train_ix, test_ix in bs)
#       LL[ix] = np.mean(np.array(res_))
#   LL_.append(LL[2])
    first_run_alpha = True
    while True:
        try:
            print "refining alpha grid to interval [{}, {}]".format(
                alphas[0], alphas[-1])
            logger.info("refining alpha grid to interval [{}, {}]".format(
                alphas[0], alphas[-1]))
            for (ix, alpha) in enumerate(alphas):
                if not first_run_alpha and ix in [0, 2, 4]:
                    print "alpha[{}] = {}, already computed".format(ix, alpha)
                    continue
                print "computing for alpha[{}] = {}".format(ix, alpha)
                if method != 'hgl' or not optim_h:
                    cov_learner_ = cov_learner(alpha=alpha, score_norm=CV_norm,
                                               **kwargs)
                    res_ = Parallel(n_jobs=n_jobs)(delayed(_eval_cov_learner)(
                        X, train_ix, test_ix, model_prec, cov_learner_,
                        ips_flag)
                        for train_ix, test_ix in bs)
                    LL[ix] = np.mean(np.array(res_))
                else:
                    LLh = np.zeros((5,))
                    hs = np.linspace(0, 1., 5)
                    first_run_h = True
                    while True:
                        try:
                            print "\trefining h-grid to " +\
                                "interval [{}, {}]".format(
                                    hs[0], hs[-1])
                            for (ixh, h) in enumerate(hs):
                                if not first_run_h and ixh in [0, 2, 4]:
                                    continue
                                cov_learner_h = cov_learner(
                                    alpha=alpha, score_norm=CV_norm,
                                    alpha_func=partial(_alpha_func, h=h,
                                                       max_level=max_level),
                                    **kwargs)
                                res_h = Parallel(n_jobs=n_jobs)(
                                    delayed(_eval_cov_learner)(
                                        X, train_ix, test_ix, model_prec,
                                        cov_learner_h, ips_flag)
                                    for train_ix, test_ix in bs)
                                LLh[ixh] = np.mean(np.array(res_h))
                            max_ixh = min(max(np.argmax(LLh), 1), 3)
                            LLh[0] = LLh[max_ixh - 1]
                            LLh[4] = LLh[max_ixh + 1]
                            LLh[2] = LLh[max_ixh]
                            hs = np.linspace(hs[max_ixh - 1],
                                             hs[max_ixh + 1], 5)
                            if hs[4] - hs[0] < .1:
                                raise StopIteration
                        except StopIteration:
                            LL[ix] = np.min(LLh)
                            h_opt = hs[np.argmin(LLh)]
                            break
                        first_run_h = False

            max_ix = min(max(np.argmax(LL), 1), 3)
            LL[0] = LL[max_ix - 1]
            LL[4] = LL[max_ix + 1]
            LL[2] = LL[max_ix]
            LL[1] = LL[3] = 0.
            alphas = np.linspace(alphas[max_ix - 1], alphas[max_ix + 1], 5)
            if alphas[4] - alphas[0] < alpha_tol:
                raise StopIteration
            LL_.append(np.min(LL))
            alpha_opt = alphas[np.argmin(LL)]
        except StopIteration:
            if score_norm == CV_norm:
                if method != 'hgl' or not optim_h:
                    return alpha_opt, LL_
                else:
                    return alpha_opt, LL_, h_opt
            else:
                if method != 'hgl' or not optim_h:
                    cov_learner_ = cov_learner(alpha=alpha_opt,
                                               score_norm=score_norm,
                                               **kwargs)
                    res_ = Parallel(n_jobs=n_jobs)(
                        delayed(_eval_cov_learner)(
                            X, train_ix, test_ix, model_prec, cov_learner_,
                            ips_flag)
                        for train_ix, test_ix in bs)
                    LL_star = np.mean(np.array(res_))
                    return alpha_opt, LL_, LL_star
                else:
                    cov_learner_ = cov_learner(
                        alpha=alpha_opt, score_norm=score_norm,
                        alpha_func=partial(_alpha_func, h=h_opt,
                                           max_level=max_level),
                        **kwargs)
                    res_ = Parallel(n_jobs=n_jobs)(
                        delayed(_eval_cov_learner)(
                            X, train_ix, test_ix, model_prec, cov_learner_,
                            ips_flag)
                        for train_ix, test_ix in bs)
                    LL_star = np.mean(np.array(res_))
                    return alpha_opt, LL_, h_opt, LL_star
        first_run_alpha = False


def _eval_cov_learner(X, train_ix, test_ix, model_prec, cov_learner,
                      ips_flag=True):
    X_train = X[train_ix, ...]
    if model_prec is None:
        X_test = X[test_ix, ...]
    else:
        eigvals, eigvecs = linalg.eigh(model_prec)
        X_test = np.diag(1. / np.sqrt(eigvals)).dot(eigvecs.T)
    cov_learner_ = clone(cov_learner)
    if not ips_flag:
        score = cov_learner_.fit(X_train).score(X_test)
    elif cov_learner.score_norm != "ell0":
        # dual split variable contains exact zeros!
        aux_prec = cov_learner_.fit(X_train).auxiliary_prec_
        mask = np.abs(aux_prec) > machine_eps(0.)
        ips = IPS(support=mask, score_norm=cov_learner_.score_norm)
        score = ips.fit(X_train).score(X_test)
    else:
        raise ValueError('ell0 scoring in CV_loop and IPS are incompatible')

    # make scores maximal at optimum
    if cov_learner_.score_norm is not None:
        score *= -1.
    return score


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
        mx  : the matrix on which the ric is to be computed

        mask: if mx does not contain exact zeros, use this matrix as a binary
            mask for edge indication (non-zero only where edges are present,
            self-loops must be included)

    returns:
    -------
        the irrepresentability condition
    """
    Gamma = np.kron(mx, mx)
    if mask is None:
        mask = mx
    edge_set = np.where(np.triu(mask != 0).flat[:])[0]
    non_edge_set = np.where(np.triu(mask == 0).flat[:])[0]
    G_ScS = Gamma[np.ix_(non_edge_set, edge_set)]
    G_SS = Gamma[np.ix_(edge_set, edge_set)]
    return np.max(np.sum(np.abs(G_ScS.dot(G_SS)), axis=1))


def machine_eps(f):
    import itertools
    return next(2 ** -i for i in itertools.count() if f + 2 ** -(i + 1) == f)
