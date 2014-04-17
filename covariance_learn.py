import logging
import numpy as np
import sklearn.utils.extmath
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
import copy
import numbers
from scipy import linalg


from sklearn.base import clone


logger = logging.getLogger(__name__)
fast_logdet = sklearn.utils.extmath.fast_logdet


class GraphLasso(EmpiricalCovariance):
    """ the estimator class for GraphLasso based on ADMM
    """
    def __init__(self, alpha, tol=1e-6, max_iter=100, verbose=0,
                 base_estimator=None,
                 scale_2_corr=True, rho=1., score=None):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.scale_2_corr = True
        self.rho = rho
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        if self.base_estimator is None:
            self.base_estimator_ = EmpiricalCovariance(assume_centered=True)
        else:
            self.base_estimator_ = clone(self.base_estimator)
        logger.setLevel(self.verbose)
        S = self.base_estimator_.fit(X).covariance_
        if self.scale_2_corr:
            S = _cov_2_corr(S)
        precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_gl(S, self.alpha, rho=self.rho, tol=self.tol,
                     max_iter=self.max_iter)

        self.precision_ = precision_
        self.covariance_ = linalg.inv(precision_)
        self.var_gap_ = copy.deepcopy(var_gap_)
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_vals_ = copy.deepcopy(f_vals_)
        return self

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
        test_cov = self.base_estimator_.fit(X_test).covariance_
        if self.scale_2_corr:
            test_cov = _cov_2_corr(test_cov)
        # compute log likelihood
        return log_likelihood(self.precision_, test_cov)

    def error_norm(self, X_test, norm="Fro"):
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
                 scale_2_corr=True, rho=1., score=None):
        self.support = support
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator = base_estimator
        self.scale_2_corr = True
        self.rho = rho
        # needed for the score function of EmpiricalCovariance
        self.store_precision = True

    def fit(self, X, y=None, **kwargs):
        if self.base_estimator is None:
            self.base_estimator_ = EmpiricalCovariance(assume_centered=True)
        else:
            self.base_estimator_ = clone(self.base_estimator)
        logger.setLevel(self.verbose)
        S = self.base_estimator_.fit(X).covariance_
        if self.scale_2_corr:
            S = _cov_2_corr(S)
        precision_, var_gap_, dual_gap_, f_vals_ =\
            _admm_ips(S, self.support, rho=self.rho, tol=self.tol,
                      max_iter=self.max_iter)

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
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
            iter_count += 1
            if r_[-1] < tol or iter_count > max_iter:
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
                if r_[-1] > mu * s_[-1]:
                    rho *= tau_inc
                elif s_[-1] > mu * r_[-1]:
                    rho /= tau_decr
            iter_count += 1
            if r_[-1] < tol or iter_count > max_iter:
                raise StopIteration
        except StopIteration:
            return Z, r_, s_, f_vals_


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
    return (correlation + correlation.T) / 2
