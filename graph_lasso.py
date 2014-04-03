import logging
import numpy as np
import scipy.linalg as linalg
import sklearn.covariance
import sklearn.utils.extmath
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance.empirical_covariance_ import EmpiricalCovariance
import sklearn.linear_model
import scipy.optimize
import copy

logger = logging.getLogger(__name__)
fast_logdet = sklearn.utils.extmath.fast_logdet

class GraphLasso(EmpiricalCovariance):
    """ the estimator class for the GraphLasso estimator

    """
    def __init__(self, alpha, Theta_init=None, tol=1e-4, max_iter=100,
                 verbose=0, assume_centered=True, inner_loop_tol=1e-16):
        self.assume_centered = assume_centered
        self.alpha = alpha
        self.Theta_init = Theta_init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.inner_loop_tol=inner_loop_tol
        # needed for the score function of EmpiricalCovariance
        self.store_precision=True

    def set_params(self,**kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self,deep=True):
        return {"alpha"     : self.alpha,
                "Theta_init": self.Theta_init,
                "tol"       : self.tol,
                "verbose"   : self.verbose,
                "max_iter"  : self.max_iter,
                "assume_centered" : self.assume_centered,
                "inner_loop_tol" : self.inner_loop_tol}

    def fit(self,X,y=None,**kwargs):
        logger.setLevel(self.verbose)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        cov_learn = sklearn.covariance.EmpiricalCovariance(\
                                    assume_centered=self.assume_centered)
        S_ = cov_learn.fit(X).covariance_.copy()
        del cov_learn
        # scale diagonal to 1.
        scaling = np.diag(np.diag(S_)**(-1./2))
        S_ = np.dot(np.dot(scaling,S_),scaling)
        S_ = (S_ + S_.T)/2.
        p = S_.shape[0]

        Theta_, Gamma_, dual_gap_, f_obj_ = _graph_lasso(
                      Cov=S_, alpha=self.alpha, Theta_init=self.Theta_init,
                      tol=self.tol, verbose=self.verbose, max_iter=self.max_iter,
                      inner_loop_tol=self.inner_loop_tol)

        self.emp_ = S_.copy()
        self.precision_ = Theta_.copy()
        self.Gamma_ = Gamma_.copy()
        self.dual_gap_ = copy.deepcopy(dual_gap_)
        self.f_obj_ = copy.deepcopy(f_obj_)
        return self

    # can be inherited from the EmpiricalCovariance estimator once integrated
    # in scikit-learn
#    def score(self, X):
#        cov_learn = sklearn.covariance.EmpiricalCovariance(\
#                            assume_centered=self.assume_centered)
#        S = cov_learn.fit(X).covariance_
#        return sklearn.covariance.log_likelihood(S, prec_estimate)


def _primal_cost(emp_cov, prec_estimate, alpha):
    # sum( A * B) = trace( np.dot( A.T, B))
    log_likelihood =  - np.sum(emp_cov * prec_estimate)
    log_likelihood += fast_logdet( prec_estimate)
    return log_likelihood - np.sum( np.abs(alpha * prec_estimate))


def _dual_cost(emp_cov, gamma):
    p = emp_cov.shape[0]
    return - p - fast_logdet( emp_cov + gamma)


def _dual_gap( emp_cov, prec_estimate, alpha):
    p = emp_cov.shape[0]
    # sum( A * B) = trace( np.dot( A.T, B))
    log_lik_gap = - p + np.sum(emp_cov * prec_estimate)
    dual_gap = log_lik_gap  + np.sum( np.abs(alpha * prec_estimate))
    return dual_gap


def _graph_lasso( Cov, alpha, Theta_init=None,
                  tol=1e-4, verbose=0, max_iter=100, inner_loop_tol=1e-16):
    """ The primal-dual implementation of the graph-lasso

    The graph lasso solves the penalized likelihood-estimation of a precision
    matrix under a Gaussian family. The penalization is of the kind ell-1,
    promoting sparsity in the precision matrix, which acts as an adjacency
    matrix for the underlying graph structure.

    The cost function is:

        J_alpha(Theta) = - log det Theta + < S, Theta > + || A Theta ||_1

    if alpha is a given scalar, then A is a matrix of all ones multiplied by
    that scalar, the cost function could be rewritten as

        J_alpha(Theta) = - log det Theta + < S, Theta > + alpha || Theta ||_1

    TODO: check whether this works for || A Theta ||_1, A a symmetric
    penalization matrix with entries in [0, 1]

    where S is the empirical covariance matrix, alpha / A the penalization
    parameter(s), and Theta the quantity to estimate. The inner product <X, Y>
    is the inner product induced by the Frobenius norm, i.e., trace X'Y.

    This implementation of the graph lasso is based upon:
    Rahul Mazumder: "The graphical lasso: New insights and alternatives",
    Electronic Journal of Statistics, Vol.6, pp.2125--2149, 2012.

    Remark: we actually maximize -J_alpha(Theta), to keep the nature of the
    log_likelihood principle

    Input arguments
    ---------------
    Cov : np.ndarray of shape (n_features,n_features)
        the empirical covariance matrix which is to be matched by a sparse precision

    alpha: float in [0,1] or np.ndarray of shape (n_features,n_features)
        the penalty parameter, could be a single float or could weigh each of the
        entries of the precision differently

    Theta_init: np.ndarray of shape (n_features, n_features), optional
        an initial gues of the precision matrix

    tol: float, optional
        the functional tolerance on the dual gap, the dual gap always closes, but not
        necessarily in finite time, so you may want to specify a tolerance that is
        less restrictive than the numerical precision

    verbose: unsigned int, optional
        level of verbosity (0 corresponds to no output)

    max_iter: unsigned int, optional
        upper bound on the number of iterations to prevent infinite loops when one
        crashes into the wall of numerical precision

    inner_loop_tol: floar, optional
        tolerance on the update of the argument, the inner loop tolerance should be
        restrictive for the graph lasso to function well

    Returns
    -------
    Theta_: np.ndarray of shape (n_features, n_features)
        the estimated 'sparse' precision matrix

    covariance_: np.ndarray of shape (n_features, n_features)
        the covariance matrix; the inverse of Theta_

    dual_gap_: list of floats
        the dual gap value as a function of the iteration number

    f_obj_: list of floats
        the functional value as a function of the iteration number
    """
    Cov_ = Cov.copy()
    p = Cov_.shape[0]
    if isinstance(alpha, float):
        alpha = alpha * (np.ones((p,p)) - np.identity(p))
    else:
        alpha = alpha.copy()

    if Theta_init is None:
        Theta_ = linalg.inv( Cov_ + alpha*np.identity(p))
    else:
        Theta_ = Theta_init.copy()

    Gamma = - Cov_.copy()
    Gamma.flat[::p+1] = alpha.flat[::p+1].copy()

    ix_set = np.arange(p)

    dual_gap_ = [_dual_gap( Cov_, Theta_, alpha)]
    f_obj_ = [_primal_cost( Cov_, Theta_, alpha)]
    iter_count = 0
    while True:
        try:
            iter_count += 1
            for ix in np.arange(p):
                ixc = ix_set[ix_set!=ix]
                Theta11 = Theta_[np.ix_(ixc,ixc)].copy()
                s22 = Cov_[ix,ix].copy()
                s12 = Cov_[ixc,ix].copy()
                alpha12 = alpha[ixc,ix].copy()
                alpha22 = alpha[ix,ix].copy()
                w22 = s22 + alpha22
                gamma12_ = Gamma[ixc,ix].copy()
#               gamma12_ = quad_prog_box_bound( s12,
#                                               A=Theta11,
#                                               alpha=alpha12,
#                                               x_init=Gamma[ixc,ix],
#                                               verbose=verbose,
#                                               tol=inner_loop_tol)
                out = scipy.optimize.fmin_l_bfgs_b(func, gamma12_,
                                              args=(Theta11,s12),
                                              bounds=[(-a,a) for a in alpha12],
                                              disp=0)
                gamma12_ = out[0].copy()
                Gamma[ixc,ix] = gamma12_
                Gamma[ix,ixc] = gamma12_.T
                theta12_ = - Theta11.dot(gamma12_ + s12) / w22
                Theta_[ixc,ix] = theta12_
                Theta_[ix,ixc] = theta12_.T
                dot_prod = (gamma12_ + s12).dot(theta12_)
                Theta_[ix,ix] = (1 - dot_prod) / w22
            dual_gap_.append(_dual_gap( Cov_, Theta_, alpha))
            f_obj_.append(_primal_cost( Cov_, Theta_, alpha))

            str_out = "iteration {it}, cost = {c}, dual gap = {dg}"
            str_out = str_out.format(it = iter_count,
                                        c = f_obj_[-1],
                                        dg = dual_gap_[-1])
            logger.log(15,str_out)

            small_dual_gap = np.abs(dual_gap_[-1]) < tol
            exceed_max_iter = iter_count >= max_iter

            if small_dual_gap:
                str_out = "Convergence reached in {n} iterations"
                logger.log(18, str_out.format(n=iter_count))
                raise StopIteration
            elif exceed_max_iter:
                str_out = "No convergence reached in {n} iterations,"\
                            " returning current result"
                logger.log(19, str_out.format(n=max_iter))
                raise StopIteration
        except StopIteration:
            covariance_ = linalg.inv( Theta_)
            return Theta_, covariance_, dual_gap_, f_obj_

def func(x,A,s):
    return 1. / 2 * (x+s).T.dot(A).dot(x+s), A.dot( x+s)


def graph_lasso_path( X, alphas, X_test=None, tol=1e-4, max_iter=100,
                      inner_loop_tol=1e-16, verbose=0, assume_centered=True):
    """ the path of graph lasso, using a decreasing sequence of alphas

    Input arguments
    ---------------
    X: np.ndarray of shape (n_samples, n_features)
        the input data of which a covariance is obtained to which the fits are
        computed

    alphas: list of floats in [0,1]
        regularization parameters

    X_test: np.ndarray of shape (n_samples, n_features), optional
        testing data, if provided a score will be computed on this data

    tol: float, optional
        the functional tolerance on the dual gap, the dual gap always closes, but not
        necessarily in finite time, so you may want to specify a tolerance that is
        less restrictive than the numerical precision

    max_iter: unsigned int, optional
        upper bound on the number of iterations to prevent infinite loops when one
        crashes into the wall of numerical precision

    inner_loop_tol: floar, optional
        tolerance on the update of the argument, the inner loop tolerance should be
        restrictive for the graph lasso to function well

    verbose: unsigned int, optional
        level of verbosity, the higher this number the more output will appear on stdout

    assume_centered: boolean
        whether or not the data `X` and `X_test` have been centered before passing them
        to this function, if `False` the data in `X_test` is centered using the mean
        value of `X`

    Returns
    -------
    covariances_: list of np.ndarrays, each of shape (n_features, n_features)
        the estimated covariance matrices along the alpha path, they are obtained
        by inverting the estimated precisions of the regularized problem

    precisions_: list of np.ndarrays, each of shape (n_features, n_features)
        the sparse precision estimators obtained by solving the regularized problem

    scores_: list of floats
        generalization error, i.e., the "error" obtained on the test set. Returned
        only if test data is passed to the function in `X_test`
    """
    logger.setLevel(self.verbose)
    if self.assume_centered:
        location_ = np.zeros(X.shape[1])
    else:
        location_ = X.mean(0)
    cov_learn = sklearn.covariance.EmpiricalCovariance(\
                                assume_centered=assume_centered)
    emp_cov = cov_learn.fit(X).covariance_
    # scale diagonal to 1.
    scaling = np.diag(np.diag(emp_cov)**(-1./2))
    emp_cov = np.dot(np.dot(scaling,emp_cov),scaling)
    emp_cov = (emp_cov + emp_cov.T)/2.
    p = emp_cov.shape[0]

    if X_test is not None:
        X_test = X_test - location_
        test_cov_learn = sklearn.covariance.EmpiricalCovariance(\
                                assume_centered = True)
        test_emp_cov = test_cov_learn.fit(X).covariance_
        # scale diagonal to 1.
        scaling = np.diag(np.diag(test_emp_cov)**(-1./2))
        test_emp_cov = np.dot(np.dot(scaling,test_emp_cov),scaling)
        test_emp_cov = (test_emp_cov + test_emp_cov.T)/2.

    covariances_ = list()
    precisions_ = list()
    scores_ = list()

    Theta_init = None

    for alpha in np.sort( alphas, kind='mergesort')[::-1]:
        if precisions_:
            Theta_init = precisions_[-1]
        covariance_, precision_, _, _ = graph_lasso_(
                                            emp_cov, alpha=alpha,
                                            Theta_init=Theta_init,
                                            tol=tol, verbose=verbose,
                                            max_iter=max_iter,
                                            inner_loop_tol=inner_loop_tol)
        covariances_.append( covariance_)
        precisions_.append( precision_)
        if X_test is not None:
            scores_.append( sklearn.covariance.log_likelihood( test_emp_cov,
                                                               precision_))

    if X_test is not None:
        return covariances_, precisions_, scores_
    return covariances_, precisions_


class Gelato(BaseEstimator, ClassifierMixin):
    """ The primal-dual implementation of the graph-lasso

    The Gelato solves a regression problem to estimate the precision matrix.
    The penalization is of the kind ell-1, promoting sparsity in the precision matrix.
    The latter acts as an adjacency matrix for the underlying graph structure.

    This implementation is based upon:
    Shuheng Zhou, Philipp Rutimann, Min Xu, and Peter Buhlmann: "High-dimensional
    covariance estimation based on Gaussian graphical models," Journal of Machine
    Learning Research, Vol. 12, pp.5975--3026, 2011.

    parameters:
    ----------
    alpha: scalar
        the penalization parameter, often denoted lambda in scientific papers
    tau: scalar
        a small scalar that serves as the hard threshold
    assume_centered: boolean, optional
        whether the data is supposed to be centered (`assume_centered` == `True`), or
        whether the mean should be subtracted from the data when computing the covariance
    Theta_init: np.ndarray of size (p,p), optional
        initial guess of the precision matrix
    tol: scalar, optional
        the tolerance of the solution
    max_iter: int, optional
        maximum number of iterations
    verbose: int, optional
        higher values mean higher verbosity (verbose conventions of the `logger` module
        are considered)
    """
    def __init__(self, alpha, tau, Theta_init=None, tol=1e-4, max_iter=100,
                 verbose=0, assume_centered=True):
        self.assume_centered = assume_centered
        self.alpha = alpha
        self.tau = tau
        self.Theta_init = Theta_init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def set_params(self,**kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self,deep=True):
        return {"alpha"     : self.alpha,
                "tau"       : self.tau,
                "Theta_init": self.Theta_init,
                "tol"       : self.tol,
                "verbose"   : self.verbose,
                "max_iter"  : self.max_iter,
                "assume_centered" : self.assume_centered}

    def fit(self,X,y=None,**kwargs):
        logger.setLevel(self.verbose)
        p = X.shape[1]
        ix_set = np.arange(p)

        Lasso = sklearn.linear_model.Lasso( self.alpha,
                                            fit_intercept=not(self.assume_centered),
                                            normalize=True)

        B = np.zeros((p,p),dtype=np.float)
        ix_set = np.arange(p)
        for ix in np.arange(p):
            ixc = ix_set[ ix_set != ix]
            B[ixc,ix] = Lasso.fit( X[:,ixc], y=X[:,ix]).coef_
        self.B_ = B

        return self



# quad_prog_box_bound
def quad_prog_box_bound(s, A=None, x_init=None, alpha=0., max_iter=1e3,
                        tol=1e-16, verbose=0, lin_constr = False):
    """ quadratic programming under box constraints

    Input arguments:
    ---------------
    s   : np.ndarray of shape (p,)
        the 'offset' vector
    A   : np.ndarray of shape (p,p)
        the metric of the space in which inner products are computed
    x_init : np.ndarray of shape (p,)
        the initialization of the unknown vector
    alpha : np.ndarray of shape (p,) or float
        determines the box size
    max_iter : unsigned integer
        maximum number of iterations
    tol : float
        small float that determines the precision at the solution
    verbose : unsigned integer
        verbose level, the higher the more feedback one gets (>10 for
        debugging purposes only)

    Return arguments:
    ----------------
    x   : np.ndarray of shape (p,)
        the point that solves the symmetric quadratic problem

                max_x -1/2 (x+s)'A(x+s) s.t. -alpha_i <= x_i <= alpha_i

    The solver is a simple projected gradient with optimal step size; a
    gradient update rule is preferred to the Newton update, since the latter
    would not allow for coordinate-wise constraints.

    The algorithm proceeds as follows:
    * initialise with the Newton update, giving the unconstrained optimum
    * for any point inside the box: the gradient ascent update rule is applied
      as it yields the optimal step size; if this would let us exceed the box,
      its length is shortened to the maximum step length that keeps us in the
      box (i.e., it yields a point on the box boundaries)
    * for any point on the box: the gradient update for which a coordinate
      would exceed the box is first projected onto the box, then an optimal
      step size is computed on the box boundary; if this would let us exceed
      the box domain, it is shortened to the maximum step length that keeps us
      within/on the box
    """
    logger.setLevel(verbose)
    p_1 = s.size

    if A is None:
        raise Exception("Need to specify a quadratic form 'A'")

    if x_init is None:
        # initial value is optimum of the unconstrained problem (using the
        # Newton update for the quadratic form)
        x = -s
    else:
        x = x_init

    if not type(alpha) is np.ndarray:
        alpha = alpha * np.ones((p_1,))

    iteration_counter = 0
    # starting with a feasible point --> project x onto convex set
    x[np.abs(x) > alpha] = \
            np.sign( x[np.abs(x) > alpha]) * alpha[np.abs(x) > alpha]

    # projected gradient method for the box constraints
    while True:
        try:
            dx = np.dot(A,x + s)
            # if the point is on the border and update points outside the box,
            # no update should be done
            ix_d = np.logical_and( np.abs(x) >= alpha - tol, dx*x < 0)
            if np.any(lin_constr):
                ix_d_c = np.logical_not( ix_d)
                # TODO: more clever way, do not update if index set did not change
                if len(lin_constr.shape) >1:
                    theta, _ = linalg.qr(lin_constr[ix_d_c,...], mode='economic')
                else:
                    theta = lin_constr[ix_d_c,...] / linalg.norm( lin_constr[ix_d_c,...])
                dx[ix_d_c] = dx[ix_d_c] - np.dot(theta, np.dot(theta.T, dx[ix_d_c]))
            dx[ix_d] = 0.
            x[ix_d] = alpha[ix_d] * np.sign( x[ix_d])
            #x[ix_d] = alpha[ix_d] * np.sign( x[ix_d])
            # unit norm for better conditioning
            dx = dx / np.sqrt( np.sum(dx**2))
            ix = np.abs(dx) > tol
            if np.any(ix):
                # eta_min = 0. = np.max(0.,np.max( min_array))
                max_array = (np.sign(dx[ix])*x[ix] + alpha[ix]) / np.abs(dx[ix])
                eta_max = np.min( max_array)
                if eta_max <= 0.:
                    eta = 0.
                else:
                    dxA = np.dot(dx,A)
                    eta_opt = np.dot(dxA,s+x) / np.dot(dxA,dx)
                    # eta_opt --> 0 < eta < eta_max
                    eta = np.median( (0., eta_opt, eta_max) )
                    x[ix] -= eta*dx[ix]
            else:
                eta = 0.

            exceed_max_iter = iteration_counter > max_iter
            small_update_norm = eta <= tol
            if small_update_norm or exceed_max_iter:
                if exceed_max_iter:
                    str_out = "did not converge within {n} iterations"
                    logger.log(19, str_out.format(n=max_iter))
                else:
                    str_out = "convergence reached in {n} iterations ("\
                    "update norm = {dx_norm})"
                    logger.log(18,str_out.format(n=iteration_counter,
                                         dx_norm=np.sum(dx[ix]**2)**(1./2)))
                raise StopIteration
            iteration_counter += 1
        except StopIteration:
            break
    return x
