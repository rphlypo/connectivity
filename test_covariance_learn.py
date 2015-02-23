from numpy.testing import assert_almost_equal, assert_equal, \
    assert_array_almost_equal, assert_array_less, assert_raises
from nose.tools import assert_greater
import numpy as np
import covariance_learn as covl
reload(covl)
from sklearn.base import BaseEstimator
import scipy.linalg
import itertools


X = np.random.normal(size=(10, 5))
X = X.T.dot(X) / 10


def test__update_rho():
    mu = 2.
    tau_inc = 3.
    tau_decr = 4.
    rho = 1.
    a = 1.
    # check whether rho does not change, but U does (mutable)
    # 1. s > mu * r => rho /= tau_decr, U *= tau_decr
    U = X.copy()
    rho_new = covl._update_rho(U, rho, a, mu * a + 1, mu, tau_inc, tau_decr)
    assert_array_almost_equal(U, X * tau_decr)
    assert_almost_equal(rho_new, rho / tau_decr)
    # 2. r > mu * s => rho *= tau_inc, U /= tau_inc
    U = X.copy()
    rho_new = covl._update_rho(U, rho, mu * a + 1, a, mu, tau_inc, tau_decr)
    assert_array_almost_equal(U, X / tau_inc)
    assert_almost_equal(rho_new, rho * tau_inc)


def test__update_X():
    rho = 1.
    S = np.random.normal(size=(10, 5))
    S = S.T.dot(S) / 10.
    U = np.zeros(X.shape)
    X_ = covl._update_X(S, X, U, rho)[0]
    assert_array_almost_equal(rho * X_ - scipy.linalg.inv(X_),
                              rho * (X - U) - S)


def test__get_indices():
    m1, m2, n = 50, 50, 20
    m = m1 + m2
    X = np.random.normal(size=(m, n))
    X_test = np.random.normal(size=(m, n))
    y = np.zeros((m,))
    y[m1:] = 1
    np.random.shuffle(y)
    y_test = np.zeros((m,))
    y_test[m1:] = 1
    np.random.shuffle(y_test)
    n_train = 10
    n_test = 12
    for (n_retest, y, y_test, X_test) in itertools.product([None, 14],
                                                           [None, y],
                                                           [None, y_test],
                                                           [None, X_test]):
        ix = covl._get_indices(X=X, y=y, X_test=X_test, y_test=y_test,
                               train_size=n_train, test_size=n_test,
                               retest_size=n_retest, n_iter=3)
        for ix_train, ix_test, ix_retest in ix:
            assert_equal(len(ix_train), n_train)
            assert_equal(len(ix_test), n_test)
            if n_retest is None or X_test is None:
                assert_equal(ix_retest is None, True)
            else:
                assert_equal(len(ix_retest), n_retest)


def test_IPS():
    Y = np.random.normal(size=(10, 5))
    A = np.random.uniform(-1., 1., size=(5, 5))
    Y = Y.dot(A)
    support = np.random.uniform(0., 1., size=(5, 5))
    p = .5
    support.flat[::5 + 1] = 1.
    support = (support + support.T) / 2.
    support[support <= p] = False
    support[support > p] = True

    cov_learner = covl.IPS(support)
    cov_learner.fit(Y)
    # we must have descending function values
    df = np.diff(cov_learner.f_vals_[1:])
    assert_array_almost_equal(df[df > 0.], 0.)


def test_GraphLasso():
    Y = np.random.normal(size=(10, 5))
    A = np.random.uniform(-1., 1., size=(5, 5))
    Y = Y.dot(A)
    alpha = np.linspace(0., 1., .1)
    for alpha_ in alpha:
        cov_learner = covl.GraphLasso(alpha)
        cov_learner.fit(Y)
        # we must have descending function values
        df = np.diff(cov_learner.f_vals_[1:])
        assert_array_less(0., scipy.linalg.eig(cov_learner.precision_)[0])
        assert_array_almost_equal(df[df > 0.], 0.)


def test__check_convergence():
    # convergence must be true at the fixed point X = Z
    convergence = covl._check_convergence(X, X, X, np.zeros(X.shape), 1.)
    assert_equal(convergence, True)


def test__cov_2_corr():
    p = X.shape[0]
    C = covl._cov_2_corr(X)
    assert_array_almost_equal(C.flat[::p + 1], np.ones((p,)))
    C_ = C.copy()
    C_.flat[::p + 1] = 0.
    assert_array_less(np.abs(C_), np.ones((p, p)))
    assert_array_less(0., scipy.linalg.eigvals(C))
    # test inverse transform
    var = X.flat[::p + 1]
    stddev = np.atleast_2d(np.sqrt(var))
    assert_array_almost_equal(C * stddev * stddev.T, X)


def test_alpha_max():
    assert_greater(1., covl.alpha_max(X))


def test__check_estimator():
    assert_raises(ValueError, covl._check_estimator,
                  base_estimator=BaseEstimator)


def test__check_2D_array():
    assert_raises(ValueError, covl._check_2D_array, 1.)
    assert_raises(ValueError, covl._check_2D_array, X[..., np.newaxis])


"""
def test_convergence_speed():
    import itertools
    alpha = np.logspace(-2, 0, 5)
    rho = np.logspace(-2., 2., 5)
    adaptive = [True, False]
    optimal = [True, False]

    for (alpha_, adap, opt) in itertools.product(alpha, adaptive, optimal):
        for _ in range(100):
            f = list()
            Y = np.random.normal(size=(90, 100))
            if not opt:
                for rho_ in rho:
                    if adap:
                        gl = covl.GraphLasso(alpha_, rho=rho_, mu=2.)
                    elif not adap:
                        gl = covl.GraphLasso(alpha_,
                                             rho=alpha_ / covl.alpha_max(Y),
                                             mu=2.)
                    f.append(gl.fit(Y).f_vals_)
            else:
                if adap:
                    gl = covl.GraphLasso(alpha_,
                                         rho=alpha_ / covl.alpha_max(Y),
                                         mu=2.)
                elif not adap:
                    gl = covl.GraphLasso(alpha_, rho=alpha_ / covl.alpha_max(Y))
                f.append(gl.fit(Y).f_vals_)
            # process f
"""
