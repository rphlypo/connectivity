from numpy.testing import assert_almost_equal, assert_equal, \
    assert_array_almost_equal, assert_array_less, assert_raises
from nose.tools import assert_greater
import numpy as np
import covariance_learn as covl
from sklearn.covariance import EmpiricalCovariance
from sklearn.base import BaseEstimator
import scipy.linalg


X = np.random.normal(size=(10, 5))
X = X.T.dot(X) / 10


def test__update_rho():
    U = X.copy()
    mu = 2.
    tau_inc = 3.
    tau_decr = 4.
    rho = 1.
    a = 1.
    # check whether rho does not change, but U does
    rho_old = 1.
    # 1. s > mu * r => rho *= tau_inc, U /= tau_inc
    rho_new = covl._update_rho(U, rho, a, mu * a + 1, tau_inc, tau_decr)
    assert_array_almost_equal(U, X / tau_inc)
    assert_almost_equal(rho, rho_old)
    assert_almost_equal(rho_new, rho_old * tau_inc)
    # 2. r > mu * s => rho /= tau_decr, U /= tau_decr
    covl._update_rho(U, rho, a, mu * a + 1, tau_inc, tau_decr)
    assert_array_almost_equal(U, X * tau_decr)
    assert_almost_equal(rho, rho_old)
    assert_almost_equal(rho_new, rho_old / tau_decr)


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
    assert_array_less(np.abs(C_), np.ones(p, p))
    assert_greater(np.min(scipy.linalg.eigvals(C)), 0.)
    # test inverse transform
    var = X.flat[::p + 1]
    stddev = np.atleast_2d(np.sqrt(var))
    assert_array_almost_equal(C * stddev * stddev.T, X)


def test_alpha_max():
    assert_greater(1., alpha_max(X))


def test__check_estimator():
    assert_raises(ValueError, covl._check_estimator,
                  base_estimator=BaseEstimator)


def test__check_2D_array():
    assert_raises(ValueError, covl._check_2D_array, 1.)
    assert_raises(ValueError, covl._check_2D_array, X[..., np.newaxis])


def test_convergence_speed():
    import itertools
    alpha = np.logspace(-2, 0, 5)
    rho = np.logspace(-2., 2., 5)
    adaptive = [True, False]
    optimal = [True, False]

    for (alpha_, adap, opt) in itertools.product(alpha, adaptive, optimal):
        for _ in range(100):
            f = list()
            Y = np.random.normal(size=(90,100))
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
                    gl = covl.GraphLasso(alpha_, rho=alpha_ / covl.alpha_max(Y), mu=2.)
                elif not adap:
                    gl = covl.GraphLasso(alpha_, rho=alpha_ / covl.alpha_max(Y))
                f.append(gl.fit(Y).f_vals_)
            # process f

