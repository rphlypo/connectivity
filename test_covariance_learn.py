import covariance_learn
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np


def test__update_rho():
    U = np.random.normal((3, 3))
    mu = 10.
    tau_incr = 2.
    tau_decr = 2.
    s = 1.
    rho = 1.

    # s > mu * r
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [s], [(1 + mu) * s], mu, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho / tau_decr)
    assert_allclose(U_, U * tau_decr)

    # r > mu * s
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [(1 + mu) * s], [s], mu, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho * tau_incr)
    assert_allclose(U_, U / tau_incr)

    # r = s
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [s], [s], mu, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho)
    assert_allclose(U_, U)

    # s > mu * r
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [s], [(1 + mu) * s], None, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho)
    assert_allclose(U_, U)

    # r > mu * s
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [(1 + mu) * s], [s], None, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho)
    assert_allclose(U_, U)

    # r = s
    U_ = U.copy()
    rho_ = covariance_learn._update_rho(
        rho, U_, [s], [s], None, tau_incr, tau_decr)
    assert_almost_equal(rho_, rho)
    assert_allclose(U_, U)
