from numpy.testing import assert_almost_equal, assert_array_less, assert_equal
import numpy as np
import scipy.linalg
import fista_gl
# reload(fista_gl)


p = 5
Theta = np.random.normal(size=(p, 2 * p))
Theta = Theta.dot(Theta.T) / p
S = np.random.normal(size=(p, 2 * p))
S = S.dot(S.T) / p
Theta_ = scipy.linalg.eigh(Theta)


def test_f_graphlasso():
    # whether or not eigendecomposition is used, result should be the same
    assert_almost_equal(fista_gl.f_graphlasso(Theta, S),
                        fista_gl.f_graphlasso(Theta_, S))
    # minimum at max likelihood solution inv(S)
    assert_array_less(fista_gl.f_graphlasso(scipy.linalg.inv(S), S),
                      fista_gl.f_graphlasso(Theta, S))


def test_grad_f_graphlasso():
    Theta_inv = Theta_[1].dot(np.diag(1. / Theta_[0]).dot(Theta_[1].T))
    # the optimal should be when S = inv(Theta), hence zero gradient
    assert_almost_equal(fista_gl.grad_f_graphlasso(Theta, Theta_inv),
                        np.zeros((p, p)))
    assert_almost_equal(fista_gl.grad_f_graphlasso(Theta, Theta_inv),
                        np.zeros((p, p)), Hessian=True)
    # test whether this works for the eigen decomposition
    assert_almost_equal(fista_gl.grad_f_graphlasso(Theta_, Theta_inv),
                        np.zeros((p, p)))
    assert_almost_equal(fista_gl.grad_f_graphlasso(Theta_, Theta_inv),
                        np.zeros((p, p)), Hessian=True)


def test_pL_graphlasso():
    grad = fista_gl.grad_f_graphlasso(Theta, S)
    pL_Theta = fista_gl.pL_graphlasso(Theta, .5, grad, 1.)
    # in absolute value the soft thresholding should have lower values
    assert_array_less(np.abs(pL_Theta), np.abs(Theta - grad))
    pL_sign = np.sign(pL_Theta)
    T_sign0 = np.abs(np.sign(Theta - grad)) < 1. / 2
    # if not put to zero, sign should not change
    assert_equal(np.sign(pL_Theta) * pL_sign,
                 np.sign(Theta - grad) * pL_sign)
    # zeros in original Theta must remain zero
    assert_equal(np.sign(pL_Theta) * T_sign0,
                 np.sign(Theta - grad) * T_sign0)


def test_Q_graphlasso():
    grad = fista_gl.grad_f_graphlasso(Theta, S)
    Z = np.random.normal(size=(p, 2 * p))
    Z = Z.dot(Z.T) / p
    f_Theta = fista_gl.f_graphlasso(Theta_, S)
    f_Z = fista_gl.f_graphlasso(Z, S)
    g_Z = fista_gl.g_graphlasso(Z, .5)
    L = np.sqrt(np.sum(1. / scipy.linalg.eigvalsh(Z) ** 2))
    L *= np.sqrt(np.sum(1. / Theta_[0] ** 2))
    assert_array_less(f_Z + g_Z,
                      fista_gl.Q_graphlasso(Z, Theta, grad, L, .5, f_Theta))
    # Does this hold for the Hessian update ?
    gradH = fista_gl.grad_f_graphlasso(Theta, S, Hessian=True)
    assert_array_less(f_Z + g_Z,
                      fista_gl.Q_graphlasso(Z, Theta, gradH, L, .5, f_Theta))
