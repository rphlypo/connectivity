
# coding: utf-8

# In[1]:

import covariance_learn
import itertools
import phase_transition
import htree
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special
from functools import partial
from sklearn.utils import check_random_state
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed


def alpha_func_(h, max_level):
    return partial(covariance_learn._alpha_func, h=h, max_level=max_level)


def cov2corr(cov):
    return np.diag(1 / np.sqrt(np.diag(cov))).dot(
        cov.dot(np.diag(1 / np.sqrt(np.diag(cov)))))


def estimate_precision(alpha, h=None, method='hgl', max_level=None,
                       title=None):
    if max_level is None:
        max_level = max([lev for (_, lev) in tree.root_.get_descendants()])
    alpha_func = alpha_func_(h=h, max_level=max_level)
    if method == 'gl':
        covl = covariance_learn.GraphLasso(alpha=alpha, score_norm='KL',
                                           max_iter=1e4, rho=2)
    elif method == 'hgl':
        covl = covariance_learn.HierarchicalGraphLasso(alpha=alpha, htree=tree,
                                                       alpha_func=alpha_func,
                                                       score_norm='KL',
                                                       max_iter=1e4, rho=2)
    covl.fit(X)
    K = covl.precision_
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.matshow(np.ma.masked_equal(K * (covl.auxiliary_prec_ != 0), 0),
                cmap=plt.cm.RdBu_r, vmin=-np.max(np.abs(K)) / 5.,
                vmax=np.max(np.abs(K)) / 5.)
    ax1.set_title('Estimated precision')
    ax1.set_axis_bgcolor('.7')
    ax2.matshow(np.array(covl.auxiliary_prec_ != 0, dtype=float),
                cmap=plt.cm.gray, vmin=0, vmax=1)
    ax2.set_title('Support (derived from split variable)')
    fig.suptitle(title)
    fig, ax = plt.subplots()
    plt.plot(covl.f_vals_, 'b-')
    plt.title("Jensen's divergence = {}".format(covl.score(Y)))
    ax.set_ylabel('energy', color='b')
    for tl in ax.get_yticklabels():
        tl.set_color('b')
    ax2 = ax.twinx()
    ax2.semilogy(covl.var_gap_, 'g-')
    ax2.semilogy(covl.dual_gap_, 'g--')
    ax2.set_ylabel('split diff & dual update norm', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    fig.suptitle(title)
    plt.show()


def grid_evaluation(X, Y, n_h=11, n_a=11):
    h_vals = np.linspace(0., 1., n_h)
    h_ix = np.arange(n_h)
    alpha_vals = np.logspace(-3., 0., n_a)
    a_ix = np.arange(n_a)
    scores = {'KL': np.zeros((n_h, n_a)), 'ell0': np.zeros((n_h, n_a))}
    score_gl = {'KL': np.zeros((n_h, n_a)), 'ell0': np.zeros((n_h, n_a))}

    alpha_star = {'hgl': dict(), 'gl': dict()}
    h_star = {'hgl': dict(), 'gl': dict()}

    for (ix, vals) in zip(itertools.product(h_ix, a_ix),
                          itertools.product(h_vals, alpha_vals)):
        for score_norm in {'KL', 'ell0'}:
            if ix[0] == 0:
                covl = covariance_learn.GraphLasso(
                    alpha=vals[1], score_norm=score_norm, max_iter=1e4,
                    rho=2.)
                score_gl[score_norm][..., ix[1]] = covl.fit(X).score(Y)
            covl = covariance_learn.HierarchicalGraphLasso(
                alpha=vals[1], htree=tree,
                alpha_func=alpha_func_(h=vals[0], max_level=max_level),
                score_norm=score_norm, max_iter=1e4, rho=2.)
            scores[score_norm][ix] = covl.fit(X).score(Y)

    X_, Y_ = np.meshgrid(h_vals, alpha_vals)
    for score in scores.keys():
        alpha_star['hgl'][score] = Y_.flat[np.argmin(scores[score].T)]
        h_star['hgl'][score] = X_.flat[np.argmin(scores[score].T)]
        arg_min_alpha = np.argmin(score_gl[score][0, ...])
        arg_min_h = np.argmin(scores[score][arg_min_alpha, ...])
        h_star['gl'][score] = h_vals[arg_min_h]
        alpha_star['gl'][score] = alpha_vals[arg_min_alpha]
    return scores, score_gl, alpha_star, h_star


def plot_grid(scores=None, score_gl=None, score='KL', transpose=True,
              zlims=None, z_offset=None, y_offset=None, x_offset=None):
    fig = plt.figure()
    if scores is not None and score_gl is not None:
        if score == 'KL':
            plot_scores = np.log(scores[score] / score_gl[score])
            title = "log-difference of Jensen's divergence:" +\
                "log J(hgl) - log J(gl)"
            score_title = 'Jmin'
        elif score == 'ell0':
            plot_scores = scores[score] - score_gl[score]
            title = "support mismatch difference: ell0(hgl) - ell0(gl)"
            score_title = 'ell0min'
    elif scores is not None or score_gl is not None:
        if scores is not None:
            title = "[HGL] "
            plot_scores = scores[score]
        else:
            title = "[GL] "
            plot_scores = score_gl[score]
        if score == 'KL':
            title += "Jensen's divergence"
            score_title = "Jmin"
        elif score == 'ell0':
            title += "support mismatch"
            score_title = "ell0min"
    else:
        raise ValueError('Neither scores, nor score_gl are defined: ' +
                         'nothing to plot')

    h_vals = np.linspace(0., 1., plot_scores.shape[0])
    alpha_vals = np.logspace(-3., 0., plot_scores.shape[1])

    if transpose:
        plot_scores = plot_scores.T
        xlabel = 'h'
        ylabel = r'$\lambda$'
    else:
        ylabel = 'h'
        xlabel = r'$\lambda$'
    if x_offset is None:
        x_offset = 0.
    if y_offset is None:
        y_offset = 1.
    if z_offset is None:
        z_offset = 0.
    ax = fig.gca(projection='3d')
    X_, Y_ = np.meshgrid(h_vals, alpha_vals)
    ax.plot_surface(X_, Y_, plot_scores, rstride=1, cstride=1,
                    cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax.contour(X_, Y_, plot_scores, zdir='z', offset=z_offset,
               cmap=plt.cm.coolwarm)
    ax.contour(X_, Y_, plot_scores, zdir='x', offset=x_offset,
               cmap=plt.cm.coolwarm)
    ax.contour(X_, Y_, plot_scores, zdir='y', offset=y_offset,
               cmap=plt.cm.coolwarm)
    if zlims is not None:
        ax.set_zlim(zlims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(title)
    if scores is not None and score_gl is not None:
        ax.set_title(score_title + '(hgl: lambda={}, h={}) = {}, '.format(
            Y_.flat[np.argmin(scores[score].T)],
            X_.flat[np.argmin(scores[score].T)],
            np.min(scores[score])) +
            score_title + '(gl: lambda={}) = {}'.format(
                Y_.flat[np.argmin(score_gl[score].T)],
                np.min(score_gl[score])))
    elif scores is not None:
        ax.set_title(score_title + '(hgl: lambda={}, h={}) = {}'.format(
            Y_.flat[np.argmin(scores[score].T)],
            X_.flat[np.argmin(scores[score].T)],
            np.min(scores[score])))
    elif score_gl is not None:
            ax.set_title(score_title + '(gl: lambda={}) = {}'.format(
                         Y_.flat[np.argmin(score_gl[score].T)],
                         np.min(score_gl[score])))
    plt.show()


def plot_covariances(X, Theta, Y=None):
    if Y is None:
        eigvals, eigvecs = scipy.linalg.eigh(Theta)
        Y = eigvecs.dot(np.diag(1 / np.sqrt(eigvals))).dot(eigvecs.T)
    plt.figure()
    plt.subplot(221)
    plt.matshow(Theta, cmap=plt.cm.RdBu_r, fignum=False)
    plt.clim((-1, 1))
    plt.title('population precision matrix')
    plt.colorbar()

    plt.subplot(222)
    sample_prec = scipy.linalg.inv(cov2corr(X.T.dot(X)))
    plt.matshow(sample_prec, cmap=plt.cm.RdBu_r,
                fignum=False)
    max_plot = np.max(np.abs(sample_prec)) / 5.
    plt.clim((-max_plot, max_plot))
    plt.title('sample (n={}) precision matrix'.format(X.shape[0]))
    plt.colorbar()

    plt.subplot(223)
    plt.matshow(cov2corr(Y.T.dot(Y)), cmap=plt.cm.RdBu_r, fignum=False)
    plt.clim((-1, 1))
    plt.title('population correlation matrix')
    plt.colorbar()

    plt.subplot(224)
    plt.matshow(cov2corr(X.T.dot(X)), cmap=plt.cm.RdBu_r, fignum=False)
    plt.clim((-1, 1))
    plt.title('sample correlation matrix')
    plt.colorbar()
    plt.show()


def plot_profiles(alpha=1., h=.8, max_level=4, levels=None):
    if levels is None:
        levels = np.arange(1, max_level + 1)
    plt.figure()
    plt.plot(levels, covariance_learn._alpha_func(alpha=alpha, lev=levels, h=h,
                                                  max_level=max_level))
    plt.show()


def lambda_path(n_samples, C, tree):
    n_bootstraps = 10
    X = np.random.normal(size=(np.max(n_samples) * n_bootstraps, C.shape[1]))
    results = Parallel(n_jobs=min(len(n_samples), 20))(
        delayed(covariance_learn.cross_val)(
            X, method='hgl', n_iter=n_bootstraps,
            train_size=np.float(n) / (np.max(n_samples) * n_bootstraps),
            model_prec=Theta, optim_h=True, htree=tree)
        for n in n_samples)
    alpha_opt_, LL_, h_opt_ = zip(*results)
    plt.figure
    plt.plot(n_samples, alpha_opt_)
    plt.show()
    return alpha_opt_, h_opt_


if __name__ == "__main__":
    tree = htree.construct_tree(arity=4, depth=2)
    max_level = max([lev for (_, lev) in tree.root_.get_descendants()])

    # Create sample
    a = -.4
    b = .28
    Theta = phase_transition._get_mx(a, b, mx_type='smith')

    n_samples = 32
    ii = np.random.randint(2 ** 31 - 1)
    print "random state: {}".format(ii)
    random_state = check_random_state(ii)  # 220013245
    X = random_state.normal(size=(n_samples, Theta.shape[0]))

    eigvals, eigvecs = scipy.linalg.eigh(Theta)
    C = np.diag(1 / np.sqrt(eigvals)).dot(eigvecs.T)

    X = X.dot(C)

    Y = C

    plot_covariances(X, Theta, Y)

    scores, score_gl, alpha_star, h_star = grid_evaluation(
        X, Y, n_a=21, n_h=21)

    estimate_precision(alpha=alpha_star['hgl']['KL'], h=h_star['hgl']['KL'],
                       method='hgl',
                       title=r'($\lambda^{*}_{hgl}$, $h^{\star}_{hgl}$)')
    estimate_precision(
        alpha=alpha_star['gl']['KL'], h=h_star['gl']['KL'], method='hgl',
        title=r'($\lambda^{*}_{gl}, h^{\star}_{hgl}|_{\lambda^{\star}_{gl}}$)')
    estimate_precision(alpha=alpha_star['hgl']['KL'], method='gl',
                       title=r'($\lambda^{*}_{hgl}$, 0)')
    estimate_precision(alpha=alpha_star['gl']['KL'], method='gl',
                       title=r'($\lambda^{*}_{gl}$, 0)')

    plot_grid(scores=scores, score='KL', transpose=True)
    plot_grid(scores=scores, score_gl=score_gl, score='KL', transpose=True)

    plot_grid(scores=scores, score='ell0', transpose=True)
    plot_grid(scores=scores, score_gl=score_gl, score='ell0', transpose=True)

    n_samples = np.logspace(1., 3., 9)
    # alpha_opt_, h_opt_ = lambda_path(n_samples, C, tree)
