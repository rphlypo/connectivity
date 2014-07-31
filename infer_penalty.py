import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from joblib import Parallel, delayed


import compute_hierarchical_parcellation as chp
import covariance_learn as covl


# some aliases, allowing to reuse computed data
# data has been masked with the hierarchical k-means mask
# the mask is a tree-level octary hierarchy (512 clusters)
mem = chp.mem
subj_dirs = chp.subject_dirs
get_data = mem.cache(chp.get_data)


def _get_train_ix(subject_dir, n_iter=3):
    """ get KFold indices for the training sets

    we cannot simply use KFold from scikit-learn here, because we need to
    sample in a stratified manner from both LR and RL given a session. Also,
    train and validation must both be taken from different sessions, whereas
    testing will be done only afterwards on different subjects.

    Arguments:
    ---------
    subject_dir : path string
        points to the subject entry in the HCP data tree

    n_iter : int
        number of splits of LR/RL for a given session

    returns:
    -------
    X : list of dicts
        each entry is a dictionary with keys
        * session
        * scan
        * data
        * train_ix
        * test_ix
    """
    X = get_data(subject_dir)
    if len(X) < 4:
        scan_sessions = {(1, 'LR'), (1, 'RL'), (2, 'LR'), (2, 'RL')}
        scan_sessions_ = [(x['session'], x['scan']) for x in X]
        missing = set(scan_sessions) - set(scan_sessions_)
        ex_str = 'Expecting 2 sessions, 2 scan types: subject {} is missing {}'
        raise ValueError(ex_str.format(subject_dir[22:28], missing))
    # sample from session i equally over LR/RL using 1/3 of the data
    for session in [1, 2]:
        n = min([x["data"].shape[0] for x in X if x["session"] == session])
        n_train = n / n_iter
        train_ix = [np.arange(j * n_train, (j + 1) * n_train)
                    for j in range(n_iter)]
        test_ix = np.arange(n)
        for j in range(len(X)):
            if X[j]["session"] == session:
                X[j]["train_ix"] = train_ix
                X[j]["test_ix"] = test_ix
    return X


def cross_val_subject(subject_dir, method='gl',
                      base_estimator=None,
                      alpha_prec=1e-2, **kwargs):
    ''' cross-validation to estimate subject-specific hyper-parameter

    Cross-validation is based on a KFold strategy with a split over the two
    available sessions, stratification is done wrt scan mode LR/RL. Due to the
    high temporal resolution (TR=0.72), temporal dependency between consecutive
    samples is high, which is why KFolds are used with a slightly higher number
    of samples than usually available from experiments with higher TR-values.

    Arguments:
    ---------
    subject_dir : path string
        points to the subject entry in the HCP data tree

    method : string
        chooses between
        * graphical lasso ('gl')
        * hierarchical graphical lasso ('hgl')
        * group graphical lasso ('ggl')

    base_estimator : string, or Estimator object for covariance estimation
        all of the estimators take a base_estimator to estimate covariance
        either use 'EmpiricalCovariance', or 'LedoitWolf' if an object is
        passed, options are, e.g., EmpiricalCovariance(assume_centered=True)

    random_state : int, instance of RandomGenerator, or None
        the initial random state, for replicability

    all other arguments are passed on to the covariance estimator determined
    by method

    Returns:
    -------
    alpha : float
        near-optimal penalisation parameter

    support : np.ndarray of dtype Boolean with shape (p, p)
        support set estimated at alpha
    '''
    if isinstance(base_estimator, basestring):
        if base_estimator == 'EmpiricalCovariance':
            # assume_centered is True since baseline has been subtracted
            base_estimator = EmpiricalCovariance(assume_centered=True)
        elif base_estimator == 'LedoitWolf':
            base_estimator = LedoitWolf(assume_centered=True)
    if base_estimator is None:
        base_estimator = EmpiricalCovariance(assume_centered=True)

    try:
        X = _get_train_ix(subject_dir)
    except ValueError:
        return
    alpha_max(X, cov_estimator=base_estimator)

    if method == 'gl':
        cov_learner = covl.GraphLasso
    elif method == 'hgl':
        cov_learner = covl.HierarchicalGraphLasso
    elif method == 'ggl':
        raise ValueError('Not implemented yet, coming soon')
        cov_learner = covl.GroupGraphLasso
    for session in [1, 2]:
        alpha_grid = np.linspace(0, 1, 5)
        score_grid = np.zeros((5,))
        for (ix_alpha, alpha) in enumerate(alpha_grid):
            if ix_alpha in [1, 3]:
                continue
            cov_learner_ = cov_learner(alpha, base_estimator=base_estimator,
                                       **kwargs)
            score_grid[ix_alpha] = _compute_score(X, cov_learner_)

        while alpha_grid[3] - alpha_grid[1] > alpha_prec:
            for (ix_alpha, alpha) in enumerate(alpha_grid):
                if ix_alpha not in [1, 3]:
                    continue
                cov_learner_ = cov_learner(alpha,
                                           base_estimator=base_estimator,
                                           **kwargs)
                score_grid[ix_alpha] = _compute_score(X, cov_learner_,
                                                      sessions=session)
            ix_max = min(3, max(1, np.argmax(score_grid)))
            alpha_opt = alpha_grid[ix_max]
            score_opt = score_grid[ix_max]
            print alpha_grid
            print score_grid
            alpha_grid[[0, 2, 4]] = alpha_grid[[ix_max - 1, ix_max,
                                                ix_max + 1]]
            alpha_grid[[1, 3]] = alpha_grid[[0, 2]] + \
                np.diff(alpha_grid[[0, 2, 4]]) / 2.
            score_grid[[0, 2, 4]] = score_grid[[ix_max - 1, ix_max,
                                                ix_max + 1]]
            for j in range(len(X)):
                if X[j]["session"] == session:
                    X[j]["alpha_opt"] = alpha_opt
                    X[j]["score_opt"] = score_opt
    return X


def alpha_max(X, cov_estimator=EmpiricalCovariance(assume_centered=True)):
    """ get the maximum penalisation parameter alpha_max for given data

    X is a mutable list passed by reference, so no need to return X!
    """
    for session in [1, 2]:
        alpha_max = list()
        Xs = [x for x in X if x["session"] == session]
        for train_ix in Xs[0]["train_ix"]:
            Y = np.concatenate([Xs[0]["data"][train_ix, :],
                                Xs[1]["data"][train_ix, :]],
                               axis=0)
            C = cov_estimator.fit(Y).covariance_
            C = covl._cov_2_corr(C)
            alpha_max.append(np.max(np.abs(C - np.diag(np.diag(C)))))
        for j in range(len(X)):
            if X[j]["session"] == session:
                X[j]["alpha_max"] = alpha_max


def _compute_score(X, cov_learner, sessions=[1, 2]):
    score = 0
    alpha = cov_learner.__getattribute__('alpha')
    if not hasattr(sessions, '__iter__'):
        sessions = [sessions]
    for session in sessions:
        X_train = [x for x in X if x["session"] == session]
        X_test = np.concatenate([x["data"][x["test_ix"], :]
                                 for x in X if x["session"] == 3 - session],
                                axis=0)
        train_ixs = X_train[0]["train_ix"]
        max_alphas = X_train[0]["alpha_max"]
        for (train_ix, alpha_max) in zip(train_ixs, max_alphas):
            X_train_ = np.concatenate([X_train[0]["data"][train_ix, :],
                                       X_train[1]["data"][train_ix, :]],
                                      axis=0)
            cov_learner.__setattr__('alpha', alpha * alpha_max)
            cov_learner.fit(X_train_)
            score += cov_learner.score(X_test)
    return score


if __name__ == "__main__":
    cv_subj = mem.cache(cross_val_subject)
    # cv_subj(subj_dirs[0])
    Parallel(n_jobs=10)(delayed(cv_subj)(subj_dir) for subj_dir in subj_dirs)
