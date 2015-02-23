"""
Apply a divisive K-means strategy to have a hierarchical set of parcels
"""
import glob
import os
import socket

from multiprocessing import cpu_count
from getpass import getuser
import copy
import numpy as np
from joblib import Memory, Parallel, delayed

import itertools

from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.utils import check_random_state
import nibabel
import covariance_learn as cvl
reload(cvl)

from sklearn.covariance import LedoitWolf
from sklearn.covariance import EmpiricalCovariance

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker
from nilearn.image import high_variance_confounds
from nilearn._utils import check_niimg
from nilearn import masking
from nilearn import signal

import htree
reload(htree)

if getuser() == 'rphlypo' and socket.gethostname() != 'drago':
    ROOT_DIR = '/volatile'
    N_JOBS = 2
elif getuser() == 'phlypor' and socket.gethostname() == 'desktop-315':
    ROOT_DIR = '/localdata/phlypor'
    N_JOBS = 2
else:
    ROOT_DIR = '/storage'
    N_JOBS = min(cpu_count() - 4, 36)


subject_dirs = sorted(glob.glob(
    os.path.join(ROOT_DIR, 'data/HCP/S500-?/' + ('[0-9]' * 6) +
                 '/MNINonLinear/Results')))

# a 8-ary tree is constructed as a list, because otherwise the tree is
# not a 'constant' for the caching
HTREE = htree.construct_tree(obj=False, rng=0)


def out_brain_confounds(epi_img, mask_img):
    """ Return the 5 principal components of the signal outside the
        brain.
    """
    mask_img = check_niimg(mask_img)
    mask_img = nibabel.Nifti1Image(
        np.logical_not(mask_img.get_data()).astype(np.int),
        mask_img.get_affine())
    sigs = masking.apply_mask(epi_img, mask_img)
    # Remove the constant signals
    non_constant = np.any(np.diff(sigs, axis=0) != 0, axis=0)
    sigs = sigs[:, non_constant]
    sigs = signal.clean(sigs, detrend=True)
    U, s, V = randomized_svd(sigs, 5, random_state=0)
    return U


def makeHOmaps():
    HOmaps = [
        '/usr/share/fsl/data/atlases/HarvardOxford/' +
        'HarvardOxford-cortl-prob-2mm.nii.gz',
        '/usr/share/fsl/data/atlases/HarvardOxford/' +
        'HarvardOxford-sub-prob-2mm.nii.gz']
    HOmaps_ = [nibabel.load(HOmap) for HOmap in HOmaps]
    map_data = [HOmap.get_data() for HOmap in HOmaps_]
    map_data = np.concatenate(map_data, axis=-1)
    map_affine = HOmaps_[0].get_affine()
    nibabel.save(nibabel.Nifti1Image(map_data, map_affine), 'HOmaps.nii.gz')


def subject_pca(subject_dir, n_components=512, smoothing_fwhm=6,
                mask_img='gm_mask.nii'):
    files = sorted(glob.glob(os.path.join(
        subject_dir, 'rfMRI_REST?_??/rfMRI_REST?_??.nii.gz')))

    confounds = get_confounds(subject_dir, files, mask_img=mask_img)

    multi_pca = MultiPCA(mask=mask_img, smoothing_fwhm=smoothing_fwhm,
                         t_r=.7, low_pass=.1,
                         n_components=n_components, do_cca=False,
                         n_jobs=N_JOBS, verbose=12)
    multi_pca.fit(files, confounds=confounds)
    return multi_pca.components_ * multi_pca.variance_[:, np.newaxis]


def get_confounds(subject_dir, files, mask_img='gm_mask.nii'):
    # A set of different confounds
    hv_confounds = Parallel(n_jobs=N_JOBS)(delayed(high_variance_confounds)(
        f, mask_img=mask_img) for f in files)
    out_brain = Parallel(n_jobs=N_JOBS)(delayed(out_brain_confounds)(
        f, mask_img=mask_img) for f in files)
    mvt = [np.loadtxt(f) for f in
           sorted(glob.glob(os.path.join(
               subject_dir, 'rfMRI_REST?_??/Movement_Regressors.txt')))]
    mvt_dt = [np.loadtxt(f) for f in
              sorted(glob.glob(os.path.join(
                  subject_dir, 'rfMRI_REST?_??/Movement_Regressors_dt.txt')))]
    return [np.concatenate((h, m, mt, ob), axis=1)
            for h, m, mt, ob in zip(hv_confounds, mvt, mvt_dt, out_brain)]


def get_data(subject_dir, labels_img='labels_level_3.nii.gz',
             mask_img='gm_mask.nii', smoothing_fwhm=6):
    if not os.path.exists(mask_img):
        os.system('python compute_gm_mask.py')
    files = sorted(glob.glob(os.path.join(
        subject_dir, 'rfMRI_REST?_??/rfMRI_REST?_??.nii.gz')))
    confounds = get_confounds(subject_dir, files, mask_img=mask_img)
    if labels_img == 'labels_level_3.nii.gz':
        masker = NiftiLabelsMasker(labels_img=labels_img,
                                   mask_img=mask_img,
                                   smoothing_fwhm=smoothing_fwhm,
                                   standardize=True,
                                   resampling_target='labels',
                                   detrend=True, low_pass=.1, t_r=.7)
    elif labels_img == 'HOmaps.nii.gz':
        if not os.path.isfile(labels_img):
            makeHOmaps()
        masker = NiftiMapsMasker(maps_img=labels_img,
                                 mask_img=mask_img,
                                 smoothing_fwhm=smoothing_fwhm,
                                 resampling_target='maps',
                                 detrend=True, low_pass=.1, t_r=.7)

    subj_data = []
    for (f_ix, f) in enumerate(files):
        base_name = os.path.basename(f)
        subj_data.append(
            {'data': masker.fit_transform(f,
                                          confounds=confounds[f_ix]),
             'session': int(base_name[10]),
             'scan': base_name[12:14]})
    return subj_data


###############################################################################
if getuser() == 'rphlypo' and socket.gethostname() == 'is151225':
    mem = Memory(cachedir='/volatile/workspace/tmp/connectivity_joblib')
elif socket.gethostname() == 'drago' and getuser() == 'rphlypo':
    mem = Memory(cachedir='/storage/workspace/rphlypo/hierarchical/joblib')
elif socket.gethostname() == 'desktop-315' and getuser() == 'phlypor':
    mem = Memory(cachedir='/localdata/phlypor/workspace')
else:
    mem = Memory(cachedir='/storage/workspace/tmp/gael_joblib')


def subject_pca_cached(*args, **kwargs):
    return mem.cache(subject_pca).call_and_shelve(*args, **kwargs)


def compute_pca(subject_dirs, mask_img='gm_mask.nii'):
    """ Compute a PCA on the whole dataset
    """
    results = Parallel(n_jobs=N_JOBS, verbose=60)(
        delayed(subject_pca_cached)(subject_dir, n_components=256,
                                    smoothing_fwhm=6)
        for subject_dir in subject_dirs)

    # Concatenate the data and do a randomized PCA
    subject_data = results[0].get()
    data = np.empty((len(results) * 256, subject_data.shape[1]))

    for idx, subject_pca in enumerate(results):
        data[idx * 256: (idx + 1) * 256] = subject_pca.get()

    print 'Starting SVD...'
    U, s, V = randomized_svd(data, 512, random_state=0)
    V *= s[:, np.newaxis]
    return V


def do_k_means(data, n_clusters):
    k_means = KMeans(n_clusters=n_clusters, random_state=0,
                     n_jobs=N_JOBS, max_iter=500, tol=1e-5, n_init=40)
    k_means.fit(data.T)
    return k_means.labels_



def compare_hgl_gl(subject_dir, random_state=None,
                   estimators=['EMP', 'LW'], methods=['hgl', 'gl'],
                   alpha_tol=1e-1, h_tol=1e-1, get_data_=None,
                   sess_ix=None):
    """ compare results from hierarchical and plain graphical lasso

    using a given base_estimator for the covariance (empirical or
    Ledoit-Wolf) the goal is to estimate the optimal alpha and h
    parameters by cross-validation

    Arguments:
    ----------
    subject_dir : string
        subject directory where the data resides

    random_state : random state or None
        if the random state needs to be state fixed for
        reproducibility of the results, then one may use the option
        'subject' here

    estimators : list of strings
        available methods are 'EMP' and 'LW' for empirical covariance
        and Ledoit-Wolf shrinkage, respectively

    Returns:
    --------
    results : dictionary with entries [estimator][method][value]

    This function passes on parameters to compute_optimal_parameters,
    its main goal is to cache this function and prepare the parameters
    for the different settings (estimators and methods)
    """
    # initialise the data structure containing the results
    if random_state == 'subject':
        random_state = int(split_path(subject_dir)[-3])
    results = dict()
    results['LW'] = {'hgl': {'score': None, 'alpha': None, 'h': None},
                     'gl': {'score': None, 'alpha': None}}
    results['EMP'] = {'hgl': {'score': None, 'alpha': None, 'h': None},
                      'gl': {'score': None, 'alpha': None}}
    # initialise the random generator
    randgen = check_random_state(random_state)
    # cache the function get_data_ so that no recomputation is required
    if get_data_ is None:
        get_data_ = mem.cache(get_data)
    cross_val = mem.cache(cvl.cross_val)
    # ... and get the data
    subj_data = get_data_(subject_dir)

    # if we do not have two session, each with two scan_modes,
    # something must have gone wrong !
    if len(subj_data) < 4:
        return results
        # raise ValueError('Incomplete data')
    # pick a random session for training
    if sess_ix is None:
        sess_ix = randgen.randint(2) + 1
    X = np.concatenate([d["data"] for d in subj_data
                        if d["session"] == sess_ix], axis=0)
    X_len = [d["data"].shape[0] for d in subj_data if d["session"] == sess_ix]
    # create an indicator for LR / RL, so stratified sampling is possible
    X_samplinglabel = np.zeros((X_len[0] + X_len[1],), dtype=np.int)
    X_samplinglabel[X_len[0]:] = 1
    # repeat for the complementary session
    Xtest = np.concatenate([d["data"] for d in subj_data
                            if d["session"] == 3 - sess_ix], axis=0)
    Xtest_len = [d["data"].shape[0] for d in subj_data
            if d["session"] == 3 - sess_ix]
    Xtest_samplinglabel = np.zeros((Xtest_len[0] + Xtest_len[1],),
                                   dtype=np.int)
    Xtest_samplinglabel[Xtest_len[0]:] = 1

    # start looping over methods and estimators
    for estimator, method in itertools.product(estimators, methods):
        if estimator == 'EMP':
            base_estimator = EmpiricalCovariance(assume_centered=True)
        elif estimator == 'LW':
            base_estimator = LedoitWolf(assume_centered=True)
        res = cross_val(X, y=X_samplinglabel,
                        X_test=Xtest, y_test=Xtest_samplinglabel,
                        method=method, n_iter=1,
                        train_size=.9, test_size=.05, retest_size=.2,
                        n_jobs=min({N_JOBS, 10}), htree=HTREE,
                        random_state=random_state,
                        base_estimator=base_estimator,
                        tol=1e-3, ips_flag=True)
        results[estimator][method]['score'] = res[1][-1]
        results[estimator][method]['alpha'] = res[0]
        if method == 'hgl':
            results[estimator][method]['h'] = res[2]
    return results


def run_analysis(subject_dirs=subject_dirs, n_jobs=N_JOBS,
                 random_state="subject", **kwargs):
    """ starting point for the analysis, allows for subject parallelisation
    """
    if len(subject_dirs) > 1:
        results = Parallel(n_jobs=N_JOBS)(delayed(compare_hgl_gl)(
            subject_dir=sd, random_state=random_state, **kwargs)
            for sd in subject_dirs)
    else:
        results = compare_hgl_gl(subject_dir=subject_dirs[0],
                                 random_state=random_state, **kwargs)
    return results


def split_path(p):
    head, tail = os.path.split(p)
    p = []
    p.append(tail)
    while tail:
        head, tail = os.path.split(head)
        p.append(tail)
    p.reverse()
    return p


if __name__ == "__main__":
    # Run a first call outside parallel computing, to debug easily
    result = subject_pca_cached(subject_dirs[0], n_components=256,
                                smoothing_fwhm=6,
                                mask_img='gm_mask.nii')
    data = mem.cache(compute_pca)(subject_dirs[::-1])

    k_means = MiniBatchKMeans(n_clusters=8, random_state=0)

    labels = mem.cache(do_k_means)(data, n_clusters=8)
    labels += 1

    masker = NiftiMasker(mask='gm_mask.nii').fit()
    cluster_map = masker.inverse_transform(labels)
    cluster_map.to_filename('labels_level_1.nii')

    # Scheme for numbering a global hierarchy: 4 levels, first one at x000
    labels *= 1000
    for level in [2, 3, 4]:
        for label in np.unique(labels):
            this_data = data[:, labels == label]
            this_labels = mem.cache(do_k_means)(this_data, n_clusters=8)
            this_labels += 1
            this_labels *= 10 ** (4 - level)
            labels[labels == label] = label + this_labels
        masker = NiftiMasker(mask='gm_mask.nii').fit()
        cluster_map = masker.inverse_transform(labels)
        cluster_map.to_filename('labels_level_%i.nii' % level)
