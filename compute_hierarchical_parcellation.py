"""
Apply a divisive K-means strategy to have a hierarchical set of parcels
"""
import glob
import os
import socket
import operator

from multiprocessing import cpu_count
from getpass import getuser

import numpy as np
from joblib import Memory, Parallel, delayed

from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans, KMeans
import nibabel
import covariance_learn as cvl

from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.utils import check_random_state

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.image import high_variance_confounds
from nilearn._utils import check_niimg
from nilearn import masking
from nilearn import signal

import htree

if getuser() == 'rphlypo' and socket.gethostname() != 'drago':
    ROOT_DIR = '/volatile'
else:
    ROOT_DIR = '/storage'

subject_dirs = sorted(glob.glob(
    os.path.join(ROOT_DIR, 'data/HCP/Q2/*/MNINonLinear/Results')))

N_JOBS = min(cpu_count() - 4, 36)

TREE = htree.construct_tree()


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
    masker = NiftiLabelsMasker(labels_img=labels_img, mask_img=mask_img,
                               smoothing_fwhm=smoothing_fwhm,
                               standardize=True, resampling_target='labels',
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


def compute_optimal_params(subject_dir, method='hgl', sess_ix=None,
                           random_state=None,
                           base_estimator=EmpiricalCovariance(
                               assume_centered=True),
                           **kwargs):
    get_data_ = mem.cache(get_data)
    subj_data = get_data_(subject_dir)
    # random session for training
    if sess_ix is None:
        randgen = check_random_state(random_state)
        sess_ix = randgen.randint(2) + 1
    if len(subj_data) < 4:
        raise ValueError("not all sessions present")
    X = np.concatenate([d["data"] for d in subj_data
                        if d["session"] == sess_ix], axis=0)
    y = np.concatenate([np.zeros((d['data'].shape[0],))
                        if d['scan'] == 'LR'
                        else np.ones((d['data'].shape[0],))
                        for d in subj_data
                        if d['session'] == sess_ix])
    # complementary session
    Y = np.concatenate([d["data"] for d in subj_data
                        if d["session"] == 3 - sess_ix], axis=0)
    base_estimator.fit(Y)
    Theta = base_estimator.precision_
    S = base_estimator.covariance_
    return cvl.cross_val(X, y, method=method, alpha_tol=1e-2, n_iter=6,
                         optim_h=True, train_size=1. / 6, test_size=2,
                         model_prec=Theta, model_cov=S,
                         n_jobs=min({N_JOBS, 10}), random_state=random_state,
                         tol=1e-2, **kwargs)


def compare_hgl_gl(subject_dir=subject_dirs[0], random_state=None):
    randgen = check_random_state(random_state)
    sess_ix = randgen.randint(2) + 1
    results = list()
    comp_opt_params = mem.cache(compute_optimal_params)
    try:
        res = comp_opt_params(subject_dir, method='hgl', sess_ix=sess_ix,
                              htree=TREE)
        print res
        raise Exception
        results.append({'method': 'hgl',
                        'cov': 'empirical',
                        'score': res[1][-1],
                        'alpha': res[0],
                        'h': res[2]})

        res = comp_opt_params(subject_dir, method='hgl', sess_ix=sess_ix,
                              htree=TREE,
                              base_estimator=LedoitWolf(assume_centered=True))
        results.append({'method': 'hgl',
                        'cov': 'LedoitWolf',
                        'score': res[1][-1],
                        'alpha': res[0],
                        'h': res[2]})

        res = comp_opt_params(subject_dir, method='gl', sess_ix=sess_ix)
        results.append({'method': 'gl',
                        'cov': 'empirical',
                        'score': res[1][-1],
                        'alpha': res[0]})

        res = comp_opt_params(subject_dir, method='gl', sess_ix=sess_ix,
                              base_estimator=LedoitWolf(assume_centered=True))
        results.append({'method': 'gl',
                        'cov': 'LedoitWolf',
                        'score': res[1][-1],
                        'alpha': res[0]})
        return results
    except ValueError:
        pass


def run_analysis(subject_dirs=subject_dirs, n_jobs=1, random_state=12345):
    compare_hgl_gl_ = mem.cache(compare_hgl_gl)
    results = Parallel(n_jobs=n_jobs)(delayed(compare_hgl_gl_)(
        sd, random_state=random_state) for sd in subject_dirs)
    results_ = [r for r in results if r is not None]
    return reduce(operator.add, results_)


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

    # Scheme for numbering a global hierarchi: 4 level, first one is at x000
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
