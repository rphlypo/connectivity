"""
Apply a divisive K-means strategy to have a hierarchical set of parcells
"""
import glob
import os

import numpy as np
from joblib import Memory, Parallel, delayed

from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans, KMeans
import nibabel

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn.input_data import NiftiMasker
from nilearn.image import high_variance_confounds
from nilearn._utils import check_niimg
from nilearn import masking
from nilearn import signal

subject_dirs = sorted(glob.glob('/storage/data/HCP/Q2/*/MNINonLinear/Results'))

N_JOBS = 36


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
    files = sorted(glob.glob(os.path.join(subject_dir,
            'rfMRI_REST?_??/rfMRI_REST?_??.nii.gz')))

    # A set of different confounds
    hv_confounds = Parallel(n_jobs=N_JOBS)(delayed(high_variance_confounds)(
                        f, mask_img=mask_img) for f in files)
    out_brain = Parallel(n_jobs=N_JOBS)(delayed(out_brain_confounds)(
                        f, mask_img=mask_img) for f in files)
    mvt = [np.loadtxt(f) for f in
           sorted(glob.glob(os.path.join(subject_dir,
            'rfMRI_REST?_??/Movement_Regressors.txt')))]
    mvt_dt = [np.loadtxt(f) for f in
           sorted(glob.glob(os.path.join(subject_dir,
            'rfMRI_REST?_??/Movement_Regressors_dt.txt')))]
    confounds = [np.concatenate((h, m, mt, ob), axis=1)
                 for h, m, mt, ob in zip(hv_confounds, mvt, mvt_dt, out_brain)]

    multi_pca = MultiPCA(mask=mask_img, smoothing_fwhm=smoothing_fwhm,
                    t_r=.7, low_pass=.1,
                    n_components=n_components, do_cca=False,
                    n_jobs=N_JOBS, verbose=12)
    multi_pca.fit(files, confounds=confounds)
    return multi_pca.components_ * multi_pca.variance_[:, np.newaxis]


###############################################################################
mem = Memory(cachedir='/storage/workspace/tmp/gael_joblib')

def subject_pca_cached(*args, **kwargs):
    return mem.cache(subject_pca).call_and_shelve(*args, **kwargs)


# Run a first call outside parallel computing, to debug easily
result = subject_pca_cached(subject_dirs[0], n_components=256,
                            smoothing_fwhm=6,
                            mask_img='gm_mask.nii')


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



