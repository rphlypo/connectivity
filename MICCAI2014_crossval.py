import numpy as np
import covariance_learn as cvl
import scipy.linalg as linalg
from joblib import Parallel, Memory, delayed
import glob
from sklearn.utils import check_random_state
import os

from compute_hierarchical_parcellation import get_data

ROOT_DIR = '/storage'

subject_dirs = sorted(glob.glob(
    os.path.join(ROOT_DIR, 'data/HCP/S500-1/*/MNINonLinear/Results')))

N_JOBS = 10

mem = Memory(cachedir='/storage/workspace/rphlypo/hierarchical/joblib')


def cross_val(subject_dir, labels_img="labels_level_3.nii.gz",
              method=None, sess_ix=None,
              random_state=None, **kwargs):
    print "processing {}".format(subject_dir)
    randgen = check_random_state(random_state)
    get_data_ = mem.cache(get_data)
    subj_data = get_data_(subject_dir, labels_img=labels_img)
    return subj_data


if __name__ == "__main__":
    Parallel(n_jobs=20)(delayed(cross_val)(sd) for sd in subject_dirs)
    Parallel(n_jobs=20)(delayed(cross_val)(sd, labels_img='HOmaps.nii.gz')
        for sd in subject_dirs)
