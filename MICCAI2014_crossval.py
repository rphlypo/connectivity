# import numpy as np
# import covariance_learn as cvl
# import scipy.linalg as linalg
from joblib import Parallel, Memory, delayed
import glob
# from sklearn.utils import check_random_state
import os

import compute_hierarchical_parcellation as chp

from compute_hierarchical_parcellation import get_data

ROOT_DIR = '/storage'

subject_dirs = sorted(glob.glob(
    os.path.join(ROOT_DIR, 'data/HCP/S500-?/*/MNINonLinear/Results')))

N_JOBS = 10

mem = Memory(cachedir=os.path.join(
    ROOT_DIR, 'workspace/rphlypo/data_extraction'))


def cross_val(subject_dir, labels_img="labels_level_3.nii.gz",
              method=None, sess_ix=None,
              random_state=None, **kwargs):
    print "processing {} with atlas {}".format(subject_dir, labels_img)
    # randgen = check_random_state(random_state)
    get_data_ = mem.cache(get_data)
    results = chp.run_analysis(subject_dirs=subject_dirs, get_data_=get_data_)
    # subj_data = get_data_(subject_dir, labels_img=labels_img)
    return results


if __name__ == "__main__":
    res = {'hier': [], 'HO': []}
    res['hier'] = Parallel(n_jobs=20)(delayed(cross_val)(sd)
                                      for sd in subject_dirs)
    res['HO'] = Parallel(n_jobs=20)(delayed(cross_val)(
                                    sd, labels_img='HOmaps.nii.gz')
                                    for sd in subject_dirs)
