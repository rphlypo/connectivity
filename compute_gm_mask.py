"""
Compute a grey matter mask from the SPM Tissue Probability Masks
"""

import os
import numpy as np
import socket

from scipy import ndimage
from nilearn.image import resample_img
import nibabel

try:
    SPM_DIR = os.environ['SPM_DIR']
    print "located SPM at {}".format(SPM_DIR)
except KeyError as ke:
    print "{} not found on your system's path".format('SPM_DIR')
    raise ke

if socket.gethostname == 'drago':
    ROOT_DIR = '/storage'
elif socket.gethostname()[:3] == 'is':
    ROOT_DIR = '/volatile'

grey_matter_template = os.path.join(SPM_DIR, 'tpm', 'grey.nii')

# An image from the HCP
print "loading subject data to obtain affine for HCP data"
hcp_mask = nibabel.load(os.path.join(
    ROOT_DIR, 'data/HCP/Q2/585862/MNINonLinear/Results/',
    'rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'))

affine = hcp_mask.get_affine()
shape = hcp_mask.shape[:3]
hcp_mask = hcp_mask.get_data()[..., 0] > 1e-13

gm_img = nibabel.load(grey_matter_template)
gm_img = resample_img(gm_img, target_affine=affine, target_shape=shape)
gm_map = gm_img.get_data()
gm_mask = (gm_map > .33)

gm_mask = ndimage.binary_closing(gm_mask, iterations=2)
gm_mask = np.logical_and(gm_mask, hcp_mask).astype(np.int)


mask_img = nibabel.Nifti1Image(gm_mask, affine)


nibabel.save(mask_img, 'gm_mask.nii')
