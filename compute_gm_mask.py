"""
Compute a grey matter mask from the SPM Tissue Probability Masks
"""

import os
import numpy as np

from scipy import ndimage
from nilearn.image import resample_img
import nibabel

SPM_DIR = os.environ['SPM_DIR']

grey_matter_template = os.path.join(SPM_DIR, 'tpm', 'grey.nii')

# The affine of the HCP images (to avoid resampling later)
affine = np.array([[  -2.,    0.,    0.,   90.],
                   [   0.,    2.,    0., -126.],
                   [   0.,    0.,    2.,  -72.],
                   [   0.,    0.,    0.,    1.]])
shape = (91, 109, 91)

gm_img = nibabel.load(grey_matter_template)
gm_img =resample_img(gm_img, target_affine=affine, target_shape=shape)
gm_map = gm_img.get_data()
gm_mask = (gm_map > .33)

gm_mask = ndimage.binary_opening(gm_mask).astype(np.int)

mask_img = nibabel.Nifti1Image(gm_mask, affine)


nibabel.save(mask_img, 'gm_mask.nii')

