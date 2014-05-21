"""
Make a nice plot of the hierarchical atlas
"""
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import nibabel

from nipy.labs import viz

atlas_file = 'labels_level_3.nii.gz'

atlas_img = nibabel.load(atlas_file)
atlas = atlas_img.get_data().astype(np.float)
affine = atlas_img.get_affine()

###############################################################################
# The different cut locus
cuts = [('z', -12), ('z', -2), ('z', 8), ('z', 18), ('z', 28),
        ('z', 38), ('z', 48),
        ('y', -84), ('y', -70), ('y', -50), ('y', -38), ('y', -24), ('y', 2),
        ('y', 28),
        ('x', -2), ('x', -16), ('x', -40), ('x', -48), ('x', -54),
        ]

# The linewidths of each level
linewidths = {1: 2, 2:1, 3:.2}


###############################################################################
# 8 visualy differentiable colors gradients
colors = {'Yl': ((1, 1, .3), (1, 1, 0), (.91, .91, 0)),
          'Or': ((1, .69, .1), (1, .65, 0), (.91, .55, .1)),
          'Rd': ((1, .34, .35), (1, 0, .03), (.86, 0, .03)),
          'Ma': ((1, .47, 1), (1, 0, 1), (.88, 0, .89)),
          'Pu': ((.43, 0, .78), (.37, 0, .56)),
          'Bl': ((0, .03, 1), (0, 0, .5)),
          'Cy': ((.38, 1, 1), (0, 1, 1), (0, .76, 1)),
          'Gr': ((0, 1, 0), (0, .5, 0))
         }
cmaps = dict((name, LinearSegmentedColormap.from_list(name, c))
             for name, c in colors.items())

cmaps_list = (cmaps['Yl'], cmaps['Gr'], cmaps['Rd'], cmaps['Ma'],
              cmaps['Or'], cmaps['Bl'], cmaps['Cy'], cmaps['Pu'])

###############################################################################

atlas_name = atlas_file.replace('.nii.gz', '')

if os.path.exists(atlas_name):
    shutil.rmtree(atlas_name, ignore_errors=True)

os.mkdir(atlas_name)

fnames = list()

for axis, cut in cuts:
    slicer = viz.plot_anat(slicer=axis, cut_coords=(cut, ),)
    for level in (1, 2, 3):
        level_labels = np.trunc(atlas / 1000. * 10 ** (level - 1))
        level_labels = level_labels.astype(np.int)
        for label in np.unique(level_labels):
            if label == 0:
                continue
            slicer.contour_map(level_labels == label, affine,
                            levels=[.5], colors='k',
                            linewidths=linewidths[level])
    for label in range(1, 9):
        this_altas = atlas.copy()
        this_altas[atlas >= 1000 * (label + 1)] = 0
        slicer.plot_map(atlas, affine,
                        cmap=cmaps_list[label - 1],
                        threshold=1000 * label - 2,
                        vmin=1000 * label)
    fname = '%s/%s_%i.png' %  (atlas_name, axis, cut)
    plt.savefig(fname, dpi=300)
    fnames.append(fname)
    plt.close('all')


os.system('montage %s -geometry 2000 -tile 7x3 %s/montage.jpg'
          % (' '.join(fnames), atlas_name))

