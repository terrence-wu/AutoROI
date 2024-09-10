#!/usr/bin/env python3.7

import os
import sys
import glob
import re
import numpy as np

from matplotlib import pylab as plt
from matplotlib.colors import ListedColormap

from skimage.transform import resize

from matplotlib import pylab as plt
# from matplotlib.colors import ListedColormap

from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.saliency.tissue_detection import get_slide_thumbnail, get_tissue_mask

from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.color_deconvolution import color_deconvolution_routine, stain_unmixing_routine

from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration, perturb_stain_concentration

from histomicstk.saliency.cellularity_detection_superpixels import Cellularity_detector_superpixels

##

cMap=plt.cm.gray

cnorm = {
    'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
    'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
}


W_target = np.array([
    [0.5807549,  0.08314027,  0.08213795],
    [0.71681094,  0.90081588,  0.41999816],
    [0.38588316,  0.42616716, -0.90380025]
])

stain_unmixing_routine_params = {
    'stains': ['hematoxylin', 'eosin'],
    'stain_unmixing_method': 'macenko_pca',
}

###
def islist(var):
    return(isinstance(var, list))

def islist1(var):
    return(isinstance(var, list) and len(var)>0 )

def file_readable(fn1):
    return(isinstance(fn1, str) and os.path.exists(fn1) and os.path.isfile(fn1) and os.access(fn1, os.R_OK))

def dir_exists(dir1):
    return(os.path.exists(dir1) and os.path.isdir(dir1))

def mkdir(dir1, warning=False):
    if dir_exists:
        if warning:
            print("Folder '%s' exists!!" % (dir1))
    else:
        try:
            os.mkdir(dir1)
        except:
            print("Error! Could not create folder '%s'!!" % (dir1))

###

params = sys.argv

n=len(params)

if n==1:
    print('  Usage: TW.norm2_cellularity.py TCGA_filename_prefix')
    sys.exit(0)

fprefix=params[1]
fprefix=os.path.expanduser(fprefix)
fprefix=os.path.abspath(fprefix)

pat=os.path.basename(fprefix)
pat=pat[:23]
# pat='TCGA-2J-AAB1-01Z-00-DX1'

#thumb_dir0='/scratch/SVS/SKCM_thumb10K'
thumb_dir0=os.path.dirname(fprefix)

print("## FileTag = %s" % (pat))
print("## Working Path = %s" % (thumb_dir0))

thumb_pat=os.path.join(thumb_dir0, pat)+"*.svs/"+pat+"*_thumb.png"
thumb_fn=glob.glob(thumb_pat)
if islist1(thumb_fn): thumb_fn=thumb_fn[0]


###
if not file_readable(thumb_fn):
    print("Error!! Tissue RGB thumbnail not found! ")
    sys.exit(1)

tissue_rgb = plt.imread(thumb_fn)
t_shape=tissue_rgb.shape
print(thumb_fn)
print(t_shape)
t_shape2=t_shape[0:2]

tissue_rgb2 = tissue_rgb*255
tissue_rgb2=tissue_rgb2.astype(np.uint8)

##


###
out_fn0=re.sub('_thumb.png', '_', thumb_fn)
out_fn0

##

mask_in, _ = get_tissue_mask(
    tissue_rgb2, deconvolve_first=False,
    n_thresholding_steps=1, sigma=1.5, min_size=30)

mask_in=mask_in>0
plt.imsave(out_fn0+'unnorm_tissue_mask.png',  mask_in.astype(np.uint8), cmap=plt.cm.gray)

###
t_rgb=tissue_rgb2
excl_mask = mask_in==0 

print("Normalization #1 by reinhard")
tissue_reinhard = reinhard(
    t_rgb, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'],
    mask_out=excl_mask)

plt.imsave(out_fn0 + "norm1.png",  tissue_reinhard)

print("Normalization #2 by deconvolution")
tissue_reinhard_deconv = deconvolution_based_normalization(
            tissue_reinhard, W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params,
            mask_out=excl_mask)

print("Normalization done, saving ...")

plt.imsave(out_fn0 + "norm2.png",  tissue_reinhard_deconv)

###

w_rgb=tissue_reinhard_deconv

print("Tissue detection #1")
tissue_labeled1, _ = get_tissue_mask(
    w_rgb, deconvolve_first=False,
    n_thresholding_steps=1, sigma=0., min_size=30)
    
print("High cellularity detection #2")
tissue_labeled2, _ = get_tissue_mask(
    w_rgb, deconvolve_first=False,
    n_thresholding_steps=2, sigma=0., min_size=30)
    
tissue_labeled_all1=(tissue_labeled1>0)+0
print(tissue_labeled_all1.sum())

tissue_labeled_all2=(tissue_labeled2>0)+0
print(tissue_labeled_all2.sum())

print("Detection done, saving ...")
plt.imsave(out_fn0+'cellularity1.png',  tissue_labeled_all1.astype(np.uint8), cmap=plt.cm.gray)
plt.imsave(out_fn0+'cellularity2.png',  tissue_labeled_all2.astype(np.uint8), cmap=plt.cm.gray)





