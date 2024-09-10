#!/usr/bin/env python3

import sys
import argparse
import os
import re
import time
import numpy as np
from   glob import glob
import scipy.ndimage
import openslide
import large_image
import histomicstk as htk
import matplotlib.pyplot as plt
from   sklearn.linear_model import LinearRegression

###

def hide_args(arglist, exclist=[]):
    for action in arglist:
        if not action in exclist:
            action.help=argparse.SUPPRESS

ver0='V3'
steptag='step1'
ver="SVS2patches "+ver0+".0 by Terrence Wu: " + steptag
parser = argparse.ArgumentParser(
                prog='TW.svs2patches.%s_%s.py' % (ver0.lower(), steptag), 
                description="Description: " + ver, 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                add_help=False)

parser.add_argument('--version', action='version', version=ver, help=argparse.SUPPRESS)
parser.add_argument("--log", metavar="log_file", type=argparse.FileType('w'), default=sys.stdout, help=argparse.SUPPRESS)
parser.add_argument("--unlock", dest='unlock', action='store_true', default=False, help=argparse.SUPPRESS)

group = parser.add_mutually_exclusive_group()
group.add_argument('-h', help='help', action='help', default=argparse.SUPPRESS)
group.add_argument('-H', '--help', help='long help', action='help', default=argparse.SUPPRESS)

parser.add_argument("-w", "--workdir", dest='workdir', default='.', metavar='folder_path',
                help="Path to work folder (e.g. ~/AI_code/SKCM_work or SKCM)")
parser.add_argument('mag_oi', type=float, metavar='magnification', help='magnification [5/10/20/40], (e.g. 20 -> 20x)',
                        choices=[5, 10, 20, 40])
parser.add_argument('sz_oi', type=int, metavar='patch_size', help='patch size, (e.g. 512 -> 512x512)')
parser.add_argument('pat', metavar='file_prefix', type=str,  help='SVS file name prefix (e.g. TCGA-XX-1234-01Z-00-DX1)')

h2 = parser.add_argument('-J', "--jpeg", dest='jpeg', metavar='nJpg', type=int, default=2, help="at most save N intermediate jpegs")
h1 = parser.add_argument("-f", "--force", dest='force', action='store_true', default=False, help="force to override previous files")

hidelist=[ h1, h2 ]

new_options={
    'stain_hit_max':    [int, 85, 1, 255,      'upper grayscale of dark region (black=0)'],
    'cell1_minfr0':     [float, 0.9, 0, 1.0,   'min frac of HTK cellularity1 overlap (fr_cell1)'],
    'cell2_minfr0':     [float, 0.1, 0, 1.0,   'min frac of HTK cellularity2 overlap (fr_cell2)'],
    'excl_maxfr':       [float, 0.005, 0, 1.0, 'max frac of HistoQC exclusion overlap (fr_excl)'],
    'hit_area_fr_min':  [float, 0.02, 0, 1.0,  'min frac of dark region (fr_dark)'],
    'hit_area_fr_max':  [float, 0.5, 0, 1.0,   'max frac of dark region (fr_dark)'],
    'hit_signal_min':   [float, 0.5, 0, 1.0,   'min of mean dark region signal (black=1)'],
    'fr_white_max0':    [float, 0.25, 0, 1.0,  'max frac of white region (fr_white)'],
    'image_every':      [int, 1000, 1, 10000,     'show message every N patches']
}
###

new_args={}
for kk in new_options.keys():
    nn=new_options.get(kk)
    if not kk=='' and not nn[0] is None:
        if len(nn)>=5:
            ee='%s (%s - %s)' % (nn[4], nn[2], nn[3])
        else:
            ee='(%s - %s)' % (nn[2], nn[3])
        hh=parser.add_argument("--"+kk, type=nn[0], action='store', default=nn[1], help=ee)
        new_args[kk]=hh
        hidelist.append(hh)

##
args=sys.argv[1:]

if len(args)==0:
    hide_args(hidelist)
    parser.print_help()
    sys.exit(0)

if '--help' in args or '-H' in args:
    parser.print_help()
    sys.exit(0)

hide_args(hidelist)

if '-h' in args:
    parser.print_help()
    sys.exit(0)

options = parser.parse_args(args)

mag_oi=options.mag_oi

sz_oi=options.sz_oi
if sz_oi<16:
    parser.print_usage()
    options.log.write("Patch size can not be smaller than 16: %g\n" % (sz_oi))
    sys.exit(1)

pat=options.pat
if pat=='':
    parser.print_usage()
    options.log.write("File pattern can not be empty: %s\n" % (pat))
    sys.exit(1)

workdir=options.workdir
forcedrun=options.force
unlock=options.unlock
step_jpeg=options.jpeg

for kk in new_args.keys():
    nn=new_options.get(kk)
    tmp=vars(options)[kk]
    if tmp<nn[2] or tmp>nn[3]:
        hh=new_args.get(kk)
        hh.help='(%s - %s)' % (nn[2], nn[3])
        parser.print_help()
        options.log.write(kk.upper() + " value range (%s - %s) error: %g\n" % (nn[2], nn[3], tmp))
        sys.exit(1)
    else:
        globals()[kk]=tmp

#####

def islist(var):
    return(isinstance(var, list))

def islist1(var):
    return(isinstance(var, list) and len(var)>0 )

def file_readable(fn1):
    return(isinstance(fn1, str) and os.path.exists(fn1) and os.path.isfile(fn1) and os.access(fn1, os.R_OK))

def dir_exists(dir1):
    return(os.path.exists(dir1) and os.path.isdir(dir1))

def mkdir(dir1, warning=False):
    if dir_exists(dir1):
        if warning:
            print("Folder '%s' exists!!" % (dir1))
    else:
        try:
            os.mkdir(dir1)
        except:
            print("Error! Could not create folder '%s'!!" % (dir1))

####

def arr2patch(m_arr, ix, iy, sz, adjX, adjY):
    mX0=int(ix*adjX)
    mX1=int((ix+sz-1)*adjX + 1)
    mY0=int(iy*adjY)
    mY1=int((iy+sz-1)*adjY + 1)
    mm_arr=m_arr[mY0:mY1, mX0:mX1]
    return(mm_arr)


def ts2im(ts, ix, iy, source_sz=None, target_mag=40.0, limitX=None, limitY=None, target_sz=None, show=False):
    if show: print(ix,iy)
    
    if source_sz is None:
        if target_sz is None:
            return(None)
        else:
            source_native_mag=ts.getMetadata()['magnification']
            source_sz=int((1.02*source_native_mag/target_mag)*target_sz)
    
    if target_sz is None:
        source_native_mag=ts.getMetadata()['magnification']
        target_sz=int((1.0*target_mag/source_native_mag)*source_sz)
    
    if limitX is None:
        limitX=ts.getMetadata()['sizeX'] - source_sz
    if limitY is None:
        limitY=ts.getMetadata()['sizeY'] - source_sz
        
    if ix>limitX:
        return(None)
    if iy>limitY:
        return(None)
        
    im10, _ = ts.getRegionAtAnotherScale(
        sourceRegion=dict(left=ix, top=iy, width=source_sz, height=source_sz, units='base_pixels'),
        targetScale=dict(magnification=target_mag),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)
    
    im10=im10[0:target_sz, 0:target_sz, 0:3]
    if show: 
        print(im10.shape)
    
    return(im10)

def xy2patch3(ts, iX, iY, sz_def, mag_oi, sz_oi, mm_adjX, mm_adjY, mask_arr, cell1_arr, cell2_arr, excl_arr, cell1_minsum, cell2_minsum, excl_maxsum):
    im10_cell2=arr2patch(cell2_arr, iX, iY, sz_def, mm_adjX, mm_adjY)
    im10_cell2_sum=np.sum(im10_cell2)
    if (im10_cell2_sum < cell2_minsum):
        return(None, None, None, None, None)
    
    im10_cell1=arr2patch(cell1_arr, iX, iY, sz_def, mm_adjX, mm_adjY)
    im10_cell1_sum=np.sum(im10_cell1)
    if (im10_cell1_sum < cell1_minsum):
        return(None, None, None, None, None)
    
    im10_excl=arr2patch(excl_arr, iX, iY, sz_def, mm_adjX, mm_adjY)
    im10_excl_sum=np.sum(im10_excl)
    if (im10_excl_sum > excl_maxsum):
        return(None, None, None, None, None)
    
    im10_mask=arr2patch(mask_arr, iX, iY, sz_def, mm_adjX, mm_adjY)
    im10_mask_sum=np.sum(im10_mask)
    
    im10=ts2im(ts, iX, iY, target_mag=mag_oi, target_sz=sz_oi, show=False)
    return(im10, im10_cell1_sum, im10_cell2_sum, im10_excl_sum, im10_mask_sum)

def im2stain(im0):
    stainColorMap = {
        'hematoxylin': [0.65, 0.70, 0.29],
        'eosin':       [0.07, 0.99, 0.11],
        'dab':         [0.27, 0.57, 0.78],
        'null':        [0.0, 0.0, 0.0]
    }
    
    # specify stains of input image
    stain_1 = 'hematoxylin'   # nuclei stain
    stain_2 = 'eosin'         # cytoplasm stain
    stain_3 = 'null'          # set to null of input contains only two stains
    
    # create stain matrix
    W = np.array([stainColorMap[stain_1],
                  stainColorMap[stain_2],
                  stainColorMap[stain_3]]).T
    
    # perform standard color deconvolution
    im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im0, W).Stains
    
    return(im_stains)

#LR_RGB=[ [inteR, coefR], [inteG, coefG], [inteB, coefB] ]

def im2norm(img, LR_RGB):
    imgw=np.float32(image255(img))
    im10norm=imgw*0.0
    im10norm[:,:,0]=imgw[:,:,0]*LR_RGB[0][1] + LR_RGB[0][0]
    im10norm[:,:,1]=imgw[:,:,1]*LR_RGB[1][1] + LR_RGB[1][0]
    im10norm[:,:,2]=imgw[:,:,2]*LR_RGB[2][1] + LR_RGB[2][0]
    im10norm[im10norm<0]=0
    im10norm[im10norm>255]=255
    im10norm=im10norm.astype(np.uint8)
    return(im10norm)

def stain2adj2(img, adjH=200, adjL=100):
    img_stain=img.copy()
    imH=np.percentile(img, 99.9)
    imL=np.percentile(img,  0.1)
    img_stain[img_stain>imH]=imH
    img_stain[img_stain<imL]=imL
    stain_max=img_stain.max()
    stain_min=img_stain.min()
    if stain_max<adjH:
        stain_max=adjH
    if stain_min>adjL:
        stain_min=adjL
    stain_range=float(stain_max)-float(stain_min)
    stain_adj=255.0*(img_stain-stain_min)/stain_range
    stain_adj=stain_adj
    return(stain_adj)

def image255(img, to=255):
    if img.ndim==3:
        if img.shape[2]>3:
            img=img[:, :, 0:3]
    if to==1 and img.max()>1:
            img=img.astype(np.float32)/255
            img[img>1]=1.0
            img[img<0]=0.0
    elif to==255 and img.max()<=1:
            img=np.float32(img)*255.0
            img[img>255]=255
            img[img<0]=0
    return(img)

def mask_white(imgw, min_rgb=200, max_diff=55, erosion=3):
    assert imgw.ndim==3, 'must be a 3-d image array'
    imgw=np.float32(image255(imgw))
    rgb_min=np.min(imgw, axis=2)
    rgb_max=np.max(imgw, axis=2)
    msk=np.logical_and(rgb_min>=min_rgb, (rgb_max-rgb_min)<=max_diff)
    if erosion>0:
        msk=scipy.ndimage.morphology.binary_erosion(msk, iterations=erosion)
        msk=scipy.ndimage.morphology.binary_dilation(msk, iterations=erosion)
    return(msk)

### main codes begin

# workdir='SKCM'; mag_oi=20; sz_oi=512; pat='TCGA-3N-A9WB-01Z-00-DX1'

if workdir in ('PAAD', 'HNSC', 'SKCM', 'UCEC', 'BRCA', 'COAD', 'READ', 'COADREAD', 
                'LUSC', 'LUAD', 'THCA', 'PRAD', 'KIRC', 'KIRP', 'ESCA', 'STAD', 
                'GBM', 'OV', 'UVM', 'BLCA', 'LIHC'):
    workdir="~/AI_code/%s_work" % workdir

workdir=os.path.abspath(os.path.expanduser(workdir))
print("## Work Folder = %s" % (workdir))

if not dir_exists(workdir):
    print('  Error: Work Folder not found: %s!!' % workdir)
    parser.print_usage()
    sys.exit(1)

###

patch_dir=os.path.join(workdir, "patches_mag%g_sz%g_%s" % (mag_oi, sz_oi, ver0))
mkdir(patch_dir)

save_dir=os.path.join(patch_dir, "summary")
mkdir(save_dir)
ofn0=os.path.join(save_dir, "save.%s.mag%g.sz%g." % (pat, mag_oi, sz_oi))
#all_npy_fns=glob(os.path.join(save_dir, "*.npy"))
all_npy_fns=glob(ofn0 + "*.npy")

ofn1=os.path.join(save_dir, "save.%s.mag%g.sz%g.step1%s" % (pat, mag_oi, sz_oi, ver0))
ofn1k=ofn1+'.LOCK.npy'
if file_readable(ofn1k):
    if unlock:
        print('Found lock file: %s' % os.path.basename(ofn1k))
        try:
            os.remove(ofn1k)
        except:
            print("Unable to unlock flie: %s" % ofn1k)
    else:
        print("Locked: %s" % ofn1k)
        sys.exit(1)

pat1=ofn1+".n[0-9]*.npy"
r1 = re.compile(pat1)
npylist1 = list(filter(r1.match, all_npy_fns))

if len(npylist1)>0:
    if ofn1k in npylist1:
        print('Found lock file: %s' % os.path.basename(ofn1k))
        sys.exit(1)
    ofn1a=npylist1[-1]
    print('Found previous step1 file: %s' % os.path.basename(ofn1a))
    if forcedrun:
        ofn1b=ofn1a+".bak"
        os.rename(ofn1a, ofn1b)
        print("Rename file: %s" % os.path.basename(ofn1b) )
    else:
        sys.exit(0)
else:
    forcedrun=False

###

svs_dir=os.path.join(workdir, "SVS")
mask_dir=os.path.join(workdir, "mask")
thumb_dir=os.path.join(workdir, "thumb")
pen_dir=os.path.join(workdir, "pen")
spur_dir=os.path.join(workdir, "spur")

if not dir_exists(svs_dir):
    print("## Working Folder should have a SVS subfolder: %s" % svs_dir)
    print('   Error: SVS Folder not found!!')
    sys.exit(1)

if not dir_exists(thumb_dir):
    thumb_dir=os.path.join(workdir, "thumbnail")

if not dir_exists(thumb_dir):
    print("## Working Folder should have a thumbnail subfolder: %s" % thumb_dir)
    print('   Error: thumbnail Folder not found!!')
    sys.exit(1)

if not dir_exists(mask_dir):
    print("## Working Folder should have a mask subfolder: %s" % mask_dir)
    print('   Error: mask Folder not found!!')
    sys.exit(1)

if not dir_exists(spur_dir):
    spur_dir=mask_dir

use_pen=False
if dir_exists(pen_dir):
    use_pen=True

all_svs_fns=glob(os.path.join(svs_dir, "*.svs"))

pat1=os.path.join(svs_dir, pat) + ".*"
r = re.compile(pat1)
newlist = list(filter(r.match, all_svs_fns))

if len(newlist)==0:
    print("Error! No SVS file matched to pattern: '%s'" % (pat))
    sys.exit(1)
elif len(newlist)>1:
    print("Error! Multiple SVS files matched to pattern: '%s'" % (pat))
    print(np.array(newlist))
    sys.exit(1)
else:
    svs_fn=newlist[0]

print("SVS file: %s" % svs_fn)

svs_name=re.sub(svs_dir, '', svs_fn)
svs_name=re.sub('^/*', '', svs_name)
svs_tag=re.sub('.svs$', '', svs_name)

##

LR_dir=os.path.join(workdir, "LR_model")
LRfn=os.path.join(LR_dir, "%s.LR_coeff.npy" % (svs_tag) )

if file_readable(LRfn):
    use_LR=True
else:
    use_LR=False

##

thumb_fn=os.path.join(thumb_dir, svs_name)+'_thumb.png'
norm2_fn=os.path.join(thumb_dir, svs_name)+'_norm2.png'
cell1_fn=os.path.join(thumb_dir, svs_name)+'_cellularity1.png'
cell2_fn=os.path.join(thumb_dir, svs_name)+'_cellularity2.png'

if not file_readable(cell2_fn):
    thumb_fn=os.path.join(thumb_dir, svs_name, svs_name)+'_thumb.png'
    norm2_fn=os.path.join(thumb_dir, svs_name, svs_name)+'_norm2.png'
    cell1_fn=os.path.join(thumb_dir, svs_name, svs_name)+'_cellularity1.png'
    cell2_fn=os.path.join(thumb_dir, svs_name, svs_name)+'_cellularity2.png'

if not use_LR:
    if not file_readable(thumb_fn):
        print("Error! No thumbnail file: '%s'" % (thumb_fn))
        sys.exit(1)
    else:
        print("Thumbnail file: '%s'" % (thumb_fn))
    
    if not file_readable(norm2_fn):
        print("Error! No normalized thumbnail file: '%s'" % (norm2_fn))
        sys.exit(1)

##

if not file_readable(cell1_fn):
    print("Error! No level1 cellularity file: '%s'" % (cell1_fn))
    sys.exit(1)

if not file_readable(cell2_fn):
    print("Error! No level2 cellularity file: '%s'" % (cell2_fn))
    sys.exit(1)

##

mask_fn=os.path.join(mask_dir, svs_name)+'_mask_use.png'
if not file_readable(mask_fn):
    mask_fn=os.path.join(mask_dir, svs_name, svs_name)+'_mask_use.png'
    if not file_readable(mask_fn):
        print("Warning! No mask file: '%s'" % (mask_fn))
        mask_fn=cell1_fn
        #sys.exit(1)
    else:
        print("Mask file: '%s'" % (mask_fn))

##

use_spur=True
spur_fn=os.path.join(spur_dir, svs_name)+'_spur.png'
if not file_readable(spur_fn):
    spur_fn=os.path.join(spur_dir, svs_name, svs_name)+'_spur.png'
    if not file_readable(spur_fn):
        use_spur=False

if use_spur:
    print("Spur file: '%s'" % (spur_fn))

##

if use_pen:
    pen_fn=os.path.join(pen_dir, svs_name)+'_pen_markings.png'
    if not file_readable(pen_fn):
        pen_fn=os.path.join(pen_dir, svs_name, svs_name)+'_pen_markings.png'
        if not file_readable(pen_fn):
            use_pen=False

if use_pen:
    print("Pen Marking file: '%s'" % (pen_fn))

##

os.chdir(workdir)

print("## Job Magnification = %g, Size = %g, FileTag = %s" % (mag_oi, sz_oi, pat))

starttime=time.ctime()
print("## Job '%s' started: %s" % (pat, starttime) )

##

cell1_arr=np.array(plt.imread(cell1_fn))
if len(cell1_arr.shape)>2:
    cell1_arr=cell1_arr[:,:,0]

cell1_arr=cell1_arr>0

cell2_arr=np.array(plt.imread(cell2_fn))
if len(cell2_arr.shape)>2:
    cell2_arr=cell2_arr[:,:,0]

cell2_arr=cell2_arr>0

if cell2_arr.sum() < 100:
    cell2_arr=cell1_arr.copy()

mmS=cell2_arr.shape[0:2]
mmY=mmS[0]
mmX=mmS[1]
print('Thumbnail image size: (%s, %s)' % mmS)

##

mask_arr=np.array(plt.imread(mask_fn))
if len(mask_arr.shape)>2:
    mask_arr=mask_arr[:,:,0]

## mask_arr represents the pixels to be used, need to minimize the number of hits during imresize
if not mask_arr.shape == mmS:
    temp_arr=imresize(np.logical_not(mask_arr), mmS)
    mask_arr = np.logical_not(temp_arr>0)
else:
    mask_arr=mask_arr>0

if cell1_arr.sum() < 100:
    cell1_arr=mask_arr.copy()

##

if mask_arr.sum()==0:
    mask_arr=cell1_arr

if mask_arr.sum()==0:
    np.save(ofn1+'.n0.npy', np.array([]))
    print('Mask file is empty, leaving!!')
    sys.exit(0)

cell1_arr=np.logical_and(mask_arr, cell1_arr)
cell2_arr=np.logical_and(mask_arr, cell2_arr)
fr_cell1=1.0*cell1_arr.sum()/mask_arr.sum()

###

if file_readable(ofn1k):
    print("Locked: %s" % ofn1k)
    sys.exit(1)

lock=True
if lock:
    np.save(ofn1k, [])

###

excl_arr=mask_arr>300  ## force to all False

## spur_arr or pen_arr represents the pixels to be excluded, need to maximize the number of hits during imresize
if use_spur:
    spur_arr=np.array(plt.imread(spur_fn))
    if len(spur_arr.shape)>2:
        spur_arr=spur_arr[:,:,0]
    if spur_arr.max()==0:
        use_spur=False

if use_spur:
    if not spur_arr.shape == mmS:
        temp_arr=imresize(spur_arr, mmS)
        spur_arr = temp_arr>0
    else:
        spur_arr=spur_arr>0
    excl_arr=np.logical_or(excl_arr, spur_arr)

if use_pen:
    pen_arr=np.array(plt.imread(pen_fn))
    if len(pen_arr.shape)>2:
        pen_arr=pen_arr[:,:,0]
    if pen_arr.max()==0:
        use_pen=False

if use_pen:
    if not pen_arr.shape == mmS:
        temp_arr=imresize(pen_arr, mmS)
        pen_arr = temp_arr>0
    else:
        pen_arr=pen_arr>0
    excl_arr=np.logical_or(excl_arr, pen_arr)
    
####
LRfn=os.path.join(LR_dir, "%s.LR_coeff.npy" % (svs_tag) )

if use_LR:
    print("Load LinearRegression Coefficients: %s" % (LRfn))
    LR_RGB=np.load(LRfn)
else:
    mkdir(LR_dir)
    print("Calculate LinearRegression Coefficients")
    newdim=mmX*mmY
    
    thumb_arr=np.array(plt.imread(thumb_fn))[:,:,0:3]
    
    thumb_arr3=np.float32(image255(thumb_arr)).reshape((newdim, 3))
    
    norm2_arr=np.array(plt.imread(norm2_fn))[:,:,0:3]
    norm2_arr3=np.float32(image255(norm2_arr)).reshape((newdim, 3))
    
    cell1_arr3=cell1_arr.reshape(newdim)
    cell1_arr4=cell1_arr3>0
    
    thumb_arr4=thumb_arr3[cell1_arr4, :]
    norm2_arr4=norm2_arr3[cell1_arr4, :]
    
    regressorR = LinearRegression()  
    regressorR.fit(thumb_arr4[:,0:1], norm2_arr4[:,0:1])
    regressorG = LinearRegression()  
    regressorG.fit(thumb_arr4[:,1:2], norm2_arr4[:,1:2])
    regressorB = LinearRegression()  
    regressorB.fit(thumb_arr4[:,2:3], norm2_arr4[:,2:3])
    
    inteR=regressorR.intercept_[0]
    coefR=regressorR.coef_[0][0]
    inteG=regressorG.intercept_[0]
    coefG=regressorG.coef_[0][0]
    inteB=regressorB.intercept_[0]
    coefB=regressorB.coef_[0][0]
    
    LR_RGB=[ [inteR, coefR], [inteG, coefG], [inteB, coefB] ]
    LR_RGB=np.array(LR_RGB)
    
    np.save(LRfn, LR_RGB)

print(LR_RGB)

###

ts = large_image.getTileSource(svs_fn)
print(ts.getNativeMagnification())

ts_meta=ts.getMetadata()

#print(ts_meta)
ts_sizeX=ts_meta['sizeX']
ts_sizeY=ts_meta['sizeY']

ts_native_mag=ts_meta['magnification']
print('ts_native_mag = %s' % ts_native_mag)
print('[ts_sizeY, ts_sizeX] = [%s, %s]' % (ts_sizeY, ts_sizeX) )

##

mm_adjX=mmX / ts_sizeX
mm_adjY=mmY / ts_sizeY
print('[mm_adjY, mm_adjX] = [%s, %s]' % (mm_adjY, mm_adjX))

####

mag_def=ts_meta['magnification']
adj_def=mag_oi/mag_def
sz_def=sz_oi*mag_def/mag_oi
print('[adj_def, sz_def] = [%s, %s]' % (adj_def, sz_def))

###

cell1_minfr=cell1_minfr0*fr_cell1
cell2_minfr=cell2_minfr0*fr_cell1

im10_temp=arr2patch(excl_arr, 0, 0, sz_def, mm_adjX, mm_adjY)
im10_pixels=np.prod(im10_temp.shape[0:2])
f_im10_pixels=1.0/im10_pixels

cell1_minsum=np.int(cell1_minfr*im10_pixels+1)
cell2_minsum=np.int(cell2_minfr*im10_pixels+1)
excl_maxsum=np.int(excl_maxfr*im10_pixels+1)

###
nJpeg=0
if step_jpeg>0:
    patch_dir1=os.path.join(patch_dir, pat)
    jpeg_dir=os.path.join(patch_dir1, 'step1_jpeg')
    mkdir(patch_dir1)
    mkdir(jpeg_dir)
    if forcedrun:
        jpgfns=glob(os.path.join(jpeg_dir, '*.jpg'))
        if len(jpgfns)>0:
            print("Delete %g previous jpg files" % len(jpgfns))
            _=[os.remove(jpgfn) for jpgfn in jpgfns]

##

img2perc=1.0/255.0

if True:
    sz_half=sz_def/2
    max_iX=ts_sizeX-sz_def
    max_iY=ts_sizeY-sz_def
    
    all_iX=np.arange(start = 0, stop = max_iX, step=sz_half).astype(np.int)
    all_iY=np.arange(start = 0, stop = max_iY, step=sz_half).astype(np.int)
    
    all_step1_summary=[]
    step1_summary=list()
    iImg=0
    
    for iX in all_iX:
        print("iX =%6g of %6g/%6g now %5g patches for TAG:\t%s" % (iX, ts_sizeX, ts_sizeY, iImg, pat))
        for iY in all_iY:
            im10, cell1_sum, cell2_sum, excl_sum, mask_sum =xy2patch3(ts, iX, iY, sz_def, mag_oi, sz_oi, mm_adjX, mm_adjY, mask_arr, cell1_arr, cell2_arr, excl_arr, cell1_minsum, cell2_minsum, excl_maxsum)
            
            if im10 is None:
                continue
            
            im10norm=im2norm(im10, LR_RGB)
            
            msk_white=mask_white(im10norm, erosion=0)
            fr_white=msk_white.mean()
            if fr_white<=fr_white_max0:
                im10N_stain=im2stain(im10norm)[:, :, 0]
                im10N_stain_adj=stain2adj2(im10N_stain)
                im10N_signal=1.0- img2perc*im10N_stain
                im10N_signal_hit=im10N_stain_adj <= stain_hit_max
                if im10N_signal_hit.max()>0:
                    im10N_hit_area_fr=im10N_signal_hit.mean()
                    im10N_hit_area_signal=im10N_signal[im10N_signal_hit].mean()
                    if im10N_hit_area_fr>=hit_area_fr_min and im10N_hit_area_fr<=hit_area_fr_max and im10N_hit_area_signal>=hit_signal_min:
                            hit_detail=[iImg, iX, iY, f_im10_pixels*cell1_sum, f_im10_pixels*cell2_sum, f_im10_pixels*excl_sum, 
                                        im10N_hit_area_fr, im10N_hit_area_signal, f_im10_pixels*mask_sum, fr_white]
                            step1_summary.append(hit_detail)
                            if iImg%image_every==0:
                                print('hit = [ %s, %s, %s, mask=%s, %s, %s, excl=%s, hit_fr=%s, signal=%s, white=%s ]:\t%s' % 
                                    (iImg, iX, iY, round(hit_detail[8],3), round(hit_detail[3],3), round(hit_detail[4],3), 
                                        round(hit_detail[5],3), round(hit_detail[6],3), round(hit_detail[7],3), 
                                        round(hit_detail[9],3), pat ) )
                                if nJpeg<step_jpeg:
                                    step1_fn="step1_mag%g_sz%g.%s.origmag%g.X%06i.Y%06i.ind%05g.jpg" % (mag_oi, sz_oi, pat, mag_def, iX, iY, iImg )
                                    step1_fn2=os.path.join(jpeg_dir, step1_fn)
                                    plt.imsave(step1_fn2, im10norm)
                                    nJpeg+=1
                            iImg=iImg+1
                            if iImg%500==0:
                                all_step1_summary.append(np.array(step1_summary))
                                step1_summary=list()
    
    if len(step1_summary)>0:
        all_step1_summary.append(np.array(step1_summary))
    if len(all_step1_summary)>0:
        step1_summary=np.concatenate(all_step1_summary)
    ofn1a=ofn1+".n%s.npy"%len(step1_summary)
    np.save(ofn1a, step1_summary)

if lock:
    try:
        os.remove(ofn1k)
    except:
        print("Unable to delete file: %s" % ofn1k)

print("Number of Step1 patches = %s" % (len(step1_summary)))

print("## Job '%s' Step1 started: %s" % (pat, starttime) )
print("## Job '%s' Step1 completed: %s" % (pat, time.ctime()) )

# hit_detail=[iImg, iX, iY, f_im10_pixels*cell1_sum, f_im10_pixels*cell2_sum, f_im10_pixels*excl_sum, 
#                                        im10N_hit_area_fr, im10N_hit_area_signal, f_im10_pixels*mask_sum, fr_white ]
    
#####

