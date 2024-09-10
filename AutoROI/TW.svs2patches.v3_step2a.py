#!/usr/bin/env python3

import sys
import argparse
import os
import re
import time
import numpy as np
from   glob import glob
import openslide
import matplotlib.pyplot as plt
import scipy.ndimage
import large_image
import cv2
import skimage
import skimage.morphology

###

def hide_args(arglist, exclist=[]):
    for action in arglist:
        if not action in exclist:
            action.help=argparse.SUPPRESS

ver0='V3'
steptag='step2a'
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

h3 = parser.add_argument("--reduce_overlap", dest='overlap', action='store_true', default=False, help="reduce overlapping of patches")
h2 = parser.add_argument('-J', "--jpeg", dest='jpeg', metavar='nJpg', type=int, default=0, help="at most save N intermediate jpegs")
h1 = parser.add_argument("-f", "--force", dest='force', action='store_true', default=False, help="force to override previous files")

hidelist=[ h1, h2, h3 ]

new_options={
    'stain_hit_max': [int, 85, 1, 255,     'upper grayscale of dark region (black=0)'],
    'n1':            [int, 3000, 1, 65535, 'select N1 patches with smaller fr_excl'],
    'n2':            [int, 2500, 1, 65535, 'select N2 patches with larger fr_cell2'],
    'n3':            [int, 2000, 1, 65535, 'select N3 patches with larger fr_dark'],
    'mask_cut':      [float, 0.9, 0, 1.0,  'min frac of HistoQC mask overlap (fr_mask)'],
    'image_every':   [int, 50, 1, 1000,    'show message every N patches']
}

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
reduce_overlap=options.overlap
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

def select_large(val, N, pre=None):
    if N==0:
        return(np.inf)
    elif not pre is None:
        if ((np.array(val)>=pre).sum()>=N):
            return(pre)
    ind=np.isnan(val)
    if ind.sum()>0:
        val=np.array(val)[np.logical_not(ind)]
    if len(val)==0:
        thr=None
    else:
        mm=np.min(val)
        if N>=len(val):
            thr=mm
        elif N>(val>mm).sum():
            thr=mm
        else:
            thr=np.sort(val)[::-1][N-1]
    return(thr)

def select_small(val, N, pre=None):
    if N==0:
        return(-np.inf)
    elif not pre is None:
        if ((np.array(val)<=pre).sum()>=N):
            return(pre)
    ind=np.isnan(val)
    if ind.sum()>0:
        val=np.array(val)[np.logical_not(ind)]
    if len(val)==0:
        thr=None
    else:
        mm=np.max(val)
        if N>=len(val):
            thr=mm
        elif N>(val<mm).sum():
            thr=mm
        else:
            thr=np.sort(val)[N-1]
    return(thr)

def mark_overlap(xy, sz_diff, threshold=2):
    xy=np.array(xy)
    assert xy.ndim==2 and xy.shape[1]>=2, 'Must be a 2d array with at least 2 columns'
    nImg=xy.shape[0]
    if nImg<=1:
        return(np.zeros(nImg).astype(np.int64))
    marked=np.zeros(nImg).astype(np.int64)
    all_iX=xy[:, 0]
    all_iY=xy[:, 1]
    for i in np.arange(nImg-1):
        iX=all_iX[i]
        iY=all_iY[i]
        #print('\n[iX, iY, marked_i]', [iX, iY, marked[i] ])
        if marked[i]<threshold:
            other_X=all_iX[(i+1):]
            other_Y=all_iY[(i+1):]
            chk_X=np.abs(other_X-iX)
            chk_Y=np.abs(other_Y-iY)
            tmp_X=np.zeros(chk_X.shape[0]).astype(np.int32)
            tmp_Y=np.zeros(chk_Y.shape[0]).astype(np.int32)
            tmp_X[chk_X<=sz_diff]=1
            tmp_Y[chk_Y<=sz_diff]=1
            tmp_X[chk_X==0]=3
            tmp_Y[chk_Y==0]=3
            tmp_X[chk_Y>sz_diff]=0
            tmp_Y[chk_X>sz_diff]=0
            tmp_XY=tmp_X+tmp_Y
            marked[(i+1):]=marked[(i+1):]+tmp_XY
            #print(np.c_[other_X, other_Y]); print('tmp_XY', tmp_XY); print('marked', marked)
    return(marked)

####

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

###

def saveimg(ofn, img):
    if isinstance(ofn, np.ndarray) and isinstance(img, str):
        temp=img
        img=ofn
        ofn=temp
    if len(img.shape)==3:
        plt.imsave(ofn, img)
    elif img.max()>1:
        cv2.imwrite(ofn, img)
    else:
        cv2.imwrite(ofn, (255*img).astype(np.uint8))

def image255(img, to=255):
    if img.ndim==3:
        if img.shape[2]>3:
            img=img[:, :, 0:3]
    if to==1 and img.max()>1:
            img=img.astype(np.float32)/255
            img[img>1]=1.0
            img[img<0]=0.0
    elif to==255 and img.max()<=1:
            img=img*255
            img[img>255]=255
            img[img<0]=0
    return(img)

def img2signal(img1):
    if img1.ndim==2:
        img1=image255(img1)
        imgS=1.0- img1/255.0
        return(imgS)

def add_rim(msk, dh=1, dw=1, value=0):
    if msk is None:
        msk2=np.zeros((2*dh, 2*dw))+value
    else:
        assert msk.ndim==2, "Must be a 2-d image"
        msk2=skimage.util.pad(msk, ((dh, dh),(dw,dw)), 'constant', constant_values=value)
    return(msk2)

def rm_rim(msk, dh=1, dw=1):
    if msk is None:
        msk2=None
    else:
        assert msk.ndim==2, "Must be a 2-d image"
        hh, ww=msk.shape
        if hh<=2*dh or ww<=2*dw:
            msk2=None
        else:
            msk2=msk[dh:(hh-dh), dw:(ww-dw)]
    return(msk2)

def mask_blur(img, use_strict=None, cut_morphology=None, fft_low=None, fft_high=None, bubble=True, bubble_thr=200, returntag=False):
    if isinstance(img, str):
        ifn=img
        ifn=re.sub('.jpg$', '', ifn, flags=re.IGNORECASE)
        ifn=re.sub('.jpeg$', '', ifn, flags=re.IGNORECASE)
        ifn=re.sub('.png$', '', ifn, flags=re.IGNORECASE)
        img=plt.imread(img)
    else:
        ifn=None
    
    if img.ndim==2:
        img_H=image255(img)
    else:
        img=img[:, :, 0:3]
        img=image255(img)
        #
        #img_stain=im2stain(img)
        #img_S=img_stain[:, :, 0]
        img_H=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mmX, mmY = img_H.shape[0:2]
    
    rows, cols = img_H.shape
    crow, ccol = np.int(rows/2), np.int(cols/2)
    
    ff = np.fft.fft2(img_H)
    fshift = np.fft.fftshift(ff)
    
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = np.fft.ifftshift(fshift)
    
    img_fft = np.fft.ifft2(f_ishift)
    img_fft = 20*np.log(np.abs(img_fft))
    
    cm0, f1, f2 = cut_morphology, fft_low, fft_high
    tag=""
    if use_strict is None:
        use_strict=None
    elif isinstance(use_strict, bool):
        if use_strict:
            fft_low=100
            fft_high=200
            cut_morphology=190
            tag='strict'
        else:
            fft_low=80
            fft_high=160
            cut_morphology=150
            tag=""
    elif cut_morphology is None and (isinstance(use_strict, int) or isinstance(use_strict, float)):
        cut_morphology = use_strict
        tag="t%s" % (cut_morphology)
    elif cut_morphology is None and len(use_strict)==1:
        cut_morphology = use_strict[0]
        tag="t%s" % (cut_morphology)
    elif len(use_strict)==2:
        fft_low, fft_high = use_strict[0:2]
        tag="fft%s_%s" % (fft_low, fft_high)
    elif cut_morphology is None and len(use_strict)>=3:
        cut_morphology, fft_low, fft_high = use_strict[0:3]
        tag="t%s_fft%s_%s" % (cut_morphology, fft_low, fft_high)
    else:
        use_strict=None
    
    if cut_morphology is None:
        cut_morphology=150
    if fft_low is None:
        fft_low=80
    if fft_high is None:
        fft_high=160
    if not cm0 is None:
         cut_morphology=cm0
         if tag=="":
            tag="t%s" % (cut_morphology)
         else:
            tag="%s_t%s" % (tag, cut_morphology)
    if not f1 is None and not f2 is None:
         fft_low=f1
         fft_high=f2
         if tag=="":
            tag="fft%s_%s" % (fft_low, fft_high)
         else:
            tag="%s_fft%s_%s" % (tag, fft_low, fft_high)
    if not f1 is None and f2 is None:
         fft_low=f1
         if fft_high<f1+50 or use_strict is None:
            fft_high=f1+50
         if tag=="":
            tag="fft%s_%s" % (fft_low, fft_high)
         else:
            tag="%s_fft%s_%s" % (tag, fft_low, fft_high)
    if not f2 is None and f1 is None:
         fft_high=f2
         if fft_low>f2-50 or use_strict is None:
            fft_low=f2-50
         if tag=="":
            tag="fft%s_%s" % (fft_low, fft_high)
         else:
            tag="%s_fft%s_%s" % (tag, fft_low, fft_high)
    
    img_gray=img_fft
    
    msk = cv2.convertScaleAbs(255-(255*img_gray/np.max(img_gray)))
    msk[msk < fft_low] = 0
    msk[msk > fft_high] = 255
    
    width=50
    dh, dw = map(lambda i: i//width, msk.shape)
    
    msk2=add_rim(msk, dh, dw, 255)
    
    msk3 = msk2.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk3 = cv2.erode(msk3, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    msk3 = cv2.morphologyEx(msk3, cv2.MORPH_CLOSE, kernel)
    
    msk4=msk3.copy()
    msk4[msk4 < cut_morphology] = 0
    msk4[msk4 >= cut_morphology] = 255
    
    msk5=rm_rim(msk4, dh, dw)
    
    if bubble:
        ww0=img_H>=bubble_thr ## default 200
        ww1=skimage.morphology.remove_small_holes(ww0, 625).astype(np.bool)
        ww1=scipy.ndimage.morphology.binary_erosion(ww1, structure=np.ones((4,4))).astype(np.bool)
        ww2=scipy.ndimage.morphology.binary_dilation(ww1, structure=np.ones((6,6))).astype(np.bool)
        ww2=skimage.morphology.remove_small_holes(ww2, 625).astype(np.bool)
        ww2=scipy.ndimage.morphology.binary_erosion(ww2, structure=np.ones((10,10))).astype(np.bool)
        ww2=scipy.ndimage.morphology.binary_dilation(ww2, structure=np.ones((4,4))).astype(np.bool)
        msk5[ww2]=0
    
    iddA=10
    msk6=msk5.copy()
    msk6=scipy.ndimage.morphology.binary_dilation(msk6, structure=np.ones((iddA, iddA))).astype(np.bool)
    
    msk6=skimage.morphology.remove_small_holes(msk6, 400).astype(np.bool)
    
    msk6=add_rim(msk6, iddA+3, iddA+3, 1)
    msk6=scipy.ndimage.morphology.binary_erosion(msk6, structure=np.ones((iddA+3, iddA+3)))
    msk6=rm_rim(msk6, iddA+3, iddA+3)
    
    msk6=skimage.morphology.remove_small_objects(msk6, 200, connectivity=iddA*3)
    msk6=scipy.ndimage.morphology.binary_dilation(msk6, iterations=4).astype(np.bool)
    
    msk9=np.logical_and(msk6, img_H>25)
    mask_fr=msk9.sum() / (mmX*mmY)
    #mask_fr
    
    if not ifn is None:
        if tag=="":
            ofn=ifn+'__fr%s_maskblur.jpg' % (round(mask_fr, 3))
        else:
            ofn=ifn+'__fr%s_maskblur_%s.jpg' % (round(mask_fr, 3), tag)
        saveimg(ofn, msk9)
        print(ofn, mask_fr, tag)
    
    if returntag:
        return(msk9, mask_fr, tag)
    else:
        return(msk9, mask_fr)

### main codes begin

# workdir='SKCM'; mag_oi=20; sz_oi=512; pat='TCGA-3N-A9WB-01Z-00-DX1'

if workdir in ('PAAD', 'HNSC', 'SKCM', 'UCEC', 'BRCA', 'COAD', 'READ', 'COADREAD', 
                'LUSC', 'LUAD', 'THCA', 'PRAD', 'KIRC', 'KIRP', 'ESCA', 'STAD', 
                'GBM', 'OV', 'UVM', 'BLCA', 'LIHC'):
    workdir="~/AI_code/%s_work" % workdir

workdir=os.path.abspath(os.path.expanduser(workdir))
print("## Working Folder = %s" % (workdir))

if not dir_exists(workdir):
    print('  Error: Working Folder not found!!')
    print('  Usage: TW.svs2patches_step1.py mag_oi size_oi filename_tag')
    sys.exit(1)

###

patch_dir=os.path.join(workdir, "patches_mag%g_sz%g_%s" % (mag_oi, sz_oi, ver0))
patch_dir

save_dir=os.path.join(patch_dir, "summary")
ofn0=os.path.join(save_dir, "save.%s.mag%g.sz%g." % (pat, mag_oi, sz_oi))
#all_npy_fns=glob(os.path.join(save_dir, "*.npy"))
all_npy_fns=glob(ofn0 + "*.npy")

ofn1=os.path.join(save_dir, "save.%s.mag%g.sz%g.step1%s" % (pat, mag_oi, sz_oi, ver0))
ofn1k=ofn1+'.LOCK.npy'
if file_readable(ofn1k):
    print('Step1 file locked!!')
    sys.exit(1)

pat1=ofn1+".n[0-9]*.npy"
r1 = re.compile(pat1)
npylist1 = list(filter(r1.match, all_npy_fns))

if len(npylist1)>0:
    ofn1a=npylist1[-1]
    print('Load previous step1 file: %s' % os.path.basename(ofn1a))
    step1_summary=np.load(ofn1a)
else:
    print('Cannot load step1 from previous npy files: %s' % os.path.basename(ofn1))
    sys.exit(1)

ofn2a=os.path.join(save_dir, "save.%s.mag%g.sz%g.step2a%s" % (pat, mag_oi, sz_oi, ver0))
ofn2ak=ofn2a+'.LOCK.npy'
if file_readable(ofn2ak):
    if unlock:
        print('Found lock file: %s' % os.path.basename(ofn2ak))
        try:
            os.remove(ofn2ak)
        except:
            print("Unable to unlock flie: %s" % ofn2ak)
    else:
        print("Locked: %s" % ofn2ak)
        sys.exit(1)

pat2a=ofn2a+".n[0-9]*.npy"
r2a = re.compile(pat2a)
npylist2a = list(filter(r2a.match, all_npy_fns))

if len(npylist2a)>0:
    if ofn2ak in npylist2a:
        print('Found lock file: %s' % os.path.basename(ofn2ak))
        sys.exit(1)
    ofn2a2=npylist2a[-1]
    print('Found previous step2a file: %s' % os.path.basename(ofn2a2))
    if forcedrun:
        ofn2a3=ofn2a2+".bak"
        os.rename(ofn2a2, ofn2a3)
        print("Rename file: %s" % os.path.basename(ofn2a3) )
    else:
        sys.exit(0)
else:
    forcedrun=False

###

svs_dir=os.path.join(workdir, "SVS")

if not dir_exists(svs_dir):
    print("## Working Folder should have a subfolder: SVS")
    print('   Error: SVS Folder not found!!')
    sys.exit(1)

all_svs_fns=glob(os.path.join(svs_dir, "*.svs"))

pat1=os.path.join(svs_dir, pat) + ".*"

r = re.compile(svs_dir+"/"+pat+".*")
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

os.chdir(workdir)

print("## Job Magnification = %g, Size = %g, FileTag = %s" % (mag_oi, sz_oi, pat))

###

if file_readable(ofn2ak):
    print("Locked: %s" % ofn2ak)
    sys.exit(1)

starttime=time.ctime()
print("## Job '%s' started: %s" % (pat, starttime) )

lock=True
if lock:
    np.save(ofn2ak, [])

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

####

mag_def=ts_meta['magnification']
adj_def=mag_oi/mag_def
sz_def=sz_oi*mag_def/mag_oi
print('[adj_def, sz_def] = [%s, %s]' % (adj_def, sz_def))

sz_half=0.5*sz_def

###
hpf_img=mag_oi>30

###
nJpeg=0
if step_jpeg>0:
    patch_dir1=os.path.join(patch_dir, pat)
    jpeg_dir=os.path.join(patch_dir1, 'step2a_jpeg')
    mkdir(patch_dir1)
    mkdir(jpeg_dir)
    if forcedrun:
        jpgfns=glob(os.path.join(jpeg_dir, '*.jpg'))
        if len(jpgfns)>0:
            print("Delete %g previous jpg files" % len(jpgfns))
            _=[os.remove(jpgfn) for jpgfn in jpgfns]

###

img2perc=1.0/255.0
sz2fr=1.0/(sz_oi*sz_oi)

###
print('step1_summary.shape = '+str(step1_summary.shape))

if True:
    good=True
    
    work_summary=step1_summary.copy()
    if len(work_summary)==0:
        good=False
    
    if good:
        all_mask_fr=work_summary[:, 8]
        all_mask_fr.min()
        ind=all_mask_fr>=mask_cut
        work_summary=work_summary[ind,:]
        print('mask_fr>=%s, nFiltered=%g' % (mask_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good and reduce_overlap and len(work_summary)>n2:
        all_cell2_fr=work_summary[:, 4]
        indorder=np.argsort(-all_cell2_fr)
        ind0=np.arange(len(work_summary))[indorder]
        xy=work_summary[indorder, 1:3]
        marked_thr=2
        marked=mark_overlap(xy, sz_half, threshold=marked_thr)
        if (marked<marked_thr).sum()<n2:
            marked_thr=4
            marked=mark_overlap(xy, sz_half, threshold=marked_thr)
        marked_cut=select_small(marked, n2, pre=0)
        #marked_cut=marked_cut if marked_cut<=marked_thr else marked_thr
        ind=np.sort(ind0[marked<=marked_cut])
        work_summary=work_summary[ind,:]
        print('overlap_mark<=%s, nFiltered=%g' % (marked_cut, len(work_summary)))
    
    if good:
        all_excl_fr=work_summary[:, 5]
        excl_cut=select_small(all_excl_fr, n1, pre=0)
        ind=all_excl_fr<=excl_cut
        work_summary=work_summary[ind,:]
        print('excl_fr<=%s, nFiltered=%g' % (excl_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        all_cell2_fr=work_summary[:, 4]
        cell2_cut=select_large(all_cell2_fr, n2, pre=0.2)
        ind=all_cell2_fr>=cell2_cut
        work_summary=work_summary[ind,:]
        print('cell2_fr>=%s, nFiltered=%g' % (cell2_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        all_hit_fr=work_summary[:, 6]
        hit_cut=select_large(all_hit_fr, n3, pre=0.3)
        ind=all_hit_fr>=hit_cut
        work_summary=work_summary[ind,:]
        print('hit_fr>=%s, nFiltered=%g' % (hit_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        all_hit_avg=work_summary[:, 7]
        avg_cut=select_large(all_hit_avg, n3, pre=0.75)
        ind=all_hit_avg>=avg_cut
        work_summary=work_summary[ind,:]
        print('hit_signal_avg>=%s, nFiltered=%g' % (avg_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    step2a_summary=[]
    iImg=0
    if good:
        nImg=len(work_summary)
        all_blur_fr=np.array(np.zeros(nImg))
        
        step2a_summary=[]
        for i in np.arange(nImg):
            hit_detail=work_summary[i]
            iX, iY=hit_detail[1:3]
            
            im10=ts2im(ts, iX, iY, target_mag=mag_oi, target_sz=sz_oi, show=False)
            
            _, blur_fr=mask_blur(im10, use_strict=hpf_img)
            
            hit_detail2=hit_detail.tolist()[0:9]+[blur_fr]+hit_detail.tolist()[9:]
            hit_detail2[0]=iImg
            step2a_summary.append(hit_detail2)
            iImg=iImg+1
            if i%image_every==0:
                print([pat, nImg, i] + np.round(hit_detail2, 3).tolist() )
                if nJpeg<step_jpeg:
                    step2a_fn="step2a_mag%g_sz%g.%s.origmag%g.X%06i.Y%06i.ind%05g.jpg" % (mag_oi, sz_oi, pat, mag_def, iX, iY, iImg )
                    step2a_fn2=os.path.join(jpeg_dir, step2a_fn)
                    plt.imsave(step2a_fn2, im10)
                    nJpeg+=1
                    
        step2a_summary=np.array(step2a_summary)
    else:
        step2a_summary=np.array([])
    
    step2a_summary=np.array(step2a_summary)
    ofn2a2=ofn2a+".n%s.npy"%len(step2a_summary)
    np.save(ofn2a2, step2a_summary)
    print("Save to %s" % (ofn2a2))

print("Number of Step2a patches = %s" % (len(step2a_summary)))

if lock:
    try:
        os.remove(ofn2ak)
    except:
        print("Unable to delete file: %s" % ofn2ak)

print("## Job '%s' Step2 started: %s" % (pat, starttime) )
print("## Job '%s' Step2 completed: %s" % (pat, time.ctime()) )

# hit_detail=[iImg, iX, iY, cell1_fr, cell2_fr, excl_fr, hit_area_fr, hit_area_signal, mask_fr, blur_fr]

#####




