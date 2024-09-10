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
import skimage.morphology

###

def hide_args(arglist, exclist=[]):
    for action in arglist:
        if not action in exclist:
            action.help=argparse.SUPPRESS

ver0='V3'
steptag='step3'
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
h2 = parser.add_argument('-J', "--jpeg", dest='jpeg', metavar='nJpg', type=int, default=500, help="at most save N final jpegs")
h1 = parser.add_argument("-f", "--force", dest='force', action='store_true', default=False, help="force to override previous files")

hidelist=[ h1, h2, h3 ]

new_options={
    'stain_hit_max':      [int, 85, 1, 255,       'upper grayscale of dark region (black=0)'],
    'n6':                 [int, 600, 1, 65535,    'select N6 patches with smaller fr_blur'],
    'n7':                 [int, 500, 1, 65535,    'select N7 patches with larger fr_good'],
    'n8':                 [int, 400, 1, 65535,    'select N8 patches with smaller fr_bad'],
    'n9':                 [int, 350, 1, 65535,    'select N9 patches with larger fr_cell2'],
    'n10':                [int, 300, 1, 65535,    'select N10 patches with larger fr_tissue'],
    'blur_cut1':          [float, 0.05, 0, 1.0,   'max of fr_blur'],
    'fr_blood_max':       [float, 0.1, 0, 1.0,    'max of fr_blood'],
    'fr_white_max':       [float, 0.2, 0, 1.0,    'max of fr_white'],
    'fr_blood_white_max': [float, 0.25, 0, 1.0,   'max of frac of blood or white regions'],
    'fr_tissue_min':      [float, 0.2, 0, 1.0,    'min fraction of tissue region (fr_tissue)'],
    'max_nucleus_ratio':  [float, 3.0, 1.1, 10.0, 'max ratio of nucleus long/short axes'],
    'good_area_fr_min':   [float, 0, 0, 1.0,      'min frac of nuclear region (fr_good), 0=auto'],
    'bad_area_fr_max':    [float, 0.25, 0, 1.0,   'max frac of not-nucleus dark region (fr_bad)'],
    'image_every':        [int, 50, 1, 1000,      'show message every N patches']
}

##

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
remove_overlapping=options.overlap

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

####

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
            img=img*255
            img[img>255]=255
            img[img<0]=0
    return(img)

def mask_white(imgw, min_rgb=200, max_diff=55, erosion=3):
    assert imgw.ndim==3, 'must be a 3-d image array'
    imgw=image255(imgw)
    rgb_min=np.float32(np.min(imgw, axis=2))
    rgb_max=np.float32(np.max(imgw, axis=2))
    msk=np.logical_and(rgb_min>=min_rgb, (rgb_max-rgb_min)<=max_diff)
    if erosion>0:
        msk=scipy.ndimage.morphology.binary_erosion(msk, iterations=erosion)
        msk=scipy.ndimage.morphology.binary_dilation(msk, iterations=erosion)
    return(msk)

def mask_tissue(imgw, r=[130, 255], g=[100, 180], b=[150, 255], r_g_min=15, b_r_min= -45, fill=True):
    assert imgw.ndim==3, 'must be a 3-d image array'
    
    if len(r)==1:
        r=np.array(r[0]-10, r[0]+10)
    else:
        r=np.sort(np.array(r))[0:2]
    if len(g)==1:
        g=np.array(g[0]-10, g[0]+10)
    else:
        g=np.sort(np.array(g))[0:2]
    if len(b)==1:
        b=np.array(b[0]-10, b[0]+10)
    else:
        b=np.sort(np.array(b))[0:2]
    
    imgw=np.float32(image255(imgw))
    r_g=np.float32(imgw[:, :, 0])-np.float32(imgw[:, :, 1])
    msk_r_g=r_g>=r_g_min
    b_r=np.float32(imgw[:, :, 2])-np.float32(imgw[:, :, 0])
    msk_b_r=b_r>=b_r_min
    msk_r=np.logical_and(imgw[:, :, 0]>=r[0], imgw[:, :, 0]<=r[1])
    msk_g=np.logical_and(imgw[:, :, 1]>=g[0], imgw[:, :, 1]<=g[1])
    msk_b=np.logical_and(imgw[:, :, 2]>=b[0], imgw[:, :, 2]<=b[1])
    msk=np.logical_and(np.logical_and(msk_r, np.logical_and(msk_g, msk_b)), np.logical_and(msk_r_g, msk_b_r))
    if fill:
        msk=mask_fill(msk, dilation=2, erosion=2, max_hole=200)
    return(msk)

def mask_brown(imgw, r=[90, 170], g=[10, 80], b=[10, 80], r_gb_min=51, diff_gb_max=30, fill=True):
    assert imgw.ndim==3, 'must be a 3-d image array'
    if len(r)==1:
        r=np.array(r[0]-10, r[0]+10)
    else:
        r=np.sort(np.array(r))[0:2]
    if len(g)==1:
        g=np.array(g[0]-10, g[0]+10)
    else:
        g=np.sort(np.array(g))[0:2]
    if len(b)==1:
        b=np.array(b[0]-10, b[0]+10)
    else:
        b=np.sort(np.array(b))[0:2]
    
    imgw=np.float32(image255(imgw))
    gb_max=np.float32(np.max(imgw[:, :, 1:3], axis=2))
    msk_r_gb=(np.float32(imgw[:,:,0])-gb_max)>=r_gb_min
    gb_diff=np.abs(np.float32(imgw[:,:,1])-np.float32(imgw[:,:,2]))
    msk_gb_diff=gb_diff<=diff_gb_max
    
    msk_r=np.logical_and(imgw[:, :, 0]>=r[0], imgw[:, :, 0]<=r[1])
    msk_g=np.logical_and(imgw[:, :, 1]>=g[0], imgw[:, :, 1]<=g[1])
    msk_b=np.logical_and(imgw[:, :, 2]>=b[0], imgw[:, :, 2]<=b[1])
    msk=np.logical_and(np.logical_and(msk_r, np.logical_and(msk_g, msk_b)), np.logical_and(msk_r_gb, msk_gb_diff))
    if fill:
        msk=mask_fill(msk, dilation=3, erosion=2, max_hole=1000)
    return(msk)

def mask_fill(msk, dilation=3, erosion=2, max_hole=1000):
    msk2=msk.copy()
    msk2=scipy.ndimage.morphology.binary_dilation(msk2, iterations=dilation)
    if max_hole>0:
        msk2=skimage.morphology.remove_small_holes(msk2, area_threshold=max_hole)
    msk2=scipy.ndimage.morphology.binary_erosion(msk2, iterations=dilation+erosion)
    msk2=scipy.ndimage.morphology.binary_dilation(msk2, iterations=erosion)
    return(msk2)

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

img_npy_dir=os.path.join(patch_dir, 'image_save')
mkdir(img_npy_dir)

save_dir=os.path.join(patch_dir, "summary")
ofn0=os.path.join(save_dir, "save.%s.mag%g.sz%g." % (pat, mag_oi, sz_oi))
#all_npy_fns=glob(os.path.join(save_dir, "*.npy"))
all_npy_fns=glob(ofn0 + "*.npy")

####

ofn3a=os.path.join(save_dir, "save.%s.mag%g.sz%g.step3a%s" % (pat, mag_oi, sz_oi, ver0))
ofn3b=os.path.join(save_dir, "save.%s.mag%g.sz%g.step3b%s" % (pat, mag_oi, sz_oi, ver0))
ofn3bk=ofn3b+'.LOCK.npy'
if file_readable(ofn3bk):
    if unlock:
        print('Found lock file: %s' % os.path.basename(ofn3bk))
        try:
            os.remove(ofn3bk)
        except:
            print("Unable to unlock flie: %s" % ofn3bk)
    else:
        print("Locked: %s" % ofn3bk)
        sys.exit(1)

pat3a=ofn3a+".n[0-9]*.npy"
r3a = re.compile(pat3a)
npylist3a = list(filter(r3a.match, all_npy_fns))

pat3b=ofn3b+".n[0-9]*.npy"
r3b = re.compile(pat3b)
npylist3b = list(filter(r3b.match, all_npy_fns))

if len(npylist3b)>0:
    if ofn3bk in npylist3b:
        print('Found lock file: %s' % os.path.basename(ofn3bk))
        sys.exit(1)
    ofn3b2=npylist3b[-1]
    print('Found previous step3b file: %s' % os.path.basename(ofn3b2))
    if forcedrun:
        ofn3b4=ofn3b2+".bak"
        os.rename(ofn3b2, ofn3b4)
        print("Rename file: %s" % os.path.basename(ofn3b4) )
        pat3b3=os.path.join(img_npy_dir, "%s.mag%g.sz%g.step3b%s.n*.Images.npy" % (pat, mag_oi, sz_oi, ver0))
        npy3b3=glob(pat3b3)
        if len(npy3b3)>0:
            ofn3b3=npy3b3[-1]
            ofn3b4=ofn3b3+".bak"
            os.rename(ofn3b3, ofn3b4)
            print("Rename file: %s" % os.path.basename(ofn3b4) )
    else:
        sys.exit(0)
else:
    forcedrun=False

step3a_summary=None
if len(npylist3a)>0:
    ofn3a2=npylist3a[-1]
    print('Found previous step3a file: %s' % os.path.basename(ofn3a2))
    if forcedrun:
        ofn3a3=ofn3a2+".bak"
        os.rename(ofn3a2, ofn3a3)
        print("Rename file: %s" % os.path.basename(ofn3a3) )
    else:
        step3a_summary=np.load(ofn3a2)

if step3a_summary is None:
    ofn2b=os.path.join(save_dir, "save.%s.mag%g.sz%g.step2b%s" % (pat, mag_oi, sz_oi, ver0))
    ofn2bk=ofn2b+'.LOCK.npy'
    if file_readable(ofn2bk):
        print('Step2b file locked!!')
        sys.exit(1)
    
    pat2b=ofn2b+".n[0-9]*.npy"
    r2b = re.compile(pat2b)
    npylist2b = list(filter(r2b.match, all_npy_fns))
    
    if len(npylist2b)>0:
        ofn2b2=npylist2b[-1]
        print('Load previous step2b file: %s' % ofn2b2)
        step2b_summary=np.load(ofn2b2)
        nImg=len(step2b_summary)
        ofn2b3=re.sub('\.n%s.npy' % (nImg), '_detail.n%s.npy' % (nImg), ofn2b2)
        if file_readable(ofn2b3):
            step2b_detail=np.load(ofn2b3)
            print('Load previous detail file: %s' % ofn2b3)
        else:
            if nImg>0:
                print('Cannot load step2b bbox detail file: %s' % ofn2b3)
                sys.exit(1)
            else:
                step2b_detail=np.array([])
    else:
        print('Cannot load step2b from previous npy files: %s' % ofn2b)
        sys.exit(1)

if file_readable(ofn3bk):
    print("Locked: %s" % ofn3bk)
    sys.exit(1)

import time
starttime=time.ctime()
print("## Job '%s' started: %s" % (pat, starttime) )
os.chdir(workdir)

print("## Job Magnification = %g, Size = %g, FileTag = %s" % (mag_oi, sz_oi, pat))

lock=True
if lock:
    np.save(ofn3bk, [])

###

if mag_oi<=8:  ## 5
    min_radius = 1.5
    max_radius = 5
    bad_radius = 10
    min_nucleus_area = 3
    max_nucleus_area = 10 
    bad_nucleus_area = 45
    argood_fr_cut = 0.002
    ngood_cut = 50
elif mag_oi<=15:  ## 10
    min_radius = 2
    max_radius = 10
    bad_radius = 30
    min_nucleus_area = 8
    max_nucleus_area = 50 
    bad_nucleus_area = 135
    argood_fr_cut = 0.015
    ngood_cut = 100  
elif mag_oi<=30: ## 20
    min_radius = 5
    max_radius = 30
    bad_radius = 50
    min_nucleus_area = 30
    max_nucleus_area = 150 
    bad_nucleus_area = 400
    argood_fr_cut = 0.03
    ngood_cut = 60  
else:    ## 40
    min_radius = 10
    max_radius = 60
    bad_radius = 100
    min_nucleus_area = 120
    max_nucleus_area = 500 
    bad_nucleus_area = 1200
    argood_fr_cut = 0.04
    ngood_cut = 30  

if good_area_fr_min>0:
    argood_fr_cut=good_area_fr_min

print('mag_oi', '[min_radius, max_radius, bad_radius, min_nucleus_area, max_nucleus_area, bad_nucleus_area, argood_fr_cut, ngood_cut]')
print(mag_oi, [min_radius, max_radius, bad_radius, min_nucleus_area, max_nucleus_area, bad_nucleus_area, argood_fr_cut, ngood_cut])

###

img2perc=1.0/255.0
sz2fr=1.0/(sz_oi*sz_oi)

###

if step3a_summary is None:
    print('step2b_summary.shape = '+str(step2b_summary.shape))
    print('step2b_detail.shape = '+str(step2b_detail.shape))
    
    if len(step2b_summary)==0:
        good=False
    else:
        work_summary=step2b_summary[:,0:13].copy()
        good=True
    
    if good:
        if len(step2b_detail)>0:
            work_detail=step2b_detail
            
            all_major_radius=work_detail[:,1]
            all_minor_radius=work_detail[:,2]
            all_hit_area=work_detail[:,3]
            
            tmp19=np.percentile(all_minor_radius, [10, 90, 99])
            tt=(3*tmp19[0] + tmp19[1])/4
            radius_cut=tt
            #print('all_minor_radius', tt, ' ', tmp19)
            ind_radius=np.logical_and(all_minor_radius>=tt, all_minor_radius<=tmp19[2])
            
            tmp19=np.percentile(all_hit_area, [10, 90, 99])
            tt=(3*tmp19[0] + tmp19[1])/4
            area_cut=tt
            #print('all_hit_area', tt, ' ', tmp19)
            ind_area=np.logical_and(all_hit_area>=tt, all_hit_area<=tmp19[2])
            
            #print('all_minor_radius', np.percentile(all_minor_radius, [0, 25, 50, 75, 100]))
            #print('all_hit_area', np.percentile(all_hit_area, [0, 25, 50, 75, 100]))
            
            all_minor_radius[all_minor_radius<0.1]=0.1
            all_radius_ratio=all_major_radius/all_minor_radius
            ind_ratio=all_radius_ratio<=max_nucleus_ratio
            
            #print('all_radius_ratio', np.percentile(all_radius_ratio, [0, 25, 50, 75, 100]))
            
            ind=np.logical_and(ind_radius, np.logical_and(ind_area, ind_ratio))
            if ind.sum()>0:
                sel_detail=work_detail[ind,]
                all_ind2=sel_detail[:,0]
                all_area2=sel_detail[:,3]
                tmp2=np.unique(all_ind2, return_counts=True)
                sel_ind=tmp2[0]
                sel_freq=tmp2[1]
                #print('sel_freq', sel_freq)
                ind3=sel_freq>=ngood_cut
                sel2_ind=np.int32(sel_ind[ind3])
                sel2_nGood=sel_freq[ind3]
                sel2_arGood=np.zeros(len(sel2_ind))
                for j in np.arange(len(sel2_ind)):
                    ii=sel2_ind[j]
                    ind4=all_ind2==ii
                    ii_area=(all_area2[ind4]).sum()
                    sel2_arGood[j]=ii_area
                work_summary=work_summary[sel2_ind,]
                work_summary[:,10]=sel2_nGood
                work_summary=np.c_[work_summary, sel2_arGood]
            else:
                work_summary=np.array([])
                good=False
            print('nucArea>=%s, nucRadius>=%s, nGood >= %s, nFiltered=%g' % (np.round(area_cut,3), np.round(radius_cut,3), ngood_cut, len(work_summary)))
        else:
            all_ngood=work_summary[:, 10]
            ind=all_ngood >= ngood_cut
            work_summary=work_summary[ind,:]
            print('nGood >= %s, nFiltered=%g' % (ngood_cut, len(work_summary)))
            if len(work_summary)==0:
                good=False
            else:
                work_summary=np.c_[work_summary, np.zeros(len(work_summary))+np.inf]
    
    if good:
        all_blur_fr=work_summary[:, 9]
        ind=all_blur_fr<=blur_cut1
        work_summary=work_summary[ind,:]
        print('blur_fr <= %s, nFiltered=%g' % (blur_cut1, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        ar_good=work_summary[:, 13]
        ar_good_fr=ar_good/(sz_oi*sz_oi)
        print('ar_good_fr', np.percentile(ar_good_fr, [0, 25, 50, 75, 100]))
        argood_cut0=argood_fr_cut*sz_oi*sz_oi
        ind=ar_good>=argood_cut0
        work_summary=work_summary[ind,:]
        print('good_area_fr >= %s, nFiltered=%g' % (argood_fr_cut, len(work_summary)))
    
    if good:
        all_bad_fr=work_summary[:, 12]
        ind=all_bad_fr <= bad_area_fr_max
        work_summary=work_summary[ind,:]
        print('bad_area_fr <= %s, nFiltered=%g' % (bad_area_fr_max, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        all_blur_fr=work_summary[:, 9]
        blur_cut=select_small(all_blur_fr, n6)
        ind=all_blur_fr<=blur_cut
        if ind.sum()<len(work_summary):
            work_summary=work_summary[ind,:]
            print('blur_fr <= %s, nFiltered=%g' % (blur_cut, len(work_summary)))
            if len(work_summary)==0:
                good=False
    
    if good:
        ar_good=work_summary[:, 13]
        argood_cut=select_large(ar_good, n7)
        ind=ar_good>=argood_cut
        if ind.sum()<len(work_summary):
            work_summary=work_summary[ind,:]
            print('area_Good >= %s, nFiltered=%g' % (argood_cut, len(work_summary)))
    
    if good:
        all_bad_fr=work_summary[:, 12]
        bad_area_cut=select_small(all_bad_fr, n8)
        ind=all_bad_fr <= bad_area_cut
        if ind.sum()<len(work_summary):
            work_summary=work_summary[ind,:]
            print('bad_area_fr <= %s, nFiltered=%g' % (bad_area_cut, len(work_summary)))
            if len(work_summary)==0:
                good=False
    
    if good:
        all_cell2_fr=work_summary[:, 4]
        cell2_cut=select_large(all_cell2_fr, n9)
        ind=all_cell2_fr>=cell2_cut
        work_summary=work_summary[ind,:]
        print('cell2_fr>=%s, nFiltered=%g' % (cell2_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        nImg=len(work_summary)
        step3a_summary=work_summary.copy()
        step3a_summary[:,0]=np.arange(nImg)
        step3a_summary=np.array(step3a_summary)
    else:
        step3a_summary=np.array([])
    
    step3a_summary=np.array(step3a_summary)
    ofn3a2=ofn3a+".n%s.npy"%len(step3a_summary)
    np.save(ofn3a2, step3a_summary)
    print("Save to %s" % (ofn3a2))

print("Number of Step3a patches = %s" % (len(step3a_summary)))

## step3b

svs_dir=os.path.join(workdir, "SVS")

if not dir_exists(svs_dir):
    print("## Working Folder should have a subfolder: SVS")
    print('   Error: SVS Folder not found!!')
    sys.exit(1)

all_svs_fns=glob(os.path.join(svs_dir, "*.svs"))

pat1=os.path.join(svs_dir, pat) + ".*"

r = re.compile(svs_dir+"/"+pat+".*")
newlist = list(filter(r.match, all_svs_fns))
#newlist

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

###

ts = large_image.getTileSource(svs_fn)
print(ts.getNativeMagnification())

ts_meta=ts.getMetadata()

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

sz_half=sz_def/2

##

LR_dir=os.path.join(workdir, "LR_model")
LRfn=os.path.join(LR_dir, "%s.LR_coeff.npy" % (svs_tag) )

if file_readable(LRfn):
    print("Load LinearRegression Coefficients: %s" % (LRfn))
    LR_RGB=np.load(LRfn)
else:
    print("Error! No LinearRegression Coefficients file: '%s'" % (LRfn))
    sys.exit(1)

###

if True:
    good=True
    
    work_summary=step3a_summary.copy()
    if len(work_summary)==0:
        step3b_summary=[]
        good=False
    
    if good:
        work_summary=step3a_summary.copy()
        all_ngood=work_summary[:, 10]
        ind=np.argsort(-all_ngood)
        work_summary=work_summary[ind,:]
        
        nImg=len(work_summary)
        ss_selected=np.zeros(nImg)>0
        ss_notoverlap=np.ones(nImg)>0
        
        all_iX=work_summary[:, 1]
        all_iY=work_summary[:, 2]
        sz_diff=sz_def
        step3b_summary=[]
        iImg=0
        for i in np.arange(nImg):
            hit_detail=work_summary[i]
            iX, iY=hit_detail[1:3]
            im10=ts2im(ts, iX, iY, target_mag=mag_oi, target_sz=sz_oi, show=False)
            im10norm=im2norm(im10, LR_RGB)
            
            selectone=False
            msk_white=mask_white(im10norm, erosion=3)
            fr_white=msk_white.mean()
            if fr_white<=fr_white_max:
                msk_tissue=mask_tissue(im10norm, fill=True)
                fr_tissue=msk_tissue.mean()
                if fr_tissue>=fr_tissue_min:
                    msk_blood=mask_brown(im10norm, fill=True)
                    msk_blood_white=np.logical_or(msk_blood, msk_white)
                    fr_blood=msk_blood.mean()
                    fr_blood_white=msk_blood_white.mean()
                    if fr_blood<=fr_blood_max and fr_blood_white<=fr_blood_white_max:
                        selectone=True
            
            if selectone and ss_notoverlap[i]:
                ss_selected[i]=True
                hit_detail[0]=iImg
                hit_detail2=np.append(hit_detail.tolist()[:13], [fr_blood, fr_white, fr_blood_white, fr_tissue])
                step3b_summary.append(hit_detail2)
                iImg=iImg+1
                if remove_overlapping:
                    chkX=np.abs(all_iX-iX)
                    chkY=np.abs(all_iY-iY)
                    chkXY=(chkX+chkY)
                    ind=(chkXY < sz_diff)
                    #ind=np.logical_and(np.logical_and(all_iX>=iX-sz_diff, all_iX<=iX+sz_diff), np.logical_and(all_iY>=iY-sz_diff, all_iY<=iY+sz_diff))
                    ind[:(i+1)]=False
                    ss_notoverlap[ind]=False
            else:
                hit_detail2=hit_detail
            if i%image_every==0:
                print([pat, nImg, i, iImg-1 ] + np.round(hit_detail2, 3).tolist())
        
        if len(step3b_summary)>0:
            step3b_summary=np.array(step3b_summary)
            all_fr_tissue=step3b_summary[:,16]
            fr_tissue_cut=select_large(all_fr_tissue, n10, pre=0.8)
            ind=all_fr_tissue>=fr_tissue_cut
            if ind.sum()<len(step3b_summary):
                step3b_summary=step3b_summary[ind,:]
                print('fr_tissue >= %s, nFiltered=%g' % (fr_tissue_cut, len(step3b_summary)))
    else:
        step3b_summary=[]
    
    step3b_summary=np.array(step3b_summary)
    nImg2=len(step3b_summary)
    ofn3b2=ofn3b+".n%s.npy" % (nImg2)
    np.save(ofn3b2, step3b_summary)
    print("Save to %s" % (ofn3b2))
    print("Number of Step3b patches = %s" % (nImg2))
        
if lock:
    try:
        os.remove(ofn3bk)
    except:
        print("Unable to delete file: %s" % ofn3bk)


### Save images
nImg2=len(step3b_summary)
ofn3b3=os.path.join(img_npy_dir, "%s.mag%g.sz%g.step3b%s.n%s.Images.npy" % (pat, mag_oi, sz_oi, ver0, nImg2))

good=True
if file_readable(ofn3b3):
    if forcedrun:
        os.rename(ofn3b3, ofn3b3+'.bak')
        print("Rename file: %s" % os.path.basename(ofn3b3)+'.bak' )
    else:
        good=False
        print("Image file already exists: %s" % ofn3b3)

patch_dir1=os.path.join(patch_dir, pat)

if step_jpeg>0:
    mkdir(patch_dir1)
    if forcedrun:
        jpat=os.path.join(patch_dir1, "norm_patch_mag%g_sz%g.%s.*.jpg" % (mag_oi, sz_oi, pat) )
        jpgfns=glob(jpat)
        #print(jpat); print(np.array(jpgfns))
        if len(jpgfns)>0:
            bk_dir=os.path.join(patch_dir1, 'bak_jpeg')
            mkdir(bk_dir)
            print("Move %g jpg files to %s" % (len(jpgfns), bk_dir) )
            _=[ os.rename(jpn, re.sub(patch_dir1, bk_dir, jpn)+'.bak' ) for jpn in jpgfns]

if good:
    work_summary=step3b_summary
    nImg2=len(work_summary)
    
    all_norm_imgs=[]
    if nImg2>0:
        for i in np.arange(nImg2):
            hit_detail=work_summary[i]
            iX, iY=hit_detail[1:3]
            
            im10=ts2im(ts, iX, iY, target_mag=mag_oi, target_sz=sz_oi, show=False)
            
            im10norm=im2norm(im10, LR_RGB)
            all_norm_imgs.append(im10norm)
            if i<step_jpeg:
                npatchfn="norm_patch_mag%g_sz%g.%s.origmag%g.X%06i.Y%06i.ind%05g.jpg" % (mag_oi, sz_oi, pat, mag_def, iX, iY, hit_detail[0] )
                npatchfn2=os.path.join(patch_dir1, npatchfn)
                plt.imsave(npatchfn2, im10norm)
        if step_jpeg>0:
            print("Save %s patches to %s" % (min(step_jpeg, nImg2), patch_dir1))
    
    all_norm_imgs=np.array(all_norm_imgs)
    np.save(ofn3b3, all_norm_imgs)
    print("Save %s patches to %s" % (nImg2, ofn3b3))

###

print("## Job '%s' Step3 started: %s" % (pat, starttime) )
print("## Job '%s' Step3 completed: %s" % (pat, time.ctime()) )

# hit_detail=[iImg, iX, iY, cell1_fr, cell2_fr, excl_fr, hit_area_fr, hit_area_signal, mask_fr, blur_fr, nGood, nBad, bad_fr]

#####


