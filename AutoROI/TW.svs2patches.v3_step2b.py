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
import skimage
import skimage.morphology

###

def hide_args(arglist, exclist=[]):
    for action in arglist:
        if not action in exclist:
            action.help=argparse.SUPPRESS

ver0='V3'
steptag='step2b'
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

h2 = parser.add_argument('-J', "--jpeg", dest='jpeg', metavar='nJpg', type=int, default=0, help="at most save N intermediate jpegs")
h1 = parser.add_argument("-f", "--force", dest='force', action='store_true', default=False, help="force to override previous file")

hidelist=[ h1, h2 ]

new_options={
    'stain_hit_max':     [int, 85, 1, 255,       'upper grayscale of dark region (black=0)'],
    'n4':                [int, 1500, 1, 65535,   'select N4 patches with smaller fr_blur'],
    'n5':                [int, 1000, 1, 65535,   'select N5 patches with larger fr_cell2'],
    'blur_cut0':         [float, 0.1, 0, 1.0,    'max frac of blurry regions (fr_blur)'],
    'fr_blood_max0':     [float, 0.15, 0, 1.0,   'max frac of blood regions (fr_blood)'],
    'max_nucleus_ratio': [float, 3.0, 1.0, 10.0, 'max ratio of nucleus long/short axes'],
    'image_every':       [int, 50, 1, 1000,      'show message every N patches']
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

save_dir=os.path.join(patch_dir, "summary")
ofn0=os.path.join(save_dir, "save.%s.mag%g.sz%g." % (pat, mag_oi, sz_oi))
#all_npy_fns=glob(os.path.join(save_dir, "*.npy"))
all_npy_fns=glob(ofn0 + "*.npy")

ofn2a=os.path.join(save_dir, "save.%s.mag%g.sz%g.step2a%s" % (pat, mag_oi, sz_oi, ver0))
ofn2ak=ofn2a+'.LOCK.npy'
if file_readable(ofn2ak):
    print('Step2a file locked!!')
    sys.exit(1)
    
pat2a=ofn2a+".n[0-9]*.npy"
r2a = re.compile(pat2a)
npylist2a = list(filter(r2a.match, all_npy_fns))

###

if len(npylist2a)>0:
    ofn2a2=npylist2a[-1]
    print('Load previous step2a file: %s' % os.path.basename(ofn2a2))
    step2a_summary=np.load(ofn2a2)
else:
    print('Cannot load step2a from previous npy files: %s' % os.path.basename(ofn2a))
    sys.exit(1)

ofn2b=os.path.join(save_dir, "save.%s.mag%g.sz%g.step2b%s" % (pat, mag_oi, sz_oi, ver0))
ofn2bk=ofn2b+'.LOCK.npy'
if file_readable(ofn2bk):
    if unlock:
        try:
            print('Found lock file: %s' % os.path.basename(ofn2bk))
            os.remove(ofn2bk)
        except:
            print("Unable to unlock flie: %s" % ofn2bk)
    else:
        print("Locked: %s" % ofn2bk)
        sys.exit(1)

pat2b=ofn2b+".n[0-9]*.npy"

r2b = re.compile(pat2b)
npylist2b = list(filter(r2b.match, all_npy_fns))
#npylist2b

if len(npylist2b)>0:
    if ofn2bk in npylist2b:
        print('Found lock file: %s' % os.path.basename(ofn2bk))
        sys.exit(1)
    ofn2b2=npylist2b[-1]
    print('Found previous step2b file: %s' % os.path.basename(ofn2b2))
    if forcedrun:
        ofn2b4=ofn2b2+".bak"
        os.rename(ofn2b2, ofn2b4)
        print("Rename file: %s" % os.path.basename(ofn2b4) )
        ofn2b3=re.sub(ofn2b, ofn2b+'_detail', ofn2b2)
        if file_readable(ofn2b3):
            ofn2b4=ofn2b3+".bak"
            os.rename(ofn2b3, ofn2b4)
            print("Rename file: %s" % os.path.basename(ofn2b4) )
    else:
        sys.exit(0)
else:
    forcedrun=False

##

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

##

os.chdir(workdir)

print("## Job Magnification = %g, Size = %g, FileTag = %s" % (mag_oi, sz_oi, pat))

###

LR_dir=os.path.join(workdir, "LR_model")
LRfn=os.path.join(LR_dir, "%s.LR_coeff.npy" % (svs_tag) )

if file_readable(LRfn):
    print("Load LinearRegression Coefficients: %s" % (LRfn))
    LR_RGB=np.load(LRfn)
else:
    print("Error! No LinearRegression Coefficients file: '%s'" % (LRfn))
    sys.exit(1)

if file_readable(ofn2bk):
    print("Locked: %s" % ofn2bk)
    sys.exit(1)

starttime=time.ctime()
print("## Job '%s' started: %s" % (pat, starttime) )

lock=True
if lock:
    np.save(ofn2bk, [])

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

sz_half=sz_def/2

###

if mag_oi<=8:  ## 5
    min_radius = 1.5
    max_radius = 5
    bad_radius = 10
    min_nucleus_area = 3
    max_nucleus_area = 10 
    bad_nucleus_area = 45
    ngood_cut = 50  
elif mag_oi<=15:  ## 10
    min_radius = 2
    max_radius = 10
    bad_radius = 30
    min_nucleus_area = 8
    max_nucleus_area = 50 
    bad_nucleus_area = 135
    ngood_cut = 100  
elif mag_oi<=30: ## 20
    min_radius = 5
    max_radius = 30
    bad_radius = 50
    min_nucleus_area = 30
    max_nucleus_area = 150 
    bad_nucleus_area = 400
    ngood_cut = 60  
else:    ## 40
    min_radius = 10
    max_radius = 60
    bad_radius = 100
    min_nucleus_area = 120
    max_nucleus_area = 500 
    bad_nucleus_area = 1200
    ngood_cut = 30  

print('mag_oi', '[min_radius, max_radius, bad_radius, min_nucleus_area, max_nucleus_area, bad_nucleus_area, ngood_cut]')
print(mag_oi, [min_radius, max_radius, bad_radius, min_nucleus_area, max_nucleus_area, bad_nucleus_area, ngood_cut])

###

nJpeg=0
if step_jpeg>0:
    patch_dir1=os.path.join(patch_dir, pat)
    jpeg_dir=os.path.join(patch_dir1, 'step2b_jpeg')
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

print('step2a_summary.shape = '+str(step2a_summary.shape))

if True:
    good=True
    
    work_summary=step2a_summary.copy()
    if len(work_summary)==0:
        good=False
    
    if good:
        all_blur_fr=work_summary[:, 9]
        blur_cut=select_small(all_blur_fr, n4, 0.01)
        if blur_cut>blur_cut0:
            blur_cut=blur_cut0
        ind=all_blur_fr <= blur_cut0
        work_summary=work_summary[ind,:]
        print('blur_fr<=%s, nFiltered=%g' % (blur_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    if good:
        all_cell2_fr=work_summary[:, 4]
        cell2_cut=select_large(all_cell2_fr, n5)
        ind=all_cell2_fr>=cell2_cut
        work_summary=work_summary[ind,:]
        print('cell2_fr>=%s, nFiltered=%g' % (cell2_cut, len(work_summary)))
        if len(work_summary)==0:
            good=False
    
    step2b_summary=[]
    all_detail_good=np.array([])
    iImg=0
    if good:
        nImg=len(work_summary)
        all_detail_good=None
        step2b_summary=[]
        for i in np.arange(nImg):
            hit_detail=work_summary[i]
            hit_detail2=hit_detail.tolist()
            iX, iY=hit_detail[1:3]
            
            im10=ts2im(ts, iX, iY, target_mag=mag_oi, target_sz=sz_oi, show=False)
            
            im10norm=im2norm(im10, LR_RGB)
            im10N_Rmsk=mask_brown(im10norm, r=[90, 255], g=[10, 128], b=[10, 128])
            fr_blood=im10N_Rmsk.mean()
            if fr_blood<=fr_blood_max0:
              im10N_stain=im2stain(im10norm)[:, :, 0]
              im10N_stain_adj=stain2adj2(im10N_stain)
              im10N_signal=1.0- img2perc*im10N_stain
              im10N_signal_hit=im10N_stain_adj <= stain_hit_max
              im_msk=im10N_signal_hit
              im_msk[im10N_Rmsk]=False
              im_msk=scipy.ndimage.morphology.binary_dilation(im_msk, iterations=2)
              im_msk = scipy.ndimage.morphology.binary_fill_holes(im_msk)
              im_msk=scipy.ndimage.morphology.binary_erosion(im_msk, iterations=3)
              im_msk=scipy.ndimage.morphology.binary_dilation(im_msk)
              # run adaptive multi-scale LoG filter
              if im_msk.max()>0:
                im_nuclei_stain=im10N_stain_adj
                im_fgnd_mask=im_msk
                im_log_max, im_sigma_max = htk.filters.shape.cdog(
                        im_nuclei_stain, im_fgnd_mask,
                        sigma_min=min_radius * np.sqrt(2),
                        sigma_max=max_radius * np.sqrt(2) )
                
                # detect and segment nuclei using local maximum clustering
                local_max_search_radius = 3
                im_nuclei_seg_mask1, seeds, maxima = htk.segmentation.nuclear.max_clustering(im_log_max, im_fgnd_mask, local_max_search_radius)
                
                if im_nuclei_seg_mask1.max()>0:
                    im_nuclei_seg_mask2 = htk.segmentation.label.area_open(im_nuclei_seg_mask1, min_nucleus_area).astype(np.int)
                    
                    if im_nuclei_seg_mask2.max()>0:
                        im_region_tbl=skimage.measure.regionprops_table(im_nuclei_seg_mask2, properties=(
                                                                         'major_axis_length',
                                                                         'minor_axis_length',
                                                                         'area'))
                        im_rg_w1=im_region_tbl['major_axis_length']
                        im_rg_w2=im_region_tbl['minor_axis_length']
                        im_rg_w2[im_rg_w2<0.1]=0.1
                        im_rg_area=im_region_tbl['area']
                        #im_rg_avg=im_region_tbl['mean_intensity']
                        im_rg_avg=np.zeros(len(im_rg_area))
                        
                        ind_area1=np.logical_and(im_rg_area>=min_nucleus_area, im_rg_area<=max_nucleus_area)
                        ind_width1=np.logical_and(im_rg_w2>=min_radius, im_rg_w1<=max_radius)
                        ind_ratio=(im_rg_w1/im_rg_w2)<=max_nucleus_ratio
                        
                        ind_good=np.logical_and(ind_area1, np.logical_and(ind_width1, ind_ratio))
                        nn_good=ind_good+1
                        nGood=ind_good.sum()
                        
                        if nGood>=ngood_cut:
                            tmp0=np.zeros(nGood)+iImg
                            tmp1=im_rg_w1[ind_good]
                            tmp2=im_rg_w2[ind_good]
                            tmp3=im_rg_area[ind_good]
                            tmp4=im_rg_avg[ind_good]
                            
                            for j in np.arange(nGood):
                                igg=nn_good[j]
                                msk=(im_nuclei_seg_mask2==igg)
                                igg_intensity=im10N_signal[msk]
                                igg_avg=igg_intensity.mean()
                                tmp4[j]=igg_avg
                            
                            detail_good=np.c_[tmp0, tmp1, tmp2, tmp3, tmp4]
                            if all_detail_good is None:
                                all_detail_good=detail_good
                            else:
                                all_detail_good=np.r_[all_detail_good, detail_good]
                            
                            ind_bad=np.logical_or(im_rg_area>=bad_nucleus_area, im_rg_w1>=bad_radius)
                            nBad=ind_bad.sum()
                            
                            nn_bad=np.array([x for x in np.arange(len(ind_bad)) if ind_bad[x] ])
                            nn1_bad=nn_bad+1
                            
                            area_bad=im_rg_area[ind_bad].sum()
                            
                            hit_detail2=hit_detail.tolist()[:10]+[nGood, nBad, sz2fr*area_bad, fr_blood]+hit_detail.tolist()[10:]
                            hit_detail2[0]=iImg
                            step2b_summary.append(hit_detail2)
                            iImg=iImg+1
            
            if i%image_every==0:
                print([pat, nImg, i] + np.round(hit_detail2, 3).tolist())
                if nJpeg<step_jpeg:
                    step2b_fn="step2b_mag%g_sz%g.%s.origmag%g.X%06i.Y%06i.ind%05g.jpg" % (mag_oi, sz_oi, pat, mag_def, iX, iY, iImg )
                    step2b_fn2=os.path.join(jpeg_dir, step2b_fn)
                    plt.imsave(step2b_fn2, im10norm)
                    nJpeg+=1
                    
        step2b_summary=np.array(step2b_summary)
    else:
        step2b_summary=np.array([])
    
    step2b_summary=np.array(step2b_summary)
    ofn2b2=ofn2b+".n%s.npy"%len(step2b_summary)
    np.save(ofn2b2, step2b_summary)
    ofn2b3=ofn2b+"_detail.n%s.npy"%len(step2b_summary)
    np.save(ofn2b3, all_detail_good)
    print("Save to %s" % (ofn2b2))

print("Number of Step2b patches = %s" % (len(step2b_summary)))

if lock:
    try:
        os.remove(ofn2bk)
    except:
        print("Unable to delete file: %s" % ofn2bk)

print("## Job '%s' Step2b started: %s" % (pat, starttime) )
print("## Job '%s' Step2b completed: %s" % (pat, time.ctime()) )

# hit_detail=[iImg, iX, iY, cell1_fr, cell2_fr, excl_fr, hit_area_fr, hit_area_signal, mask_fr, blur_fr, nGood, nBad, bad_fr]

#####

