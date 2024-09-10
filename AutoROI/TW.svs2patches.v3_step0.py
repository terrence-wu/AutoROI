#!/usr/bin/env python3

import sys
import argparse

def hide_args(arglist, exclist=[]):
    for action in arglist:
        if not action in exclist:
            action.help=argparse.SUPPRESS

ver='SVS2patches v3.0 by Terrence Wu'+ ": " + "steps"
ver0='V3'
parser = argparse.ArgumentParser(
                prog='TW.svs2patches.step1'+ver0+'.py', 
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
h1 = parser.add_argument("-f", "--force", dest='force', action='store_true', default=False, help="force to override previous files")

hidelist=[ h1, h2 ]

new_options={
    'stain_hit_max':    [int, 85, 1, 255,    'upper grayscale of dark region (black=0)'],
    'cell1_minfr0':     [float, 0.9, 0, 1.0, 'min frac of HTK cellularity1 overlap'],
    'cell2_minfr0':     [float, 0.1, 0, 1.0, 'min frac of HTK cellularity2 overlap'],
    'image_every':      [int, 50, 1, 1000,   'show message every N patches']
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
step1_jpeg=options.jpeg

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

print(options)
sys.exit(0)

