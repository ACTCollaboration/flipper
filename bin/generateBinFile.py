#!/bin/env python

import os,sys
import argparse
import numpy as np

#- 
# genBinFile.py
#-
# Author: DW
# Status: Development 
#
# This script generates a bin file given l_min, l_max, and delta_l (step).
# The output is saved under param directory
#

def __terminate__():
    print("terminate process")
    sys.exit()

argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

argparser.add_argument('-lm', '--lmin',        
        default='100',
        help='minimum ell value to use')

argparser.add_argument('-lM', '--lmax',
        default='5000',
        help='maximum ell value to use')

argparser.add_argument('-d', '--deltal',
        default='100',
        help='bin size')

argparser.add_argument('-f', '--fname',
        default='',
        help='name of the output bin file')

args = argparser.parse_args()
        
lmin        = float(args.lmin)
lmax        = float(args.lmax)
delta_l     = float(args.deltal)

output_dir  = "../params"
output_file = args.fname if args.fname else "BIN_%s_%s_%s" \
        %(args.lmin, args.lmax, args.deltal)
output_file = os.path.join(output_dir, output_file)

#basic validation
if(lmin < 0.0):
    print('lmin has to be equal to or greater than 0')
    __terminate__()
elif(lmax <= lmin):
    print('lmax must be bigger than lmin')
    __terminate__()
elif(delta_l <= 0.0):
    print('step size must be bigger than 0')
    __terminate__() 
else:
    pass

nbin = int(np.floor((lmax - lmin) / delta_l)) # number of bins

print("Binning File %s, lmin %0.2f, lmax %0.2f, binsize %0.2f, # bins %d" \
        %(output_file, lmin, lmax, delta_l, nbin))

with open(output_file, "wb") as handle:
    handle.write(str(nbin) + '\n') # write number of bin

    ctr     = 0
    
    for binL in np.arange(lmin, lmax, delta_l):
        binC = binL + delta_l/2.0
        binU = binL + delta_l

        if ctr > nbin: 
            break # do I need this?
        else:
            ctr += 1
            handle.write("%0.2f %0.2f %0.2f\n" % (binL, binU, binC))
    
    handle.close()






















