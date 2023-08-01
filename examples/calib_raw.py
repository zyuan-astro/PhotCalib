import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import numpy as np
from astropy.table import Table
import torch

from photcalib import generate_newcat, deform, argparse_apply_model


in_path = "data/"
mod_path = "model/"
out_path = "output/"

args =  argparse_apply_model()

run = args.run
input = args.input

if args.device != None:

    if args.device == 'gpu':
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    else:
        DEVICE = torch.device(args.device)

    print (DEVICE)

else:

    DEVICE = torch.device("cpu")

    print (DEVICE)




mod_files = glob.glob(mod_path+"*%s*.mod"%run) 

mod_file = max(mod_files, key=os.path.getctime)
print ("run model:", mod_file)


p = Table.read(in_path+"%s"%input)
print ("input file:", input)    

mod = torch.load(mod_file,map_location=DEVICE)    
    
t = generate_newcat(DEVICE, mod, p)
    
tic = time.perf_counter()
    
t.write(out_path+"%s_calib.fits"%input.split('.fits')[0], overwrite=True)

toc = time.perf_counter()
    
print(f"Saved calibrated {run } in {toc - tic:0.4f} seconds as output/{input.split('.fits')[0]}_calib.fits")

    