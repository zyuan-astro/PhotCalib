import os
import sys
import glob

from math import *
import pandas as pd
from matplotlib.pyplot import *
import numpy as np
from astropy.table import Table
import torch

from photcalib import generate_newcat, deform, argparse_apply_model


in_path = "data/"
mod_path = "model/"
out_path = "output/"

args =  argparse_apply_model()

run_mod = args.run_mod
run = args.run_calib

if args.device != None:

    if args.device == 'gpu':
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    else:
        DEVICE = torch.device(args.device)

    print ("DEVICE", DEVICE)

else:

    DEVICE = torch.device("cpu")

    print ("DEVICE", DEVICE)




mod_files = glob.glob(mod_path+"*%s*.mod"%run_mod) 

mod_file = max(mod_files, key=os.path.getctime)
print ("run model:", mod_file)


# p = Table.read(in_path+"%s"%input)
p = pd.read_csv(in_path+"combined_catalogue_%s"%run, sep=r'\s+')
p= p.rename(columns={"#RA": "RA", 'CaHK': 'CaHK_uncalib'})
print ("run calib:", run)    

fn_nb = p['image_nb']
fn_nb_list = np.unique(fn_nb)
fn_id_list = np.arange(len(fn_nb_list))

fn_id = np.empty(len(p), dtype=np.int32)
fn_id.fill(-1)

for i in range(len(fn_nb_list)):
    
    ind_f = np.in1d(p['image_nb'], fn_nb_list[i])
    fn_id[ind_f] = fn_id_list[i]
  

p['image_runid'] = fn_id
p['run'] = run

mod = torch.load(mod_file,map_location=DEVICE)    
    
t = generate_newcat(DEVICE, mod, p, run)
    
tic = time.perf_counter()
    
t.to_csv(out_path+"%s.csv"%run, index=False)

toc = time.perf_counter()
    
print(f"Saved calibrated {run } in {toc - tic:0.4f} seconds as output/{run}.csv")

    