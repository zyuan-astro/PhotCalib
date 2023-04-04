import os
import sys
import glob

from math import *

import numpy as np

import torch

from deform_norm import *

# run M1
DEVICE = torch.device("cpu")






def add_correction(t):
    
    xn = np.array(t['Xg']/19000, dtype=np.float64)
    yn = np.array(t['Yg']/19000, dtype=np.float64)

    zn = np.array(t['CaHK_uncalib'])
    zn_err = np.array(t['d_CaHK'])


    fn_id = np.array(t['image_runid'])

    ra = np.float64(t['RA'])
    dec = np.float64(t['Dec'])
    
    x = torch.from_numpy(xn).to(DEVICE)
    y = torch.from_numpy(yn).to(DEVICE)
    f_id = torch.from_numpy(fn_id).to(torch.int64).to(DEVICE)
    
    tic = time.perf_counter()

    dz = model(x, y, f_id).cpu().detach().numpy().T[0]
    zpt = model.zpt.cpu().detach().numpy().T
    dz_zpt =  zpt[fn_id]
    dz_fov = dz - zpt[fn_id]

    z_new = zn-dz

    toc = time.perf_counter()
    print(f"Apply the model in {toc - tic:0.4f} seconds")
    
    
    return dz_zpt, dz_fov, z_new