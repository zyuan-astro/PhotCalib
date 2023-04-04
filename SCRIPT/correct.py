import os
import sys
import glob

from math import *

import numpy as np

import torch

from deform_norm import *

# run M1
DEVICE = torch.device("cpu")

# run CUDA on GPU servers
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print (DEVICE)

# print (torch.cuda.get_device_name(0))



def add_correction(t, model):
    
    xn = np.array(t['Xg']/19000, dtype=np.float64)
    yn = np.array(t['Yg']/19000, dtype=np.float64)

    zn = np.array(t['CaHK_uncalib'])
    zn_err = np.array(t['d_CaHK'])

    
    x = torch.from_numpy(xn).to(DEVICE)
    y = torch.from_numpy(yn).to(DEVICE)
    fs_id = torch.from_numpy(np.zeros(len(xn))).to(torch.int64).to(DEVICE)
    
    

    dz = model(x, y, fs_id).cpu().detach().numpy().T[0]
    zpt = model.zpt.cpu().detach().numpy().T
    dz_zpt =  zpt[0]
    dz_fov = dz - dz_zpt

    z_new = zn-dz
   
    return dz_fov, z_new