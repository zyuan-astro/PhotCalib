import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import numpy as np
# sys.path.append("../SCRIPT/")
# from correct import *
import torch
torch.set_default_dtype(torch.float32)



def add_correction(DEVICE, t, model):
    
    xn = np.array(t['Xg']/19000)
    yn = np.array(t['Yg']/19000)

    zn = np.array(t['CaHK'])
    zn_err = np.array(t['d_CaHK'])

    
    x = torch.from_numpy(xn).to(DEVICE)
    y = torch.from_numpy(yn).to(DEVICE)
    fs_id = torch.from_numpy(np.zeros(len(xn))).to(torch.int64).to(DEVICE)
    
    print (x.dtype, y.dtype, fs_id.dtype)

    dz = model(x, y, fs_id).cpu().detach().numpy().T[0]

    zpt = model.zpt.cpu().detach().numpy().T
    dz_zpt =  zpt[0]
    dz_fov = dz - dz_zpt

    z_new = zn-dz
   
    return dz_fov, z_new
    
def generate_newcat(DEVICE, mod, p):
     
  
    fn_nb = p['image_nb']
    fn_nb_list = np.unique(fn_nb)

    # p['run'] = run
    

    dz_fov = np.array([0])
    z_new = np.array([0])
        
    tic = time.perf_counter()
    
    for i in range(len(fn_nb_list)):

        ind = fn_nb == fn_nb_list[i]
        batch_p = p[ind]   
        batch_dz_fov, batch_z_new = add_correction(DEVICE, batch_p, mod)
        
        dz_fov = np.append(dz_fov, batch_dz_fov)
        z_new = np.append(z_new, batch_z_new)
        
        
    toc = time.perf_counter()
    
    print(f"Apply the model in {toc - tic:0.4f} seconds")   
    
    
    dz_fov = dz_fov[1:]
    z_new = z_new[1:]
    
    p["FOV_corr"] = -dz_fov
    p["CaHK_calib"] = z_new
       
    
    return p
     
   
