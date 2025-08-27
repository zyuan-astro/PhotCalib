import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import numpy as np
import torch
torch.set_default_dtype(torch.float32)



def add_correction(DEVICE, t, model):
    
    xn = np.array(t['Xg']/19000)
    yn = np.array(t['Yg']/19000)

    zn = np.array(t['CaHK_uncalib'])
    zn_err = np.array(t['d_CaHK'])
    fn_id = np.array(t['image_runid'])

    
    x = torch.from_numpy(xn).to(torch.float32).to(DEVICE)
    y = torch.from_numpy(yn).to(torch.float32).to(DEVICE)
    f_id = torch.from_numpy(fn_id).to(torch.int32).to(DEVICE)
    
    dz = model(x, y, f_id).cpu().detach().numpy().T[0]
    zpt = model.zpt.cpu().detach().numpy().T
    dz_zpt =  zpt[fn_id]
    dz_fov = dz - zpt[fn_id]
    z_new = zn-dz
   
    return dz_zpt, dz_fov, z_new
    
def generate_newcat(DEVICE, mod, p, run):
      
    
    fn_id = p['image_runid']
    fn_id_list = np.unique(fn_id)

    dz_zpt = np.array([0])
    dz_fov = np.array([0])
    z_new = np.array([0])
        
    tic = time.perf_counter()
    
    for i in range(len(fn_id_list)):

        ind = fn_id == fn_id_list[i]
        batch_p = p[ind]   
        batch_dz_zpt, batch_dz_fov, batch_z_new = add_correction(DEVICE, batch_p, mod)
        
        dz_zpt = np.append(dz_zpt, batch_dz_zpt)
        dz_fov = np.append(dz_fov, batch_dz_fov)
        z_new = np.append(z_new, batch_z_new)
        
   
    toc = time.perf_counter()
    
    print(f"Apply the model in {toc - tic:0.4f} seconds")   
    
    
    dz_zpt = dz_zpt[1:]
    dz_fov = dz_fov[1:]
    z_new = z_new[1:]
    
    p["ZPT_corr"] = -dz_zpt
    p["FOV_corr"] = -dz_fov
    p["CaHK_calib"] = z_new
        
    return p
     
   
