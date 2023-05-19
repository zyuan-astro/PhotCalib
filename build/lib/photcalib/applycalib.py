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

# run M1
DEVICE = torch.device("cpu")

# run CUDA on GPU servers
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print (DEVICE)

# print (torch.cuda.get_device_name(0))



# in_path = "../data/raw/"

# mod_path = "model/"
def add_correction(t, model):
    
    xn = np.array(t['Xg']/19000, dtype=np.float64)
    yn = np.array(t['Yg']/19000, dtype=np.float64)

    zn = np.array(t['CaHK'])
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
    
def generate_newcat(mod, p):
     
    # p['run'] = run

    # mod_files = glob.glob(mod_path+"*%s*.mod"%run) 
    # mod_file = max(mod_files, key=os.path.getctime)
    # # get the most updated mod file (can be min)
    # print (mod_file)
    
    # mod = torch.load(mod_file,map_location=DEVICE)    
          
       
    batch_size = 1000
    num_batches = int(len(p) / batch_size)
    
    dz_fov = np.array([0])
    z_new = np.array([0])
    
    tic = time.perf_counter()
    
    for batch in range(num_batches):
        
        batch_p = p[batch*batch_size:(batch+1)*batch_size]   
        batch_dz_fov, batch_z_new = add_correction(batch_p, mod)
        
        
        dz_fov = np.append(dz_fov, batch_dz_fov)
        z_new = np.append(z_new, batch_z_new)
        
    if len(p) - (batch+1)*batch_size > 0:
        
        batch_p = p[(batch+1)*batch_size:] 
        batch_dz_fov, batch_z_new = add_correction(batch_p, mod)
        
        
        dz_fov = np.append(dz_fov, batch_dz_fov)
        z_new = np.append(z_new, batch_z_new)
        
        
    toc = time.perf_counter()
    
    print(f"Apply the model in {toc - tic:0.4f} seconds")   
    
    
    dz_fov = dz_fov[1:]
    z_new = z_new[1:]
    
    p["FOV_corr"] = -dz_fov
    p["CaHK_calib"] = z_new
       
    
    return p
     
   
