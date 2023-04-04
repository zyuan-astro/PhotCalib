import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import numpy as np
sys.path.append("../SCRIPT/")
from correct import *

# run M1
DEVICE = torch.device("cpu")

# run CUDA on GPU servers
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print (DEVICE)

# print (torch.cuda.get_device_name(0))



in_path = "../INPUT/"

mod_path = "../MODEL/"

def Generate_NewCat(run, p):
     
    p['run'] = run

    mod_files = glob.glob(mod_path+"*%s*.mod"%run) 
    mod_file = max(mod_files, key=os.path.getctime)
    # get the most updated mod file (can be min)
    print (mod_file)
    
    mod = torch.load(mod_file,map_location=DEVICE)    
          
       
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
     
   
