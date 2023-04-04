import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
sys.path.append("../SCRIPT/")

from training_zpt import *

from makeplots_small import *



from model_old import *


import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import * 





in_path = "../INPUT/RUN/"


files = glob.glob(in_path+"*_err_0.1.npy", recursive = True)
files.sort()

run_list = []
for file in files:
    
    
    run_list.append(file.split("_")[-3].split(".npy")[0])
        
# print (run_list)
# run_small_list = ["22Am04"]
# run_model_list = ["22Am05"]


run_small_list = ["19Bm01"]
run_model_list = ["19Am06"]

print (f"Calibrate {run_small_list[0]} with the model of {run_model_list[0]} ")



i = 0
for run in run_small_list:
    
       
    print (i, run)
    inputs = np.load(in_path+'inputs_%s_err_0.1.npy'%run)

    run_model = run_model_list[i]
    calib_zpt(run, run_model, inputs)
    makeplots(run, run_model,inputs)
    
    i += 1