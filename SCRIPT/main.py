import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import numpy as np
sys.path.append("../SCRIPT/")

from calib import *

# from makeplots import *
# from model_old import *




in_path = "../INPUT/"
mod_path = "../MODEL/"
out_path = "../OUTPUT/"


files = glob.glob(in_path+"combined_catalogue*", recursive = True)
files.sort()

run_list = []
for file in files:
    
    run_list.append(file.split("_")[2])
        
run_list = np.array(run_list)


print (run_list)


for i in range(len(run_list)):
    
    run = run_list[i]
    p = pd.read_csv(in_path+"combined_catalogue_%s_sel"%run, delim_whitespace=True)
    p= p.rename(columns={"#RA": "RA", 'CaHK': 'CaHK_uncalib'})
    
    t = Generate_NewCat(run, p)
    
    tic = time.perf_counter()
    
    t.to_csv(out_path+"%s.csv"%run, index=False)
    
    toc = time.perf_counter()
    
    print(f"Saved {run } in {toc - tic:0.4f} seconds")

    