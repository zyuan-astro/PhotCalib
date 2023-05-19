import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import numpy as np
import torch

from photcalib import TrainingModule, make_diagnostic_plots, argparse_train_model

in_path = "data/"
mod_path = "model/"
out_path = "output/"

args =  argparse_train_model()

if args.device != None:

    if args.device == 'gpu':
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    else:
        DEVICE = torch.device(args.device)

    print (DEVICE)

else:

    DEVICE = torch.device("cpu")

    print (DEVICE)
    
run = args.run
inputs = np.float32(np.load(in_path+'inputs_%s.npy'%run))

print ("calibrate run %s"%run)

if args.lr != None:
    lr = args.lr
else:
    lr = 1e-6
    
if args.N_epochs != None:
    N_epochs = args.N_epochs
else:
    N_epochs = 400

if args.momentum != None:
    momentum = args.momentum
else:
    momentum = 0.9

if args.thr != None:
    thr = args.thr
else:
    thr = 1e-2

print ("learning rate:", lr, "N_epochs:", N_epochs, "momentum:", momentum, "threshold:", thr)
mod = TrainingModule(lr, N_epochs, momentum, thr)
mod.train_model(DEVICE, run, inputs)
    
print ("calibration done")
    
make_diagnostic_plots(DEVICE, run)
    
print ("plots done")
