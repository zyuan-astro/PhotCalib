import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
sys.path.append("../src/")

from deform_norm import *
from model_old import *


import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import * 

# run M1
# DEVICE = torch.device("cpu")
# run CUDA on GPU servers
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (DEVICE)
print (torch.cuda.get_device_name(0))


run = "15Am02"


in_path = "../data/input/"
out_path = "../output/"


# files = os.listdir(in_path)
# files = glob.glob(in_path+'*.npy', recursive = True)
# run_list = []
# for file in files:
    
#     if file.endswith('.npy'):
        
#         run_list.append(file.split("_")[1].split(".npy")[0])
    
# run=run_list[run_index]

# print (run_list)

print (run)

out_mod_path = out_path+"/MOD/RUN/"+run+"/"

out_res_path = out_path+"/RES/RUN/"+run+"/"


if not os.path.exists(out_mod_path):

    os.makedirs(out_mod_path)    
    
if not os.path.exists(out_res_path):

    os.makedirs(out_res_path)   
    
inputs = np.load(in_path+'inputs_%s.npy'%run)
print (inputs.shape)




xn = inputs[:,0]
yn = inputs[:,1]
zn = inputs[:,2]
z0n = inputs[:,4]
zn_old = inputs[:,6]
fn_nb = inputs[:,7]
ra = inputs[:,8]
dec = inputs[:,9]

z0n_err = inputs[:,5]

ind_valid = z0n_err <= 0.015


fn_nb_list = np.unique(fn_nb)
num_fields = len(fn_nb_list)

fn_id_list = np.arange(len(fn_nb_list))
fn_id = np.empty(len(fn_nb), dtype=int)

for i in range(len(fn_nb)):
    
    ind_f = np.in1d(fn_nb_list, fn_nb[i])

    fn_id[i]= fn_id_list[ind_f]

num_fs = np.empty(len(fn_nb_list))
for i in range(len(fn_nb_list)):
    
    ind_fs = np.in1d(fn_nb, fn_nb_list[i])
    num_fs[i] = len(fn_nb[ind_fs])
#     print (fn_nb_list[i], num_fs[i])
    

print ("number of stars in the smallest field: %s, and the largest field: %s"%(np.min(num_fs), np.max(num_fs)))


x = torch.from_numpy(xn).to(DEVICE)
y = torch.from_numpy(yn).to(DEVICE)
z = torch.unsqueeze(torch.from_numpy(zn), dim=1).to(DEVICE)
z0 = torch.unsqueeze(torch.from_numpy(z0n), dim=1).to(DEVICE)
f_id = torch.from_numpy(fn_id).to(torch.int64).to(DEVICE)
nf = torch.tensor(num_fields).to(DEVICE)


batch_size = np.max(num_fs) # will probably change lr accordingly
# Divide batches in each field

x_batches = []
y_batches = []
z_batches = []
z0_batches = []
f_id_batches = []
flag_batches = []

num_batches = 0
k = 0
id_batches = []
for i in range(len(fn_nb_list)):
    
    # first choose all the stars in a field
    ind_f = np.in1d(fn_nb, fn_nb_list[i])
    
    x_fields = x[ind_f]
    y_fields = y[ind_f]
    z_fields = z[ind_f]
    z0_fields = z0[ind_f]
    
    # f_id is the psudo field id in the field list
    f_id_fields = f_id[ind_f]
    # record the number of stars in each field
    ns_fields = len(f_id_fields)
    
    
    # if the number of stars in a field is smaller than the batch_size, put them into one batch
               
    x_batches.append(x_fields)
    y_batches.append(y_fields)
    z_batches.append(z_fields)
    z0_batches.append(z0_fields)
    f_id_batches.append(f_id_fields)
        

    id_batches.append(k)
        
        
    k+= 1
   
   

        
num_batches = k


id_batches = np.array(id_batches)
print ("number of fields: %s"%num_fields)

print ("number of batches: %s"%num_batches)


lr = 1e-3

model_batch = deform(2, [200, 200, 200, 200], 1, nf).to(DEVICE)
momentum=0.9
optimizer_batch = torch.optim.SGD(model_batch.parameters(), lr=lr, momentum=momentum) 
scheduler = ReduceLROnPlateau(optimizer_batch, mode='min',
            factor=0.5, patience=20, threshold=1e-5, threshold_mode='abs')


history = [] # monitoring
n_epoch = 0

idx_batch = np.arange(num_batches)
rng = np.random.default_rng()



N_epochs = 400
for t in range(N_epochs):

    train_loss = 0.0
    
    rng.shuffle(idx_batch) 
    
    for i in idx_batch:
        
        if len(x_batches[i]) > 1:
        
            dz_batches = model_batch(x_batches[i], y_batches[i], f_id_batches[i])
            loss = criterion(z_batches[i]-dz_batches, z0_batches[i])  

            optimizer_batch.zero_grad() 
            loss.backward() 
            optimizer_batch.step() 
            train_loss += loss.item()

    # break    
    scheduler.step(train_loss/num_batches)
    history += [[n_epoch, train_loss/num_batches]]        
    
    if t % 10 == 0:
        print ('{} {:.7f}'.format(n_epoch, train_loss/num_batches), "%1e"%optimizer_batch.param_groups[0]['lr'])
        
#     torch.save(model_batch, out_path+"model_norm_calib_%s_0.05_old_%s_%s_%s_%s"
#            %(run_list[0], batch_size, n_epoch, lr, momentum))
    
    n_epoch += 1
    
    


print (n_epoch)
params = "_%s_%i_%i_%s_%s"%(run, batch_size, n_epoch, lr, momentum)


history_pd = pd.DataFrame(history,columns=['Epoch','trainLoss'])

np.savetxt(out_mod_path+"history_model_calib%s.txt"%params,history_pd)

figure()

plot(1 + history_pd['Epoch'], history_pd['trainLoss'],'.-',label=r'train loss')


xlim(1,n_epoch)
ylim(1e-3, 2e-2)

xscale('log')
yscale('log')
savefig(out_mod_path+"history_model_calib%s.png"%params)

torch.save(model_batch, out_mod_path+"model_calib%s.mod"%params)


# results of the calibration sample
dz = model_batch(x, y, f_id).cpu().detach().numpy().T[0]
zpt = model_batch.zpt.cpu().detach().numpy().T

dz_zpt = zpt[fn_id]
dz_fov = dz - dz_zpt
z_new = zn-dz




inputs1 = np.hstack((inputs, fn_id[:,None], dz_zpt[:,None], dz_fov[:,None], z_new[:,None]))
print (inputs1.shape)
np.save(out_res_path+"outputs_calib%s.npy"%params, inputs1)

