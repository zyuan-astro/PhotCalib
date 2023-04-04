import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from recal_zpt import *
# from model_old import *
from deform_norm import *


import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import * 
# run M1
# DEVICE = torch.device("cpu")
# run CUDA on GPU servers
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print (DEVICE)
print (torch.cuda.get_device_name(0))
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


out_path = "../OUTPUT"
 


def calib_zpt(run, run_mod, inputs):
    
    
    in_mod_path = out_path+"/MOD/RUN/"+run_mod+"/"
    
    out_mod_path = out_path+"/MOD/RUN/"+run+"/"

    out_res_path = out_path+"/RES/RUN/"+run+"/"
    
    list_of_files = glob.glob(in_mod_path+"*step1*.mod")
    mod_file = max(list_of_files, key=os.path.getctime)
    # check the FoV model is the most updated one

    print (mod_file)
    
    model = torch.load(mod_file,map_location=DEVICE)

    xn = inputs[:,0]
    yn = inputs[:,1]
    zn = inputs[:,2]
    zn_err = inputs[:,3]
    z0n = inputs[:,4]
    z0n_err = inputs[:,5]
    fn_nb = inputs[:,6]
    ra = inputs[:,7]
    dec = inputs[:,8]


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
    
    print ("number of stars in the run: %s"%(len(xn)))
    print ("number of stars in the smallest field: %i, and the largest field: %i"%(np.min(num_fs), np.max(num_fs)))


    x = torch.from_numpy(xn).to(DEVICE)
    y = torch.from_numpy(yn).to(DEVICE)
    z = torch.unsqueeze(torch.from_numpy(zn), dim=1).to(DEVICE)
    z_err = torch.unsqueeze(torch.from_numpy(zn_err), dim=1).to(DEVICE)
    z0 = torch.unsqueeze(torch.from_numpy(z0n), dim=1).to(DEVICE)
    z0_err = torch.unsqueeze(torch.from_numpy(z0n_err), dim=1).to(DEVICE)
    fs_id = torch.from_numpy(np.zeros(len(xn))).to(torch.int64).to(DEVICE)
    f_id = torch.from_numpy(fn_id).to(torch.int64).to(DEVICE)
    nf = torch.tensor(num_fields).to(DEVICE)

    batch_size = np.max(num_fs) # will probably change lr accordingly
    
    x_batches = []
    y_batches = []
    z_batches = []
    z_err_batches = []
    z0_batches = []
    z0_err_batches = []
    f_id_batches = []
    fs_id_batches = []
    flag_batches = []

    num_batches = 0
    k = 0
    id_batches = []
    for i in range(len(fn_nb_list)):
    
        ind_f = np.in1d(fn_nb, fn_nb_list[i])
    
        x_fields = x[ind_f]
        y_fields = y[ind_f]
        z_fields = z[ind_f]
        z_err_fields = z_err[ind_f]
        z0_fields = z0[ind_f]
        z0_err_fields = z0_err[ind_f]
    
        f_id_fields = f_id[ind_f]
        fs_id_fields = fs_id[ind_f]
        ns_fields = len(f_id_fields)
    
    
               
        x_batches.append(x_fields)
        y_batches.append(y_fields)
        z_batches.append(z_fields)
        z_err_batches.append(z_err_fields)
        z0_batches.append(z0_fields)
        z0_err_batches.append(z0_err_fields)
        f_id_batches.append(f_id_fields)
        fs_id_batches.append(fs_id_fields)
        

        id_batches.append(k)
        
        
        k+= 1
           
    num_batches = k


    id_batches = np.array(id_batches)
    print ("number of fields: %s"%num_fields)
    print ("number of batches: %s"%num_batches)
    
 
    lr = 1e-5

    model_batch = get_zpt(nf).to(DEVICE)
    momentum=0.9
    optimizer_batch = torch.optim.SGD(model_batch.parameters(), lr=lr, momentum=momentum) 
    scheduler = ReduceLROnPlateau(optimizer_batch, mode='min',
            factor=0.5, patience=20, threshold=1e-2, threshold_mode='rel')


    history = [] # monitoring
    n_epoch = 0

    idx_batch = np.arange(num_batches)
    rng = np.random.default_rng()



    N_epochs = 100
    for t in range(N_epochs):

        train_loss = 0.0
    
        rng.shuffle(idx_batch) 
        
    
        for i in idx_batch:
        
            if len(x_batches[i]) > 1:
                
                
        
                dz_batches = model(x_batches[i], y_batches[i], fs_id_batches[i])
                             
                dz_fov_batches = dz_batches - model.zpt[fs_id_batches[i]]
                # print (dz_fov_batches.shape)
            
                zpt_batches = model_batch(f_id_batches[i])
                # print (dz_fov_batches.shape, zpt_batches.shape)

                
                sigma_s = torch.square(z_err_batches[i])+torch.square(z0_err_batches[i])
                
                loss = torch.mean(torch.square(z_batches[i]-dz_fov_batches-zpt_batches-z0_batches[i])/sigma_s)
                optimizer_batch.zero_grad() 
                loss.backward() 
                optimizer_batch.step() 
                train_loss += loss.item()
                

            
        scheduler.step(train_loss/num_batches)
        history += [[n_epoch, train_loss/num_batches]]       
        
        if t % 10 == 0:
            print ('{} {:.7f}'.format(n_epoch, train_loss/num_batches), "%1e"%optimizer_batch.param_groups[0]['lr'])
  
        n_epoch += 1
    

    print (n_epoch)
    

    params = "_err_0.1_%s_%s_zpt_%i_%i_%s_%s"%(run, run_mod, batch_size, n_epoch, lr, momentum)


    history_pd = pd.DataFrame(history,columns=['Epoch','trainLoss'])

    np.savetxt(out_mod_path+"history_model_calib%s.txt"%params,history_pd)

    figure()

    plot(1 + history_pd['Epoch'], history_pd['trainLoss'],'.-',label=r'train loss')


    xlim(1,n_epoch)
    # ylim(1e-3, 2e-2)

    xscale('log')
    yscale('log')
    savefig(out_mod_path+"history_model_calib%s.png"%params)

    torch.save(model_batch, out_mod_path+"model_calib%s.mod"%params)


        
    
    # Calibration plots
    
    dz0 = model(x, y, fs_id).cpu().detach().numpy().T[0]
    dz_zpt = model_batch(f_id).cpu().detach().numpy().T[0]
    
    zpt = model.zpt.cpu().detach().numpy().T
    dz0_zpt = zpt[0]
    dz_fov = dz0-dz0_zpt

    dz = dz_fov + dz_zpt
    z_new = zn-dz


    inputs1 = np.hstack((inputs, fn_id[:,None], dz_zpt[:,None], dz_fov[:,None], z_new[:,None]))
    print (inputs1.shape)
    np.save(out_res_path+"outputs_calib%s.npy"%params, inputs1)
    
    
    return()



    