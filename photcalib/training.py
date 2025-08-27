import os
import sys
import glob
from math import *
import pandas as pd
import numpy as np

from matplotlib.pyplot import *
from .deform_norm import deform



import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import * 

torch.set_default_dtype(torch.float32)

in_path = "data/"
mod_path = "model/"
out_path = "output/" 


class TrainingModule():

    def __init__(self,  lr = 1e-6, N_epochs=400, momentum = 0.9, thr = 1e-2):

         self.lr = lr      
         self.N_epochs = N_epochs
         self.momentum = momentum
         self.thr = thr
        
    def train_model(self, DEVICE, run, inputs):
        
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
        f_id = torch.from_numpy(fn_id).to(torch.int32).to(DEVICE)
        nf = torch.tensor(num_fields).to(DEVICE)

        batch_size = np.max(num_fs) 
    
        x_batches = []
        y_batches = []
        z_batches = []
        z_err_batches = []
        z0_batches = []
        z0_err_batches = []
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
            z_err_fields = z_err[ind_f]
            z0_fields = z0[ind_f]
            z0_err_fields = z0_err[ind_f]
    
            # f_id is the psudo field id in the field list
            f_id_fields = f_id[ind_f]
            # record the number of stars in each field
            ns_fields = len(f_id_fields)
    
    
            # if the number of stars in a field is smaller than the batch_size, put them into one batch
               
            x_batches.append(x_fields)
            y_batches.append(y_fields)
            z_batches.append(z_fields)
            z_err_batches.append(z_err_fields)
            z0_batches.append(z0_fields)
            z0_err_batches.append(z0_err_fields)
            f_id_batches.append(f_id_fields)
        

            id_batches.append(k)
        
        
            k+= 1
           
        num_batches = k


        id_batches = np.array(id_batches)
        print ("number of fields: %s"%num_fields)
        print ("number of batches: %s"%num_batches)
    
 
        

        model_batch = deform(2, [200, 200, 200, 200], 1, nf).to(DEVICE)
        optimizer_batch = torch.optim.SGD(model_batch.parameters(), lr=self.lr, momentum=self.momentum) 
        scheduler = ReduceLROnPlateau(optimizer_batch, mode='min',
            factor=0.5, patience=20, threshold=self.thr, threshold_mode='rel')


        history = [] # monitoring
        n_epoch = 0

        idx_batch = np.arange(num_batches)
        rng = np.random.default_rng()



    
        for t in range(self.N_epochs):

            train_loss = 0.0
    
            rng.shuffle(idx_batch) 
        
    
            for i in idx_batch:
        
                if len(x_batches[i]) > 1:
        
                    dz_batches = model_batch(x_batches[i], y_batches[i], f_id_batches[i])                            
        
                    sigma_s = torch.square(z_err_batches[i])+torch.square(z0_err_batches[i])
                
                    loss = torch.mean(torch.square(z_batches[i]-dz_batches-z0_batches[i])/sigma_s)
                    optimizer_batch.zero_grad() 
                    loss.backward() 
                    optimizer_batch.step() 
                    train_loss += loss.item()

            

            # break
            
            scheduler.step(train_loss/num_batches)
            history += [[n_epoch, train_loss/num_batches]]       
        
            if t % 10 == 0:
                print ('{} {:.7f}'.format(n_epoch, train_loss/num_batches), "%1e"%optimizer_batch.param_groups[0]['lr'])
  
            n_epoch += 1
    

        print (n_epoch)
    

        params = "_%s_%i_%i_%s_%s_%s"%(run, batch_size, n_epoch, self.lr, self.momentum, self.thr)


        history_pd = pd.DataFrame(history,columns=['Epoch','trainLoss'])

        np.savetxt(mod_path+"history_model_calib%s.txt"%params,history_pd)

        figure()

        plot(1 + history_pd['Epoch'], history_pd['trainLoss'],'.-',label=r'train loss')


        xlim(1,n_epoch)
        # ylim(1e-3, 2e-2)

        xscale('log')
        yscale('log')
        savefig(mod_path+"history_model_calib%s.png"%params)

        torch.save(model_batch, mod_path+"model_calib%s.mod"%params)


        # results of the calibration sample
        dz = model_batch(x, y, f_id).cpu().detach().numpy().T[0]
        zpt = model_batch.zpt.cpu().detach().numpy().T

        dz_zpt = zpt[fn_id]
        dz_fov = dz - dz_zpt
        z_new = zn-dz




        inputs1 = np.hstack((inputs, fn_id[:,None], dz_zpt[:,None], dz_fov[:,None], z_new[:,None]))
        print (inputs1.shape)
        np.save(out_path+"outputs_calib%s.npy"%params, np.float32(inputs1))
    
    
        return()



    