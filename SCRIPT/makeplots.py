import os
import sys
import glob

from math import *
from matplotlib.pyplot import *
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from deform import *
from model_old import *


import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import * 

# run M1
DEVICE = torch.device("cpu")
# run CUDA on GPU servers
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print (DEVICE)

# print (torch.cuda.get_device_name(0))


in_path = "../INPUT/"
out_path = "../OUTPUT"


batch2_list =  ["21Am01", "21Am02"]
batch3_list = ["21Am03", "21Am04", "21Am05", "21Am06", "21Bm01", "21Bm02", "21Bm03", "21Bm04", "21Bm06", "22Am01", "22Am02", "22Am03"]
batch4_list = ['17Bm02']
batch5_list = ['22Am04', '22Am05', '22Am06', '22Am07', '22Bm01', '22Bm02']

def makeplots(run, run_mod, inputs):
    
    inputs = np.load(in_path+'inputs_%s_err_0.1.npy'%run)

    in_mod_path = out_path+"/MOD/RUN/"+run_mod+"/"
    out_mod_path = out_path+"/MOD/RUN/"+run+"/"

    # out_res_path = out_path+"/RES/RUN/"+run+"/"

    out_fig_path = out_path+"/FIG/RUN/"+run+"/"


    semester = run.split("m")[0]
    
#     if not os.path.exists(out_fig_path):

#         os.makedirs(out_fig_path)    
    

    list_of_files = glob.glob(in_mod_path+"*err*.mod") # * means all if need specific format then *.csv
    mod_fov_file = max(list_of_files, key=os.path.getctime)

    print (mod_fov_file)
    
    model_fov = torch.load(mod_fov_file,map_location=DEVICE)
    
    
    list_of_files = glob.glob(out_mod_path+"*zpt*.mod") # * means all if need specific format then *.csv
    mod_zpt_file = max(list_of_files, key=os.path.getctime)

    print (mod_zpt_file)
    
    model_zpt = torch.load(mod_zpt_file,map_location=DEVICE)
            
    params = mod_zpt_file.split("calib")[1].split(".mod")[0]
    print (params)


    xn = inputs[:,0]
    yn = inputs[:,1]
    zn = inputs[:,2]
    zn_err = inputs[:,3]
    z0n = inputs[:,4]
    z0n_err = inputs[:,5]
    fn_nb = inputs[:,6]
    ra = inputs[:,7]
    dec = inputs[:,8]

    

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
    

    batch = semester
    
    if run in batch2_list:
        
        batch = "batch2"
        
        
    if run in batch3_list:
        
        batch = "batch3"
        
    if run in batch4_list:
        
        batch = "batch4"
        
    if run in batch5_list:
        
        batch = "batch5"
    
    
    fov_old = model_old(batch)
    dz_fov_old=fov_old.calib(xn,yn)
    z1_old = zn - dz_fov_old



    x = torch.from_numpy(xn).to(DEVICE)
    y = torch.from_numpy(yn).to(DEVICE)
    z = torch.unsqueeze(torch.from_numpy(zn), dim=1).to(DEVICE)
    z0 = torch.unsqueeze(torch.from_numpy(z0n), dim=1).to(DEVICE)
    fs_id = torch.zeros(len(x)).to(torch.int64).to(DEVICE)

    f_id = torch.from_numpy(fn_id).to(torch.int64).to(DEVICE)
    nf = torch.tensor(num_fields).to(DEVICE)



    # Calibration plots
    
    dz0 = model_fov(x, y, fs_id).cpu().detach().numpy().T[0]
    dz_zpt = model_zpt(f_id).cpu().detach().numpy().T[0]
    
    zpt = model_fov.zpt.cpu().detach().numpy().T
    dz0_zpt = zpt[0]
    dz_fov = dz0-dz0_zpt

    
    dz = dz_fov + dz_zpt
    dz1 = zn - z0n - dz_zpt
    z_new = zn-dz

    r_new = zn-z0n-dz
    r_raw = zn-z0n
    
    
    zpt_old = np.zeros(len(fn_nb_list))
    z_old = np.zeros(len(z))

    for i in range(len(fn_nb_list)):
    
        ind_f = np.in1d(fn_nb, fn_nb_list[i])
        
        z1_old_f = z1_old[ind_f]
        
        dz1_old_f = z1_old_f-z0n[ind_f]
    
        z_low = np.percentile(dz1_old_f, 2.5)
        z_upp = np.percentile(dz1_old_f, 97.5)
        
        ind = (dz1_old_f >= z_low)*(dz1_old_f <= z_upp)
    
    
        zpt_old[i] = np.mean(dz1_old_f[ind])
        
        z_old[ind_f] = z1_old_f - zpt_old[i]    
        
    
    r_old = z_old - z0n

    n_bin = 50
    # bin_size = 500
    H, xedges, yedges = np.histogram2d(xn, yn, bins=[n_bin, n_bin], range=[[-.12, 1.12], [0, 1]])
    fov_grid = np.zeros([n_bin, n_bin])
    dz1_grid = np.zeros([n_bin, n_bin])
    dz_old_grid = np.zeros([n_bin, n_bin])



    r_old_calib_grid = np.zeros([n_bin, n_bin])
    r_new_calib_grid = np.zeros([n_bin, n_bin])
    
    chi_old_calib_grid = np.zeros([n_bin, n_bin])
    chi_new_calib_grid = np.zeros([n_bin, n_bin])

    for i in range(n_bin):
    
        ind_x = (xn >= xedges[i])*(xn < xedges[i+1])
    
        for j in range(n_bin):
         
            ind_y = (yn >= yedges[j])*(yn < yedges[j+1])
        
            if len(xn[ind_x*ind_y])>0:        
            
                dz1_grid[i,j] = np.mean(dz1[ind_x*ind_y])
            
                dz_old_grid[i,j] = np.mean(dz_fov_old[ind_x*ind_y])
        
        
                fov_grid[i,j] = np.mean(dz_fov[ind_x*ind_y])
            
                r_old_calib_grid[i,j] = np.mean(r_old[ind_x*ind_y])
        
                r_new_calib_grid[i,j] = np.mean(r_new[ind_x*ind_y])
            
                chi_old_calib_grid[i,j] = np.mean(np.square(r_old[ind_x*ind_y]))
                chi_new_calib_grid[i,j] = np.mean(np.square(r_new[ind_x*ind_y]))
 
      
            if j in [12, 38]:
        
                r_old_calib_grid[i,j] = 0.0
                r_new_calib_grid[i,j] = 0.0
                
                chi_old_calib_grid[i,j] = 0.0
                chi_new_calib_grid[i,j] = 0.0

    xm_grid = np.linspace(-0.12, 1.12, 124)
    ym_grid = np.linspace(0., 1., 100)
    
    xm = np.zeros(1)
    ym = np.zeros(1)

    for i in range(len(xm_grid)):
    
        xm_ = np.zeros(100)
        xm_.fill(xm_grid[i])
    
        xm = np.append(xm, xm_)
        ym = np.append(ym, ym_grid)
        
    xm = xm[1:]
    ym = ym[1:]
    
    fm_id = np.zeros(len(xm), dtype=np.int64)
    
    dz_m_old = fov_old.calib(xm,ym)
    
    x1 = torch.from_numpy(xm).to(DEVICE)
    y1 = torch.from_numpy(ym).to(DEVICE)

    f1_id = torch.from_numpy(fm_id).to(torch.int64).to(DEVICE)
    
    dz_m_new = (model_fov(x1, y1, f1_id)-model_fov.zpt[f1_id]).cpu().detach().numpy().T[0]
   
    
    
    n_bin = 50
    H, xedges, yedges = np.histogram2d(xm, ym, bins=[n_bin, n_bin], range=[[-0.12, 1.12], [0, 1]])
    fov_m_old_grid = np.zeros([n_bin, n_bin])
    fov_m_new_grid = np.zeros([n_bin, n_bin])


    for i in range(n_bin):
    
        ind_x = (xm >= xedges[i])*(xm < xedges[i+1])
    
        for j in range(n_bin):
         
            ind_y = (ym >= yedges[j])*(ym < yedges[j+1])
        
            if len(xm[ind_x*ind_y])>0:        
            
       
                fov_m_new_grid[i,j] = np.mean(dz_m_new[ind_x*ind_y])
                fov_m_old_grid[i,j] = np.mean(dz_m_old[ind_x*ind_y])
            


    
    
    
    
    
   
    ns_f = np.empty(num_fields)


    ra_f = np.array([0])
    dec_f = np.array([0])


    r_new_f = np.array([0])
    r_old_f = np.array([0])

    chi_f = np.array([0])
    chi_old_f = np.array([0])

    for i in range(num_fields):
    
        ind_f = np.in1d(fn_id, fn_id_list[i])
        ind = ind_f
        ns_f[i] = len(fn_id[ind])
   
        if len(fn_id[ind]) > 0:
        
            ra_f = np.append(ra_f, ra[ind])
            dec_f = np.append(dec_f, dec[ind])
        
        
        
            r0 = np.zeros(len(ra[ind]))
            r0.fill(np.mean(r_new[ind]))
            r_new_f = np.append(r_new_f, r0)
        
            r0_old = np.zeros(len(ra[ind]))
            r0_old.fill(np.mean(r_old[ind]))
            r_old_f = np.append(r_old_f, r0_old)
        
            chi0 = np.zeros(len(ra[ind]))
            chi0.fill(sqrt(np.mean(np.square(r_new[ind]))))
            chi_f = np.append(chi_f, chi0)
        
    
            chi0 = np.zeros(len(ra[ind]))
            chi0.fill(sqrt(np.mean(np.square(r_old[ind]))))
            chi_old_f = np.append(chi_old_f, chi0)
        
       
        
    ra_f = ra_f[1:]  
    dec_f = dec_f[1:] 


    r_new_f = r_new_f[1:] 
    r_old_f = r_old_f[1:] 


    chi_f = chi_f[1:] 
    chi_old_f = chi_old_f[1:]   
    
    dz1_grid_1d = dz1_grid.flatten()  

    ind_1d = dz1_grid_1d != 0
    vmin = np.percentile(dz1_grid_1d[ind_1d],2.5)
    vmax = np.percentile(dz1_grid_1d[ind_1d],97.5)
           
    

    figure(figsize = (22,5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.4, hspace = 0.)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])
    ax3 = subplot(gs[0, 2])

    fov_m_old_grid_ = fov_m_old_grid.flatten()  
    fov_m_new_grid_ = fov_m_new_grid.flatten() 

    if semester == "15A":
    
        fov_m_new_ctr = fov_m_new_grid_.max()
#         vmin_old = vmin - vmax
#         vmax_old = 0
#         ticks = np.linspace((vmin - vmax), 0, 4)
    
    else:
        
        fov_m_new_ctr = fov_m_new_grid_.min()
    
#         vmin_old = 0
#         vmax_old = vmax - vmin
#         ticks = np.linspace(0, (vmax - vmin),4)

    
    
    ind_1d = fov_m_old_grid_  != 0
    vmin = np.percentile(fov_m_old_grid_[ind_1d],2.5)
    vmax = np.percentile(fov_m_old_grid_[ind_1d],97.5)
    fig = figure(figsize = (15,5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.05, right= 0.95, bottom = 0.15, top = 0.9, wspace=0.2, hspace = 0.)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])
    ax3 = subplot(gs[0, 2])



    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax1.pcolormesh(xedges,  yedges, fov_m_old_grid.T, cmap='jet',vmin=vmin,vmax=vmax)
    cb1=colorbar(im,cax=cax,orientation="vertical", format="%.2f", ticks=np.linspace(vmin, vmax, 3))
    cax.yaxis.set_ticks_position('right')
    # cb3.set_label('FoV (old)', fontsize=20)
    ax1.set_title("Old Model", fontsize=20)


    fov_m_new_grid_ctr = fov_m_new_grid_.min()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax2.pcolormesh(xedges,  yedges, fov_m_new_grid.T-fov_m_new_ctr, cmap='jet',vmin=vmin,vmax=vmax)
    cb2=colorbar(im,cax=cax,orientation="vertical", format="%.2f", ticks=np.linspace(vmin, vmax, 3))
    cax.yaxis.set_ticks_position('right')
    # cb3.set_label('FoV (old)', fontsize=20)
    ax2.set_title("New Model", fontsize=20)


#     vmin = np.percentile((fov_new_grid_-fov_new_grid_.min() - fov_old_grid_)/fov_old_grid_, 2.5)
#     vmax = np.percentile((fov_new_grid_-fov_new_grid_.min() - fov_old_grid_)/fov_old_grid_, 97.5)

    vmin = -.05
    vmax = .05
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax3.pcolormesh(xedges,  yedges, fov_m_new_grid.T-fov_m_new_ctr-fov_m_old_grid.T, cmap='seismic',vmin=vmin,vmax=vmax)
    cb3=colorbar(im,cax=cax,orientation="vertical", format="%.2f", ticks=np.linspace(vmin, vmax, 3))
    cax.yaxis.set_ticks_position('right')
    # cb3.set_label('FoV (old)', fontsize=20)
    ax3.set_title("New-Old", fontsize=20)
    savefig(out_fig_path+"model_FoV_new-old%s.png"%params, dpi=200)

    
    
    
    figure(figsize = (10, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.15, right= 0.85, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)



    vmin = 0
    vmax = .05
    fig = figure(figsize = (10, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.15, right= 0.85, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)



    im=ax.scatter(z0n, r_new, c=z0n_err, s=10, marker="s", 
              cmap='jet', vmin=vmin, vmax=vmax)
    cb=colorbar(im,cax=cax,orientation="vertical", format="%.2f",ticks=np.linspace(vmin, vmax,4))
    cb.set_label(r'$\delta$CaHK$_{Gaia}}$', fontsize=20)
    cb.ax.tick_params(labelsize=20)
    
    ax.axhline(np.percentile(r_new, 16), zorder=1)
    ax.axhline(np.percentile(r_new, 50), zorder=1)
    ax.axhline(np.percentile(r_new, 84), zorder=1)
    

    ax.set_title("NEW (run:%s)"%run, fontsize=20)
    ax.set_xlabel(r'CaHK$_{\mathrm{Gaia}}$', fontsize=20)
    ax.set_ylabel(r'$\Delta$CaHK=CaHK$_{\mathrm{New}}$-CaHK$_{\mathrm{Gaia}}$', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    savefig(out_fig_path+"residual_gaia_new%s.png"%params, dpi=200)




    vmin = np.percentile(r_raw,2.5)
    vmax = np.percentile(r_raw,97.5)
    fig = figure(figsize = (16, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)


    im = ax.scatter(ra, dec, c=r_raw, s=1, cmap='jet', vmin=vmin, vmax=vmax)

    cb=colorbar(im,cax=cax,orientation="vertical", format="%.1f",ticks=np.linspace(vmin, vmax,4))
    cb.set_label('$\Delta$CaHK (Raw-Gaia)', fontsize=20)
    cb.ax.tick_params(labelsize=20)

    ax.set_title("$\Delta$CaHK=CaHK$_{\mathrm{Raw}}$-CaHK$_{\mathrm{Gaia}}$ (run:%s)"%run, fontsize=20)
    ax.set_xlim(np.max(ra), np.min(ra))
    ax.set_xlabel(r'$\alpha$($^{\circ}$)', fontsize=20)
    ax.set_ylabel(r'$\delta$($^{\circ}$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)


    savefig(out_fig_path+"radec_residual_raw%s.png"%params, dpi=200)





    figure(figsize = (16, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)


    im = ax.scatter(ra, dec,
                c=r_old, s=1, cmap='jet', vmin=-.1, vmax=.1)

    cb=colorbar(im,cax=cax,orientation="vertical", ticks=[-.1, 0, .1])
    cb.set_label('$\Delta$CaHK (Old-Gaia)', fontsize=20)

    ax.set_title("$\Delta$CaHK=CaHK$_{\mathrm{Old}}$-CaHK$_{\mathrm{Gaia}}$ (run:%s)"%run, fontsize=20)
    cb.ax.tick_params(labelsize=20)

    ax.set_xlim(np.max(ra), np.min(ra))
    ax.set_xlabel(r'$\alpha$($^{\circ}$)', fontsize=20)
    ax.set_ylabel(r'$\delta$($^{\circ}$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)

    savefig(out_fig_path+"radec_residual_old%s.png"%params, dpi=200)


    figure(figsize = (16, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)


    im = ax.scatter(ra, dec,
                c=r_new, s=1, cmap='jet', vmin=-.1, vmax=.1)

    cb=colorbar(im,cax=cax,orientation="vertical", ticks=[-.1, 0, .1])
    ax.set_title("$\Delta$CaHK=CaHK$_{\mathrm{New}}$-CaHK$_{\mathrm{Gaia}}$ (run:%s)"%run, fontsize=20)
    cb.ax.tick_params(labelsize=20)
    cb.set_label('$\Delta$CaHK (New-Gaia)', fontsize=20)


    ax.set_xlim(np.max(ra), np.min(ra))
    ax.set_xlabel(r'$\alpha$($^{\circ}$)', fontsize=20)
    ax.set_ylabel(r'$\delta$($^{\circ}$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)

    savefig(out_fig_path+"radec_residual_new%s.png"%params, dpi=200)


    vmin = np.percentile(dz_zpt,2.5)
    vmax = np.percentile(dz_zpt,97.5)
    
    print (vmin, vmax)
    fig = figure(figsize = (16, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax = subplot(gs[0, 0])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)

    if len(dz_zpt) < 1000:
        
        s = 10
        
    elif len(dz_zpt) < 10000:
        
        s = 5
        
    else:
        
        s = 1

    im=ax.scatter(ra, dec, c=dz_zpt, s=s, marker="s", cmap='seismic', vmin=vmin, vmax=vmax)
    cb=colorbar(im,cax=cax,orientation="vertical", format="%.1f",ticks=np.linspace(vmin, vmax,4))
    cb.set_label('zero point offset', fontsize=20)
    cb.ax.tick_params(labelsize=20)

    ax.set_title("NEW (run:%s)"%run, fontsize=20)
    ax.set_xlim(np.max(ra), np.min(ra))
    ax.set_xlabel(r'$\alpha$($^{\circ}$)', fontsize=20)
    ax.set_ylabel(r'$\delta$($^{\circ}$)', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)


    savefig(out_fig_path+"radec_zpt_new%s.png"%params, dpi=200)


    figure(figsize = (13,5))
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.85, wspace=0.5, hspace = 0.2)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax1.scatter(xn, yn, c=r_old,s=1,cmap='jet',vmin=-0.1, vmax=.1)
    cb1=colorbar(im,cax=cax,orientation="vertical", ticks=[-.1,0, .1])
    cb1.set_label('$\Delta$CaHK (old-Gaia)', fontsize=20)
    ax1.text(.8, 1.15, "Calibration Sample ($\delta$ CaHK < 0.1)", fontsize=20)
    ax1.text(.0, 1.15, "run:%s"%run, fontsize=20)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax2.scatter(xn, yn, c=r_new,s=1,cmap='jet',vmin=-0.1, vmax=.1)
    cb2=colorbar(im,cax=cax,orientation="vertical", ticks=[-.1,0, .1])
    cax.yaxis.set_ticks_position('right')


    cb2.set_label('$\Delta$CaHK (new-Gaia)', fontsize=20)

    for cb in [cb1, cb2]:

        cb.ax.tick_params(labelsize=20)

    for ax in [ax1, ax2]:
        ax.set_xticks([.2, .4, .6, .8, 1.])
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.tick_params(axis='both', which='major', labelsize=20)
    
    
    savefig(out_fig_path+"residual_scatter_FOV_new%s.png"%params, dpi=200)


    r_new_m = np.percentile(r_new[ind_valid], 50)
    r_new_u = np.percentile(r_new[ind_valid], 84)
    r_new_l = np.percentile(r_new[ind_valid], 16)



    figure(figsize = (8,6))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.16, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.4, hspace = 0.)
    ax1 = subplot(gs[0, 0])

    ax1.hist(r_raw[ind_valid]-np.median(r_raw[ind_valid]), 
     lw=2, bins = 50, range=(-.2, .3),alpha=0.5, stacked=True,density=True,
     color='gray',label='Raw (shift)') 
    ax1.hist(r_new,bins = 50, lw=2, range=(-.2, .3), stacked=True,density=True,
     histtype='step', hatch="//",color='midnightblue',label='Calib ($\delta$CaHK < 0.1)') 
    ax1.hist(r_new[ind_valid],bins = 50, lw=3, range=(-.2, .3), stacked=True,density=True,
     histtype='step', color='orangered', label='Valid ($\delta$CaHK < 0.015)') 
 
    ax1.set_xlabel('$\Delta$CaHK', fontsize=20)
    ax1.set_ylabel('number function', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax1.axvline(0, ls='dashed', lw=3, color='mediumseagreen')  
    ax1.set_title('run:%s, $\Delta$CaHK=%.3f$^{+%.3f}_{-%.3f}$'%(run, r_new_m, r_new_u-r_new_m, r_new_m-r_new_l), fontsize=20)
    ax1.legend(fontsize=15)

    savefig(out_fig_path+"hist_residual_new%s.png"%params, dpi=200)



    figure(figsize = (8,6))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.16, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.4, hspace = 0.)
    ax1 = subplot(gs[0, 0])
    ax1.hist(r_raw[ind_valid]-np.median(r_raw[ind_valid]), 
     lw=2, bins = 50, range=(-.2, .3), alpha=0.5, color='gray',
     label='Raw (shift)') 
    ax1.hist(r_old[ind_valid],bins = 50, lw=2, range=(-.2, .3), histtype='step', hatch="//",color='midnightblue',label='Old') 
    ax1.hist(r_new[ind_valid],bins = 50, lw=3, range=(-.2, .3), histtype='step', color='orangered', label='New') 
 
    ax1.set_xlabel('$\Delta$CaHK', fontsize=20)
    ax1.set_ylabel('number function', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax1.axvline(0, ls='dashed', lw=3, color='mediumseagreen')  
    ax1.set_title('run:%s, Validation Sample ($\delta$ CaHK < 0.015)'%run, fontsize=20)
    ax1.legend(fontsize=20)


    savefig(out_fig_path+"hist_residual_old_new%s.png"%params, dpi=200)

            
    
            
    dz1_grid_1d = dz1_grid.flatten()  

    ind_1d = dz1_grid_1d != 0
    vmin = np.percentile(dz1_grid_1d[ind_1d],2.5)
    vmax = np.percentile(dz1_grid_1d[ind_1d],97.5)
           
    

    figure(figsize = (22,5))
    gs = gridspec.GridSpec(1, 3)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.4, hspace = 0.)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])
    ax3 = subplot(gs[0, 2])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax1.pcolormesh(xedges, yedges, dz1_grid.T, cmap='jet',vmin=vmin,vmax=vmax)
    cb1=colorbar(im,cax=cax,orientation="vertical", format="%.2f",ticks=np.linspace(vmin, vmax,4))
    cb1.set_label('$\Delta$CaHK (Raw-Gaia-zpt)', fontsize=20)
    ax1.text(.0, 1.05, "run:%s"%run, fontsize=20)

    ax1.text(.5, 1.05, "RAW", fontsize=20)

    if semester == "15A":
    
        vmin_old = vmin - vmax
        vmax_old = 0
        ticks = np.linspace((vmin - vmax), 0, 4)
    
    else:
    
        vmin_old = 0
        vmax_old = vmax - vmin
        ticks = np.linspace(0, (vmax - vmin),4)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax2.pcolormesh(xedges,  yedges, dz_old_grid.T, cmap='jet',vmin=vmin_old,vmax=vmax_old)
    cb2=colorbar(im,cax=cax,orientation="vertical", format="%.2f",ticks=ticks)
    cax.yaxis.set_ticks_position('right')
    cb2.set_label('$\Delta$CaHK (FoV,old)', fontsize=20)
    ax2.text(.5, 1.05, "OLD", fontsize=20)



    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax3.pcolormesh(xedges,  yedges, fov_grid.T, cmap='jet',vmin=vmin,vmax=vmax)
    cb3=colorbar(im,cax=cax,orientation="vertical", format="%.2f",ticks=np.linspace(vmin, vmax,4))
    cax.yaxis.set_ticks_position('right')
    ax3.text(.5, 1.05, "NEW", fontsize=20)



    cb3.set_label('$\Delta$CaHK (FoV,new)', fontsize=20)

    for cb in [cb1, cb2, cb3]:

        cb.ax.tick_params(labelsize=20)

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([.2, .4, .6, .8, 1.])
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.tick_params(axis='both', which='major', labelsize=20)
        


    savefig(out_fig_path+"model_FoV_raw_old_new%s.png"%params, dpi=200)


    figure(figsize = (11,5))
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.05, right= 0.9, bottom = 0.15, top = 0.9, wspace=0.3, hspace = 1)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])



    v = chi_new_calib_grid.flatten()
    vmin = np.percentile(v, 2.5)
    vmax = np.percentile(v, 97.5)



    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax1.pcolormesh(xedges, yedges, chi_old_calib_grid.T, cmap='jet',vmin=vmin, vmax=vmax)
    cb1=colorbar(im,cax=cax,orientation="vertical", format="%.2f", ticks=np.linspace(vmin, vmax, 3))
    cb1.set_label('$\Delta$CaHK (Old-Gaia)', fontsize=20)
    ax1.set_title("Old", fontsize=20)
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax2.pcolormesh(xedges, yedges, chi_new_calib_grid.T, cmap='jet',vmin=vmin, vmax=vmax)
    cb2=colorbar(im,cax=cax,orientation="vertical", format="%.2f", ticks=np.linspace(vmin, vmax, 3))
    cb2.set_label('$\Delta$CaHK (New-Gaia)', fontsize=20)
    ax2.set_title("New", fontsize=20)


    
    savefig(out_fig_path+"chi_FoV_old_new%s.png"%params, dpi=200)
    
    
    

    vmin = -.1
    vmax = .1


    figure(figsize = (12,5))
    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.85, wspace=0.5, hspace = 0.2)
    ax1 = subplot(gs[0, 0])
    ax2 = subplot(gs[0, 1])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax1.pcolormesh(xedges, yedges, r_old_calib_grid.T, cmap='seismic',vmin=vmin,vmax=vmax)
    cb1=colorbar(im,cax=cax,orientation="vertical", format="%.1f",ticks=np.linspace(vmin, vmax,4))
    cb1.set_label('$\Delta$CaHK (old-Gaia)', fontsize=20)
    ax1.text(.4, 1.1, "Calibration Sample (run:%s, $\delta$ CaHK < 0.1)"%run, fontsize=20)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im=ax2.pcolormesh(xedges,  yedges, r_new_calib_grid.T, cmap='seismic',vmin=vmin,vmax=vmax)
    cb2=colorbar(im,cax=cax,orientation="vertical", format="%.1f",ticks=np.linspace(vmin, vmax,4))
    cax.yaxis.set_ticks_position('right')


    cb2.set_label('$\Delta$CaHK (new-Gaia)', fontsize=20)



    for cb in [cb1, cb2]:

        cb.ax.tick_params(labelsize=20)

    for ax in [ax1, ax2]:
        ax.set_xticks([.2, .4, .6, .8, 1.])
        ax.set_yticks([0, .2, .4, .6, .8, 1.])
        ax.tick_params(axis='both', which='major', labelsize=20)
    

    savefig(out_fig_path+"residual_FoV_grid_new%s.png"%params, dpi=200)









    figure(figsize = (10, 6))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax1 = subplot(gs[0, 0])


    im=ax1.scatter(ra_f, dec_f, c=chi_old_f, s=1, cmap='Blues', vmin=0., vmax=.1)
    cb=colorbar(im,orientation="vertical")

    cb.set_label('Chi square (Old - Gaia)', fontsize=20)
    ax1.set_xlim(np.max(ra), np.min(ra))
    ax1.set_title ("OLD (run:%s)"%run)
    savefig(out_fig_path+"residual_chi_old%s.png"%params, dpi=200)



    fig = figure(figsize = (10, 6))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax1 = subplot(gs[0, 0])


    im=ax1.scatter(ra_f, dec_f, c=chi_f, s=1, cmap='Blues', vmin=0., vmax=.1)
    cb=colorbar(im,orientation="vertical")

    cb.set_label('Chi square (New - Gaia)', fontsize=20)
    ax1.set_title ("NEW (run:%s)"%run)

    ax1.set_xlim(np.max(ra), np.min(ra))
    savefig(out_fig_path+"residual_chi_new%s.png"%params, dpi=200)
    
    
    vmin = -0.02
    vmax = 0.02
    
    figure(figsize = (10, 6))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.1, right= 0.9, bottom = 0.1, top = 0.9, wspace=0.3, hspace = 0.1)
    ax1 = subplot(gs[0, 0])


    im=ax1.scatter(ra_f, dec_f, c=chi_f-chi_old_f, s=1, cmap='seismic', vmin=vmin, vmax=vmax)
    cb=colorbar(im,orientation="vertical")

    cb.set_label('Chi square (New - Old)', fontsize=20)
    ax1.set_xlim(np.max(ra), np.min(ra))
    ax1.set_title ("New-OLD (run:%s)"%run)
    savefig(out_fig_path+"residual_chi_new-old%s.png"%params, dpi=200)




    return









