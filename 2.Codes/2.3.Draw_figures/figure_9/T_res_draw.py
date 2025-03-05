################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
import math 
import os
import sys


temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00")
print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 3;  j_end = len(size); print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 1;  t_end = 10;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end) + "Myr")

ticks_size = 14
label_size = 16
vmax = 0.6
vmin = -0.6


save_flux_2D = 1000 * np.load(file="2D_flux.npy")
save_flux_2D[save_flux_2D==np.inf]=-np.inf
for _ in range(t_start,t_end+1):
  max_in_row_index = np.argmax(save_flux_2D[:,:,_], axis=1)
  for __ in range(0,len(temp)):
    save_flux_2D[__,max_in_row_index[__]:,_]=np.inf
save_flux_2D[save_flux_2D==-np.inf]=np.inf


X, Y = np.meshgrid(np.float64(size[j_start:j_end])/1000, np.float64(temp[i_start:i_end]))
# Create a canvas
figure = plt.figure(figsize=(12,18), dpi=300)
# Create a layer
figure_ax0 = figure.add_axes([0.1,0.06,0.9,0.93])

left = 0.1
bottom = 0.095
width = 0.40
length = 0.19
gap_LR = 0.08
gap_TB = 0.03
cbar_width=0.02

figure_ax1 = figure.add_axes([left + 0*(width + gap_LR),bottom + 3*(length + gap_TB),width,length]) #Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([left + 0*(width + gap_LR),bottom + 2*(length + gap_TB),width,length])
figure_ax3 = figure.add_axes([left + 0*(width + gap_LR),bottom + 1*(length + gap_TB),width,length])
figure_ax4 = figure.add_axes([left + 0*(width + gap_LR),bottom + 0*(length + gap_TB),width,length])
figure_ax5 = figure.add_axes([left + 1*(width + gap_LR),bottom + 3*(length + gap_TB),width,length]) 
figure_ax6 = figure.add_axes([left + 1*(width + gap_LR),bottom + 2*(length + gap_TB),width,length])
figure_ax7 = figure.add_axes([left + 1*(width + gap_LR),bottom + 1*(length + gap_TB),width,length]) 
figure_ax8 = figure.add_axes([left + 1*(width + gap_LR),bottom + 0*(length + gap_TB),width,length])
figure_ax9 = figure.add_axes([left, bottom-cbar_width-gap_TB-0.01, 2*(width + gap_LR) - gap_LR,cbar_width])

dx = -0.08
dy = 1.03
figure_ax1.annotate('(a) D100: 100km, 2100K (+268K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax2.annotate('(b) D125: 125km, 2100K (+268K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax3.annotate('(c) D175: 175km, 2100K (+268K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax4.annotate('(d) D200: 200km, 2100K (+268K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax5.annotate('(e) T2000: 150km, 2000K (+168K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax6.annotate('(f) T2050: 150km, 2050K (+218K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax7.annotate('(g) T2150: 150km, 2150K (+318K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax8.annotate('(h) T2200: 150km, 2200K (+368K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)

cm_colorbar = cm.navia_r
cm_colorbar.set_bad(color='black')

figure_ax0.set_xlim([0,1.0])
figure_ax0.set_ylim([0,1.0])
figure_ax0.plot([0.485,0.485],[0.0,1.0],linewidth=2.0,c='black',linestyle='dashed')
figure_ax0.text(0.11, 0.99, 'Different Diameters', fontsize = label_size+3, color = "black",bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7, lw=2.0))
figure_ax0.text(0.63, 0.99, 'Different Temperatures', fontsize = label_size+3, color = "black",bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7, lw=2.0))
figure_ax0.set_xticks([])
figure_ax0.set_yticks([])
figure_ax0.axis('off')

ref_i = 15
ref_j = 19
save_err = np.load(file="D100.npy")
save_err = save_err / (400 * 20)

num_average = 20
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=25; opt_j=6
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_1 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_11 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img1 = figure_ax1.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img1_contour=figure_ax1.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,-0.0], colors = 'black', linewidths=2.0, linestyles=':')
h11,ll=img1_contour.legend_elements()
img11_contour=figure_ax1.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h12,ll=img11_contour.legend_elements()
img11 = figure_ax1.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img111 = figure_ax1.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax1.legend([h11[0], h12[0], img11, img111], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D D100', '2D Opt_D100'], loc='upper right', fontsize=ticks_size)
figure_ax1.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")


ref_i = 15
ref_j = 24
save_err = np.load(file="D125.npy")
save_err = save_err / (400 * 20)

num_average = 15
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=24; opt_j=7
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_2 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_22 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img2 = figure_ax2.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img2_contour=figure_ax2.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,-0.0], colors = 'black', linewidths=2.0, linestyles=':')
h21,ll=img2_contour.legend_elements()
img22_contour=figure_ax2.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h22,ll=img22_contour.legend_elements()
img22 = figure_ax2.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img222 = figure_ax2.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax2.legend([h21[0], h22[0], img22, img222], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D D125', '2D Opt_D125'], loc='upper right', fontsize=ticks_size)
figure_ax2.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")


ref_i = 15
ref_j = 34
save_err = np.load(file="D175.npy")
save_err = save_err / (400 * 20)

num_average = 10
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=18; opt_j=14
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_3 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_33 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img3 = figure_ax3.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img3_contour=figure_ax3.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.1,0.1,0.3], colors = 'black', linewidths=2.0, linestyles=':')
h31,ll=img3_contour.legend_elements()
img33_contour=figure_ax3.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h32,ll=img33_contour.legend_elements()
img33 = figure_ax3.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img333 = figure_ax3.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax3.legend([h31[0], h32[0], img33, img333], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D D175', '2D Opt_D175'], loc='upper right', fontsize=ticks_size)
figure_ax3.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")

t_end=6
ref_i = 15
ref_j = 39
save_err = np.load(file="D200.npy")
save_err = save_err / (400 * 20)

num_average = 10
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=18; opt_j=16
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_4 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_44 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img4 = figure_ax4.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img4_contour=figure_ax4.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.1,0.1,0.3], colors = 'black', linewidths=2.0, linestyles=':')
h41,ll=img4_contour.legend_elements()
img44_contour=figure_ax4.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h42,ll=img44_contour.legend_elements()
img44 = figure_ax4.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img444 = figure_ax4.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax4.legend([h41[0], h42[0], img44, img444], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D D200', '2D Opt_D200'], loc='upper right', fontsize=ticks_size)
figure_ax4.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")
t_end=10


ref_i = 5
ref_j = 29
save_err = np.load(file="T2000.npy")
save_err = save_err / (400 * 20)

num_average = 32
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.floor(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=14; opt_j=9
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_5 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_55 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img5 = figure_ax5.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img5_contour=figure_ax5.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [-0.6,-0.4,-0.2], colors = 'black', linewidths=2.0, linestyles=':')
h51,ll=img5_contour.legend_elements()
img55_contour=figure_ax5.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h52,ll=img55_contour.legend_elements()
img55 = figure_ax5.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img555 = figure_ax5.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax5.legend([h51[0], h52[0], img55, img555], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D T2000', '2D Opt_T2000'], loc='upper right', fontsize=ticks_size)
figure_ax5.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")


ref_i = 10
ref_j = 29
save_err = np.load(file="T2050.npy")
save_err = save_err / (400 * 20)

num_average = 10
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=15; opt_j=11
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_6 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_66 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img6 = figure_ax6.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img6_contour=figure_ax6.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,0.0], colors = 'black', linewidths=2.0, linestyles=':')
h61,ll=img6_contour.legend_elements()
img66_contour=figure_ax6.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h62,ll=img66_contour.legend_elements()
img66 = figure_ax6.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img666 = figure_ax6.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax6.legend([h61[0], h62[0], img66, img666], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D T2050', '2D Opt_T2050'], loc='upper right', fontsize=ticks_size)
figure_ax6.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")



ref_i = 20
ref_j = 29
save_err = np.load(file="T2150.npy")
save_err = save_err / (400 * 20)

num_average = 10
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=24; opt_j=11
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_7 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_77 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img7 = figure_ax7.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img7_contour=figure_ax7.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.1,+0.1,0.3], colors = 'black', linewidths=2.0, linestyles=':')
h71,ll=img7_contour.legend_elements()
img77_contour=figure_ax7.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h72,ll=img77_contour.legend_elements()
img77 = figure_ax7.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img777 = figure_ax7.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax7.legend([h71[0], h72[0], img77, img777], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D T2150', '2D Opt_T2150'], loc='lower right', fontsize=ticks_size)
figure_ax7.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")


t_end=8
ref_i = 25
ref_j = 29
save_err = np.load(file="T2200.npy")
save_err = save_err / (400 * 20)

num_average = 3
array_2d = (np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1))
array_1d = array_2d.flatten()
sorted_indices = np.argsort(array_1d)
smallest_indices = sorted_indices[:num_average]
smallest_2d_indices = [np.unravel_index(index, array_2d.shape) for index in smallest_indices]
smallest_values = array_1d[smallest_indices]
opt_i=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[0])
opt_j=int(np.round(np.dot(1.0/smallest_values/smallest_values,smallest_2d_indices)/np.sum(1.0/smallest_values/smallest_values))[1])
print('------------------------------------------------------------------------------')
print('Opt_Model for ',temp[ref_i],'(K); ',size[ref_j],'(km) is: ')
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])
opt_i=27; opt_j=11
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_8 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_88 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img8 = figure_ax8.pcolormesh(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img8_contour=figure_ax8.contour(X,Y,np.log10(np.sum(save_err[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.0,+0.2,0.4], colors = 'black', linewidths=2.0, linestyles=':')
h81,ll=img8_contour.legend_elements()
img88_contour=figure_ax8.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,1:t_end+1],2)/(t_end)), [np.log10(np.sum(save_flux_2D[opt_i,opt_j,1:t_end+1])/(t_end))], colors = 'blue', linewidths=2.0, linestyles='--')
h82,ll=img11_contour.legend_elements()
img88 = figure_ax8.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img888 = figure_ax8.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax8.legend([h81[0], h82[0], img88, img888], [r'$T_{res}$ Contour', r'$Q_{2D}$ Contour', '3D T2200', '2D Opt_T2200'], loc='lower right', fontsize=ticks_size)
figure_ax8.text(20, 1960, str(t_start) + '~' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "white")
t_end=10

figure_ax1.set_xlabel('', fontsize = label_size)
figure_ax1.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(labelsize = ticks_size)
figure_ax1.set_xticks([])
figure_ax1.minorticks_on()

figure_ax2.set_xlabel('', fontsize = label_size)
figure_ax2.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.minorticks_on()
figure_ax2.set_xticks([])

figure_ax3.set_xlabel('', fontsize = label_size)
figure_ax3.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax3.minorticks_on()
figure_ax3.set_xticks([])

figure_ax4.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax4.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax4.tick_params(labelsize = ticks_size)
figure_ax4.minorticks_on()


figure_ax5.set_xlabel('', fontsize = label_size)
figure_ax5.set_ylabel('', fontsize = label_size)
figure_ax5.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax5.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax5.tick_params(labelsize = ticks_size)
figure_ax5.minorticks_on()
figure_ax5.set_xticks([])
figure_ax5.set_yticks([])

figure_ax6.set_xlabel('', fontsize = label_size)
figure_ax6.set_ylabel('', fontsize = label_size)
figure_ax6.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax6.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax6.tick_params(labelsize = ticks_size)
figure_ax6.minorticks_on()
figure_ax6.set_xticks([])
figure_ax6.set_yticks([])

figure_ax7.set_xlabel('', fontsize = label_size)
figure_ax7.set_ylabel('', fontsize = label_size)
figure_ax7.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax7.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax7.tick_params(labelsize = ticks_size)
figure_ax7.minorticks_on()
figure_ax7.set_xticks([])
figure_ax7.set_yticks([])

figure_ax8.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax8.set_ylabel('', fontsize = label_size)
figure_ax8.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax8.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax8.tick_params(labelsize = ticks_size)
figure_ax8.minorticks_on()
figure_ax8.set_yticks([])


  # Plot colorbar
cbar = figure.colorbar(img3, cax=figure_ax9, orientation = 'horizontal',extend='max')
cbar.set_label(r'$\log_{10}\left(T_{res}\right)$ Between 3D & 2D Modeling', fontsize=label_size)
cbar.ax.tick_params(labelsize = ticks_size)
cbar.minorticks_on()
  
  # Displays the drawing results
plt.savefig("./figure_9_1_10Myr.png", dpi=300, format="png")