################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cmcrameri import cm
import math 
import os
import sys


temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("50000.00", "60000.00", "70000.00", "80000.00", "90000.00", "100000.00", "110000.00", "120000.00", "130000.00", "140000.00", "150000.00", "160000.00", "170000.00", "180000.00", "190000.00", "200000.00", "210000.00", "220000.00", "230000.00", "240000.00", "250000.00", "260000.00", "270000.00", "280000.00", "290000.00", "300000.00", "310000.00", "320000.00", "330000.00", "340000.00", "350000.00", "360000.00", "370000.00", "380000.00", "390000.00", "400000.00")
print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 0;  j_end = len(size); print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 1;  t_end = 10;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end-1) + "Myr")


ticks_size = 13
label_size = 16
vmax = 0.5
vmin = -0.7

ref_i = 15
ref_j = 15
opt_i = 14
opt_j = 11

print(temp[ref_i])
print(size[ref_j])

save_err = np.load(file="Ref_Model.npy")
save_err = save_err / (240 * 20)
X, Y = np.meshgrid(np.float64(size[j_start:j_end])/1000, np.float64(temp[i_start:i_end]))
# Create a canvas
figure = plt.figure(figsize=(18,6), dpi=300)
# Create a layer
figure_ax1 = figure.add_axes([0.05,0.11,0.27,0.82]) #Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([0.35,0.11,0.27,0.82]) #Sets the position of the layer on the canvas
figure_ax3 = figure.add_axes([0.65,0.11,0.27,0.82])
figure_ax1.annotate('(a) '+str(t_start)+' Myr', xy=(-0.06,1.04), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax2.annotate('(b) '+str(t_end)+' Myr', xy=(-0.06,1.04), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax3.annotate('(c) '+str(t_start) + '–' + str(t_end)+' Myr', xy=(-0.06,1.04), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax4 = figure.add_axes([0.93,0.11,0.015,0.82])

cm_colorbar = cm.navia_r
cm_colorbar.set_bad(color='black')

save_err_1 = np.log10(save_err[ref_i,ref_j,t_start])
save_err_11= np.log10(save_err[opt_i,opt_j,t_start])

img1 = figure_ax1.pcolormesh(X,Y,np.log10(save_err[:,:,t_start]),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img1_contour=figure_ax1.contour(X,Y,np.log10(save_err[:,:,t_start]),[-0.3,-0.2,-0.1], colors = 'black', linewidths=2.0, linestyles=':')
h11,ll=img1_contour.legend_elements()
img11 = figure_ax1.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img111 = figure_ax1.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=1.5, linestyles='--', s = 90)
figure_ax1.legend([h11[0], img11,img111], [r'$T_{res}$ Contour', '3D Ref_Model', '2D Opt_Ref_Model'], loc='upper right', fontsize=ticks_size)

save_err_2 = np.log10(save_err[ref_i,ref_j,t_end])
save_err_22= np.log10(save_err[opt_i,opt_j,t_end])

img2 = figure_ax2.pcolormesh(X,Y,np.log10(save_err[:,:,t_end]),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img2_contour=figure_ax2.contour(X,Y,np.log10(save_err[:,:,t_end]),[-0.7,-0.5,-0.3], colors = 'black', linewidths=2.0, linestyles=':')
h21,ll=img2_contour.legend_elements()
img22 = figure_ax2.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img222 = figure_ax2.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=1.5, linestyles='--', s = 90)
figure_ax2.legend([h21[0], img22,img222], [r'$T_{res}$ Contour', '3D Ref_Model', '2D Opt_Ref_Model'], loc='upper right', fontsize=ticks_size)

save_err_3 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_33 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img3 = figure_ax3.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img3_contour=figure_ax3.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.5,-0.3,-0.2], colors = 'black', linewidths=2.0, linestyles=':')
h31,ll=img3_contour.legend_elements()
img33 = figure_ax3.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c='none', marker='s', edgecolors='red', linewidths=2.0, s = 90)
img333 = figure_ax3.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c='none', marker='s', edgecolors='blue', linewidths=2.0, s = 90)
figure_ax3.legend([h31[0], img33,img333], [r'$T_{res}$ Contour', '3D Ref_Model', '2D Opt_Ref_Model'], loc='upper right', fontsize=ticks_size)



figure_ax1.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax1.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(labelsize = ticks_size)
figure_ax1.minorticks_on()


figure_ax2.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax2.set_ylabel('', fontsize = label_size)
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.set_yticks([])
figure_ax2.minorticks_on()

figure_ax3.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax3.set_ylabel('', fontsize = label_size)
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax3.set_yticks([])
figure_ax3.minorticks_on()


  # Plot colorbar
cbar = figure.colorbar(img2, cax=figure_ax4, orientation = 'vertical',extend='max')
cbar.set_label(r'$log_{10} \left( T_{res} \right)$ Between 3D & 2D Modeling', fontsize=label_size, rotation=-90, labelpad=18)
cbar.ax.tick_params(labelsize = ticks_size)
cbar.minorticks_on()
  
  # Displays the drawing results
plt.savefig("./figure_4.png", dpi=300, format="png")
