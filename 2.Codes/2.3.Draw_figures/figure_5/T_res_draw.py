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
size   = ("50000.00", "60000.00", "70000.00", "80000.00", "90000.00", "100000.00", "110000.00", "120000.00", "130000.00", "140000.00", "150000.00", "160000.00", "170000.00", "180000.00", "190000.00", "200000.00", "210000.00", "220000.00", "230000.00", "240000.00", "250000.00", "260000.00", "270000.00", "280000.00", "290000.00", "300000.00", "310000.00", "320000.00", "330000.00", "340000.00", "350000.00", "360000.00", "370000.00", "380000.00", "390000.00", "400000.00")
print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 0;  j_end = len(size); print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 1;  t_end = 10;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end-1) + "Myr")


ticks_size = 14
label_size = 16
vmax = 0.4
vmin = -0.7

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
figure_ax1.annotate('(a) D_125: 125km, 2100K (+301K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax2.annotate('(b) D_150: 150km, 2100K (+301K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax3.annotate('(c) D_300: 300km, 2100K (+301K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax4.annotate('(d) D_400: 400km, 2100K (+301K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax5.annotate('(e) T_2000: 200km, 2000K (+201K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax6.annotate('(f) T_2050: 200km, 2050K (+251K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax7.annotate('(g) T_2150: 200km, 2150K (+351K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)
figure_ax8.annotate('(h) T_2200: 200km, 2200K (+401K)', xy=(dx,dy), xycoords="axes fraction", fontsize=label_size)


cm_colorbar = cm.navia_r
cm_colorbar.set_bad(color='k')

figure_ax0.set_xlim([0,1.0])
figure_ax0.set_ylim([0,1.0])
figure_ax0.plot([0.485,0.485],[0.0,1.0],linewidth=2.0,c='black',linestyle='dashed')
figure_ax0.text(0.11, 0.99, 'Different Diameters', fontsize = label_size+3, color = "black",bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7, lw=2.0))
figure_ax0.text(0.63, 0.99, 'Different Temperatures', fontsize = label_size+3, color = "black",bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7, lw=2.0))
figure_ax0.set_xticks([])
figure_ax0.set_yticks([])
figure_ax0.axis('off')

ref_i = 15
ref_j = 7
save_err = np.load(file="D_125.npy")
save_err = save_err / (240 * 20)

num_average = 1
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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_1 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_11 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img1 = figure_ax1.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img1_contour=figure_ax1.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.6,-0.4,-0.2], colors = 'black', linewidths=2.0, linestyles=':')
h11,ll=img1_contour.legend_elements()
img12 = figure_ax1.scatter(125,2100, c=save_err_1, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img13 = figure_ax1.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_11, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax1.legend([h11[0], img12, img13], [r'$T_{res}$ Contour', '3D D_125', '2D Opt_D_125'], loc='upper right', fontsize=ticks_size)
figure_ax1.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")


ref_i = 15
ref_j = 10

save_err = np.load(file="D_150.npy")
save_err = save_err / (240 * 20)

num_average = 1
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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_2 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_22 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img2 = figure_ax2.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img2_contour=figure_ax2.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.5,-0.3,-0.2], colors = 'black', linewidths=2.0, linestyles=':')
h21,ll=img2_contour.legend_elements()
img22 = figure_ax2.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_2, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img23 = figure_ax2.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_22, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax2.legend([h21[0], img22, img23], [r'$T_{res}$ Contour', '3D D_150', '2D Opt_D_150'], loc='upper right', fontsize=ticks_size)
figure_ax2.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 15
ref_j = 25

save_err = np.load(file="D_300.npy")
save_err = save_err / (240 * 20)

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
opt_i = 14
opt_j = 17
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_3 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_33 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img3 = figure_ax3.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img3_contour=figure_ax3.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,-0.0], colors = 'black', linewidths=2.0, linestyles=':')
h31,ll=img3_contour.legend_elements()
img32 = figure_ax3.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_3, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img33 = figure_ax3.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_33, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax3.legend([h31[0], img32, img33], [r'$T_{res}$ Contour', '3D D_300', '2D Opt_D_300'], loc='upper right', fontsize=ticks_size)
figure_ax3.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 15
ref_j = 35

save_err = np.load(file="D_400.npy")
save_err = save_err / (240 * 20)

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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_4 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_44 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img4 = figure_ax4.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img4_contour=figure_ax4.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.3,-0.2,-0.0], colors = 'black', linewidths=2.0, linestyles=':')
h41,ll=img4_contour.legend_elements()
img42 = figure_ax4.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_4, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img43 = figure_ax4.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_44, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax4.legend([h41[0], img42, img43], [r'$T_{res}$ Contour', '3D D_400', '2D Opt_D_400'], loc='upper right', fontsize=ticks_size)
figure_ax4.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 5
ref_j = 15

save_err = np.load(file="T_2000.npy")
save_err = save_err / (240 * 20)

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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_5 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_55 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img5 = figure_ax5.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img5_contour=figure_ax5.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.8,-0.5,-0.3], colors = 'black', linewidths=2.0, linestyles=':')
h51,ll=img5_contour.legend_elements()
img52 = figure_ax5.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_5, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img53 = figure_ax5.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_55, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax5.legend([h51[0], img52, img53], [r'$T_{res}$ Contour', '3D T_2000', '2D Opt_T_2000'], loc='upper right', fontsize=ticks_size)
figure_ax5.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 10
ref_j = 15

save_err = np.load(file="T_2050.npy")
save_err = save_err / (240 * 20)

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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_6 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_66 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img6 = figure_ax6.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img6_contour=figure_ax6.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.5,-0.3,-0.2], colors = 'black', linewidths=2.0, linestyles=':')
h61,ll=img6_contour.legend_elements()
img62 = figure_ax6.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_6, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img63 = figure_ax6.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_66, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax6.legend([h61[0], img62, img63], [r'$T_{res}$ Contour', '3D T_2050', '2D Opt_T_2050'], loc='upper right', fontsize=ticks_size)
figure_ax6.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 20
ref_j = 15

save_err = np.load(file="T_2150.npy")
save_err = save_err / (240 * 20)

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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_7 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_77 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img7 = figure_ax7.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img7_contour=figure_ax7.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,-0.1], colors = 'black', linewidths=2.0, linestyles=':')
h71,ll=img7_contour.legend_elements()
img72 = figure_ax7.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_7, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img73 = figure_ax7.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_77, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax7.legend([h71[0], img72, img73], [r'$T_{res}$ Contour', '3D T_2150', '2D Opt_T_2150'], loc='upper right', fontsize=ticks_size)
figure_ax7.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

ref_i = 25
ref_j = 15

save_err = np.load(file="T_2200.npy")
save_err = save_err / (240 * 20)

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
print('opt_i = ',opt_i, 'opt_j = ', opt_j,'Temp: ', temp[opt_i], 'Size: ', size[opt_j])

save_err_8 = np.log10(np.sum(save_err[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
save_err_88 = np.log10(np.sum(save_err[opt_i,opt_j,t_start:t_end+1])/(t_end-t_start+1))

img8 = figure_ax8.pcolormesh(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img8_contour=figure_ax8.contour(X,Y,np.log10(np.sum(save_err[:,:,t_start:t_end+1],2)/(t_end-t_start+1)), [-0.4,-0.2,-0.1], colors = 'black', linewidths=2.0, linestyles=':')
h81,ll=img8_contour.legend_elements()
img82 = figure_ax8.scatter(np.float64(size[ref_j])/1000.0,np.float64(temp[ref_i]), c=save_err_8, cmap=cm_colorbar, marker='s', edgecolors='red', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
img83 = figure_ax8.scatter(np.float64(size[opt_j])/1000.0,np.float64(temp[opt_i]), c=save_err_88, cmap=cm_colorbar, marker='s', edgecolors='blue', linewidths=2.0, s = 90, label="3D ref_model", vmin = vmin, vmax = vmax)
figure_ax8.legend([h81[0], img82, img83], [r'$T_{res}$ Contour', '3D T_2200', '2D Opt_T_2200'], loc='upper right', fontsize=ticks_size)
figure_ax8.text(80, 1960, str(t_start) + '–' + str(t_end)+' Myr', style = 'oblique', fontsize = label_size, color = "black")

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
cbar.set_label(r'$log_{10} \left( T_{res} \right)$ Between 3D & 2D Modeling', fontsize=label_size)
cbar.ax.tick_params(labelsize = ticks_size)
cbar.minorticks_on()
  
  # Displays the drawing results
plt.savefig("./noTail1_10Myr.png", dpi=300, format="png")
plt.ioff() # Open the interactive drawing. If closed, the program pauses while drawing 