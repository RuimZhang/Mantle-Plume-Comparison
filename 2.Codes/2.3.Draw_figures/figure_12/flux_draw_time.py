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
from scipy.interpolate import interp1d
from scipy.optimize import minimize


temp   = np.array(["1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0"])
size   = np.array(["5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00"])
model_name=["D100", "D125", "RefModel", "D175", "D200", "T2000", "T2050", "T2150", "T2200"]
i2_size = np.array([40000,55000,70000,82500,100000])
i2_temp = np.array([2100, 2100, 2100, 2100, 2100])
i3_size = np.array([100000, 125000, 150000, 175000, 200000, 150000, 150000, 150000, 150000])
i3_temp = np.array([2100, 2100, 2100, 2100, 2100, 2000, 2050, 2150, 2200])
i3_size1 = np.array([100000, 125000, 150000, 175000, 200000])
i3_temp1 = np.array([2100, 2100, 2100, 2100, 2100])
i3_size2= np.array([150000, 150000,150000, 150000, 150000])
i3_temp2 = np.array([2000, 2050, 2100, 2150, 2200])
print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 3;  j_end = len(size)-3; print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 1;  t_end = 10;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end) + "Myr")


ticks_size = 13
label_size = 15
vmax = -1.5
vmin = -5.0

ref_i = 15
ref_j = 29

print(temp[ref_i])
print(size[ref_j])

save_flux_2D = 1000 * np.load(file="2D_flux.npy")
save_flux_2D[save_flux_2D==np.inf]=-np.inf
for _ in range(t_start,t_end+1):
  max_in_row_index = np.argmax(save_flux_2D[:,:,_], axis=1)
  #print(max_in_row_index)
  for __ in range(0,len(temp)):
    save_flux_2D[__,max_in_row_index[__]:,_]=np.inf
save_flux_2D[save_flux_2D==-np.inf]=np.inf

save_flux_3D = 1000 * 1000 * np.load(file="3D_flux.npy")
save_flux_3D = np.log10(np.sum(save_flux_3D[:,t_start:t_end+1],1)/(t_end-t_start+1))


X, Y = np.meshgrid(np.float64(size[j_start:j_end])/1000, np.float64(temp[i_start:i_end]))
# Create a canvas
figure = plt.figure(figsize=(15,12), dpi=300)
# Create a layer
figure_ax1 = figure.add_axes([0.07,0.55,0.36,0.40]) #Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([0.54,0.55,0.36,0.40]) #Sets the position of the layer on the canvas
figure_ax3 = figure.add_axes([0.07,0.08,0.36,0.40])
figure_ax4 = figure.add_axes([0.54,0.08,0.36,0.40])
figure_ax1.annotate('(a)', xy=(-0.15,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax2.annotate('(b)', xy=(-0.15,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax3.annotate('(c)', xy=(-0.15,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax4.annotate('(d)', xy=(-0.15,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax5 = figure.add_axes([0.92,0.55,0.015,0.40])
figure_ax6 = figure.add_axes([0.92,0.08,0.015,0.40])

cm_colorbar = cm.lapaz_r
cm_colorbar.set_bad(color='black')


save_err_2 = np.log10(save_flux_2D[ref_i,ref_j,t_end])


save_err_3 = np.log10(np.sum(save_flux_2D[ref_i,ref_j,t_start:t_end+1])/(t_end-t_start+1))
LL=[-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3.0,-2.9,-2.8]
LL2=[-3.5,-3.3,-3.1,-2.9]

img1 = figure_ax1.pcolormesh(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img1_contour=figure_ax1.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), LL, colors = 'black', linewidths=1.0, linestyles='dashdot')
h11,ll=img1_contour.legend_elements()
figure_ax1.legend([h11[0]], [r'$Q_{2D}$ in 2D Numerical Models'], loc='upper right', fontsize=ticks_size)


ff=[]

for _ in range(10):
  contour_paths = img1_contour.collections[_].get_paths()
  for i, contour_path in enumerate(contour_paths):
      v = contour_path.vertices
      D_2d = v[:,0] * 1000
      T_2d = v[:,1]
      f = interp1d(T_2d,D_2d)
      ff.append(f)
TT=np.linspace(1955,2245,100)
def func_res0(para, sign=1.0):
  r=0
  for _ in range(10):
    r0 = np.linalg.norm(para[0] * ff[_](TT) ** para[1] * (TT-1832) ** para[2] - ff[_](2100),ord=2)
    r=r+r0
  return r
  
para0=[ 2.946e-03, 1.043e+00, 9.602e-01]
err0 = minimize(func_res0, para0, args=(1,), method='trust-constr',tol=1.0e-20)
print(err0)

def func_res1(para, sign=1.0):
  r=0
  for _ in range(1,10):
    r = r + (para[1] * np.log10(ff[_](2100)) + np.log10(para[0]) - LL[_])**2
  return r

para1=[2.278e-15, 2.409e+00]
err1 = minimize(func_res1, para1, args=(1,), method='Nelder-Mead',tol=1.0e-20)
print(err1)

X1, Y1 = np.meshgrid(np.linspace(20,200,1000), np.linspace(1950,2250,1000))
img2=figure_ax2.pcolormesh(X1,Y1, err1.x[1] * np.log10(err0.x[0] * (X1*1000)**err0.x[1] * (Y1 - 1832)**err0.x[2]) + np.log10(err1.x[0]) ,cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img2_contour1 = figure_ax2.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), LL2, colors = 'black', linewidths=1.2, linestyles='dashdot')
img2_contour2 = figure_ax2.contour(X1,Y1,err1.x[1] * np.log10(err0.x[0] * (X1*1000)**err0.x[1] * (Y1 - 1832)**err0.x[2]) + np.log10(err1.x[0]), LL2, colors = 'red', linewidths=1.8, linestyles='-')
h21,ll=img2_contour1.legend_elements()
h22,ll=img2_contour2.legend_elements()
figure_ax2.legend([h21[0], h22[0]], [r'$Q_{2D}$: Numerical Models' ,r'$Q_{2D}$: Analytical Predictions'], loc='upper right', fontsize=ticks_size)

def func_res4(para, sign=1.0):
  r=0
  r = r + (np.log10(para[0]) + np.log10(100000)*para[1] +  np.log10(2100-1832)*para[2]- save_flux_3D[0])**2
  r = r + (np.log10(para[0]) + np.log10(125000)*para[1] +  np.log10(2100-1832)*para[2]- save_flux_3D[1])**2
  r = r + (np.log10(para[0]) + np.log10(150000)*para[1] +  np.log10(2100-1832)*para[2]- save_flux_3D[2])**2 * 2
  r = r + (np.log10(para[0]) + np.log10(175000)*para[1] +  np.log10(2100-1832)*para[2]- save_flux_3D[3])**2
  r = r + (np.log10(para[0]) + np.log10(200000)*para[1] +  np.log10(2100-1832)*para[2]- save_flux_3D[4])**2
  r = r + (np.log10(para[0]) + np.log10(150000)*para[1] +  np.log10(2000-1832)*para[2]- save_flux_3D[5])**2
  r = r + (np.log10(para[0]) + np.log10(150000)*para[1] +  np.log10(2050-1832)*para[2]- save_flux_3D[6])**2
  r = r + (np.log10(para[0]) + np.log10(150000)*para[1] +  np.log10(2150-1832)*para[2]- save_flux_3D[7])**2
  r = r + (np.log10(para[0]) + np.log10(150000)*para[1] +  np.log10(2200-1832)*para[2]- save_flux_3D[8])**2
  return r

para4=[ 5.390e-44, 7.010e+00, 4.046e+00]
err4 = minimize(func_res4, para4, args=(1,), method='Nelder-Mead')
print(err4)

figure_ax32 = figure_ax3.twinx()
img31,=figure_ax3.plot((save_flux_3D[5], save_flux_3D[6],save_flux_3D[2],save_flux_3D[7],save_flux_3D[8]), (2000,2050,2100,2150,2200), color='red', marker='o', markersize = 5, linestyle='--')
img32,=figure_ax32.plot((save_flux_3D[0], save_flux_3D[1],save_flux_3D[2],save_flux_3D[3],save_flux_3D[4]), (100,125,150,175,200), color='blue', marker='o', markersize = 5, linestyle='dashdot')
model_name1 = ['T2000', 'T2050', 'T2150', 'T2200']
x = [10,10,-50,-50]
for i, name in enumerate(model_name1):
  figure_ax3.annotate(name, ([save_flux_3D[5], save_flux_3D[6],save_flux_3D[7],save_flux_3D[8]][i],[2000,2050,2150,2200][i]), textcoords='offset points', xytext=(x[i],-10), ha='left', fontsize=ticks_size)
model_name2 = ['D100', 'D125', 'RefModel', 'D175', 'D200']
x = [5,-50,10,5,-40]
for i, name in enumerate(model_name2):
  figure_ax32.annotate(name, ([save_flux_3D[0], save_flux_3D[1],save_flux_3D[2],save_flux_3D[3],save_flux_3D[4]][i],[100,125,150,175,200][i]), textcoords='offset points', xytext=(x[i],-10), ha='left', fontsize=ticks_size)

img33=figure_ax3.scatter(np.log10(err4.x[0]*i3_size2**err4.x[1]*(np.array(i3_temp2)-1832)**err4.x[2]), i3_temp2, c='black',marker='+', s = 150, linewidth=2)
img34=figure_ax32.scatter(np.log10(err4.x[0]*i3_size1**err4.x[1]*(np.array(i3_temp1)-1832)**err4.x[2]), np.array(i3_size1)/1000, c='black',marker='+', s = 150, linewidth=2)
figure_ax32.legend([img31, img32, img33], ['3D Models with Diff. Temp.', '3D Models with Diff. Diam. ', '$Q_{3D}$: Analytical Predictions'], loc='upper left', fontsize=ticks_size)

vmax=6
vmin=-3
cm_colorbar = cm.batlow_r
XX,YY = np.meshgrid(np.linspace(50,300,1000), np.linspace(1900,2300,1000))
img4=figure_ax4.pcolormesh(XX,YY, np.log10(err4.x[0]*(XX*1000)**err4.x[1]*(YY-1832)**err4.x[2]),cmap=cm_colorbar, vmin = vmin, vmax = vmax)
img4_contour = figure_ax4.contour(XX,YY, np.log10(err4.x[0]*(XX*1000)**err4.x[1]*(YY-1832)**err4.x[2]), colors = 'black', linewidths=1.5, linestyles='--')
h41,ll=img4_contour.legend_elements()
img41=figure_ax4.scatter(i3_size/1000, i3_temp, c='none', marker='s', s = 120, linewidth=2.0, edgecolors='black')
x=[-30,-25,-25,-5,-0,-15,-15,-15,-15]
for i, name in enumerate(model_name):
  figure_ax4.annotate(name, (i3_size[i]/1000,i3_temp[i]), textcoords='offset points', xytext=(x[i],-20), ha='left', fontsize=ticks_size-2)

figure_ax4.legend([h41[0],img41], [r'$Q_{3D}$: Analytical Predictions', r'3D Numerical Models'], loc='upper right', fontsize=ticks_size)

Q3D = err4.x[0]*i3_size1**err4.x[1]*(i3_temp1-1832)**err4.x[2]
Q2D = err1.x[0]*(err0.x[0] * (i2_size)**err0.x[1] * (i2_temp - 1832)**err0.x[2])**err1.x[1]

print(err1.x[0]*err0.x[0]**err1.x[1], err0.x[1]*err1.x[1], err0.x[2]*err1.x[1])
print(err4.x)

figure_ax1.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax1.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax1.tick_params(labelsize = ticks_size)
figure_ax1.minorticks_on()

figure_ax2.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax2.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax2.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.minorticks_on()


figure_ax3.set_xlabel(r'$log_{10}(\overline{Q_{3D}})$ $(m^3/s)$', fontsize = label_size)
figure_ax32.set_ylabel('Diameter (km)', color='blue', fontsize = label_size, labelpad=-55)
figure_ax3.set_ylabel('Temperature (K)', color='red', fontsize = label_size)
figure_ax3.set_yticks([2000,2050,2100,2150,2200])
figure_ax32.set_yticks([100,125,150,175,200])
figure_ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='--', color='red', colors='red')
figure_ax32.tick_params(axis='y', labelrotation=0, grid_linestyle='--', color='blue', colors='blue')
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax32.tick_params(labelsize = ticks_size)
figure_ax3.grid(which='major')
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_off()
figure_ax32.minorticks_off()

figure_ax4.set_xlabel('Diameter (km)', fontsize = label_size)
figure_ax4.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
figure_ax4.tick_params(labelsize = ticks_size)
figure_ax4.minorticks_on()


  # Plot colorbar
cbar = figure.colorbar(img1, cax=figure_ax5, orientation = 'vertical')
cbar.set_label(r'$log_{10}(\overline{Q_{2D}})$ $(m^2/s)$', fontsize=label_size, rotation=-90, labelpad=20)
cbar.ax.tick_params(labelsize = ticks_size)
cbar.minorticks_on()

  # Plot colorbar
cbar = figure.colorbar(img4, cax=figure_ax6, orientation = 'vertical')
cbar.set_label(r'$log_{10}(\overline{Q_{3D}})$ $(m^3/s)$', fontsize=label_size, rotation=-90, labelpad=26)
cbar.ax.tick_params(labelsize = ticks_size)
cbar.minorticks_on()
  
  # Displays the drawing results
plt.savefig("./figure_12.png", dpi=300, format="png")