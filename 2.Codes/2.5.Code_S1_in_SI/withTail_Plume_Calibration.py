################################################################################
## File Name: withTail_Plume_Calibration.py                                   ##
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
## Purpose:                                                                   ##
## This python script is used for directly calculating the corresponding      ##
## 2D plume parameters for approximating a given 3D plume evolution with      ##
## a continuous tail.                                                         ##
## Output: ./Calibration.png                                                  ##
################################################################################

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import jn, struve, jn_zeros

temp   = np.array(["1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0"])
size   = np.array(["5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00"])
print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp)
num_j = len(size);   j_start = 3;  j_end = len(size)-3
num_time = 21;       t_start = 1;  t_end = 10


###################################################################################
## load data: Q_2D(diam,temp) && Q_3D(diam,temp) from numerical modeling results ##
###################################################################################
save_flux_2D = 1000 * np.load(file="./2D_flux.npy")
save_flux_2D[save_flux_2D==np.inf]=-np.inf
for _ in range(t_start,t_end+1):
  max_in_row_index = np.argmax(save_flux_2D[:,:,_], axis=1)
  for __ in range(0,len(temp)):
    save_flux_2D[__,max_in_row_index[__]:,_]=np.inf
save_flux_2D[save_flux_2D==-np.inf]=np.inf
save_flux_3D = 1000 * 1000 * np.load(file="./3D_flux.npy")
save_flux_3D = np.log10(np.sum(save_flux_3D[:,t_start:t_end+1],1)/(t_end-t_start+1))


#####################################################################
## Fit Q_2D = A(d_2D)^m * (T_2D)^n from numerical modeling results ##
## Equation 11 --> 13                                              ##
#####################################################################
X, Y = np.meshgrid(np.float64(size[j_start:j_end])/1000, np.float64(temp[i_start:i_end]))
figure = plt.figure(figsize=(15,12), dpi=300)
figure_ax1 = figure.add_axes([0.07,0.55,0.36,0.40]) #Sets the position of the layer on the canvas
LL=[-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,-3.0,-2.9,-2.8]
img1_contour=figure_ax1.contour(X,Y,np.log10(np.sum(save_flux_2D[i_start:i_end,j_start:j_end,t_start:t_end+1],2)/(t_end-t_start+1)), LL)
h11,ll=img1_contour.legend_elements()
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

def func_res1(para, sign=1.0):
  r=0
  for _ in range(1,10):
    r = r + (para[1] * np.log10(ff[_](2100)) + np.log10(para[0]) - LL[_])**2
  return r
para1=[2.278e-15, 2.409e+00]
err1 = minimize(func_res1, para1, args=(1,), method='Nelder-Mead',tol=1.0e-20)

#####################################################################
## Fit Q_3D = B(d_3D)^i * (T_3D)^j from numerical modeling results ##
## Equation 12 --> 14                                              ##
#####################################################################
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


###########################################################
## Fit Q_2D^2 = C*(Q_3D) from numerical modeling results ##
## Equation 10                                           ##
###########################################################
opt_i = [21,21,17,16,17,18,13,24,26] # Opt_model at 5Myr
opt_j = [6,8,12,15,16,8,12,11,11] # Opt_model at 5Myr
t_start = 1; t_end = 5
save_flux_3D = 1000 * 1000 * np.load(file="./3D_flux.npy")
q_3D = np.sum(save_flux_3D[:,t_start:t_end+1],1)/(t_end-t_start+1)
save_flux_2D = 1000 * np.load(file="./2D_flux.npy")
q_2D__2=(np.sum(save_flux_2D[opt_i,opt_j,t_start:t_end+1],1)/(t_end-t_start+1))**2
# fitting curve
def func_res2(para, sign=1.0):
  return np.sum(((para[0] * q_3D - q_2D__2) / q_2D__2)**2)
para2 = [1.816e-09]
err2 = minimize(func_res2, para2, args=(1,), method='Nelder-Mead', tol=1.0E-6)


###########################################################
## Calculation of 2D parameters corresponding to a given ##
## diameter and temperature of 3D mantle plume.          ##
## (d_3D, T_3D) --> (d_2D, T_2D)                         ##
## Equation 13 && Equation 16 (iteration)                ##
###########################################################


print("\n>>> Preparations Completed. >>>")
print("\nStep 0: input parameter")

print("\nFor mantle plume with a continuous tail: ")
Diam_3d = np.float64(input('The diameter of 3D mantle plume (d_3D) is: ____ (km) ')) * 1000.0
Temp_3d = np.float64(input('The initial temperature anomaly of mantle plume (ΔT_3D) is: ____ (K) '))

Q3D = err4.x[0]*Diam_3d**err4.x[1]*(Temp_3d)**err4.x[2]
print("\nStep 1: From equation 14, the 3D volume flux (Q_3D) is: ", f'{Q3D:.3e}', "(m^3/s) ")
Q2D = np.sqrt(err2.x[0] * Q3D)
print("\nStep 2: From equation 10, the 2D volume flux (Q_2D) is: ", f'{Q2D:.3e}', "(m^2/s) ")

print("\nStep 3: d_2d and ΔT_2d should satisfied Equations 13 and 16. ")
print(">>> Solving Equations 13 and 16 >>> \n")

k = 1.3E-6
dl = (660*0.8-140)*1000
t3D=dl/(Q3D/(Diam_3d*Diam_3d*np.pi/4.0))

B = 0
num_Bessel = 30
Bessel_Zero = jn_zeros(0, num_Bessel)
for _ in range(1,num_Bessel,1):
  B += np.pi/Bessel_Zero[_-1] * np.exp(-k * Bessel_Zero[_-1] * Bessel_Zero[_-1] * 4.0 / Diam_3d / Diam_3d * t3D)*struve(0,Bessel_Zero[_-1])


def func_res5(para, sign=1.0):
  t2D=dl/(Q2D/(para[0]))
  A = 0
  num_Fourier = 100
  for _ in range(1,num_Fourier,2):
    A += 8.0/np.pi/np.pi * (np.exp(-k * np.pi * np.pi * _ * _ / para[0] / para[0] * t2D)) / _ / _
  T_2D_Prediction_1 = B/A*Temp_3d
  T_2D_Prediction_2 = ((Q2D / err1.x[0])**(1.0 / err1.x[1]) / err0.x[0] / (para[0])**err0.x[1])**(1.0/err0.x[2])
  return (T_2D_Prediction_1 - T_2D_Prediction_2)**2
para5 = [Diam_3d * 0.5]
err5 = minimize(func_res5, para5, args=(1,), method='Nelder-Mead', tol=1.0E-8)
t2D=dl/(Q2D/(err5.x[0]))
A = 0
num_Fourier = 100
for _ in range(1,num_Fourier,2):
  A += 8.0/np.pi/np.pi * (np.exp(-k * np.pi * np.pi * _ * _ / err5.x[0] / err5.x[0] * t2D)) / _ / _
T_2D_Prediction = B/A*Temp_3d

if err5.x[0]/1000 < 15 or err5.x[0]/1000 > 250 or T_2D_Prediction < 30 or T_2D_Prediction > 500: 
  print("\n<<< Error! The calculation did not converge. Suggestions: d_3D in [125,400] km, ΔT_3D in [50,400] K <<<\n")
else: 
  print(err5)
  print("\n<<< Solving Equations 13 and 16 Finished! <<<")
  print("\nFor 3D mantle plume with a continuous tail: d_3D = ", f'{Diam_3d/1000:.2f} (km), ', "ΔT_3D = ", f'{Temp_3d:.2f} (K);')
  print("The optimal 2D mantle plume: d_2D = ", f'{err5.x[0]/1000:.2f} (km), ', "ΔT_3D = ", f'{T_2D_Prediction:.2f} (K);') 

###########################################################
## Drawing Illustration of Calibration                   ##
###########################################################
  ticks_size = 13
  label_size = 15
  figure = plt.figure(figsize=(10,8), dpi=300)
  figure_ax1 = figure.add_axes([0.08,0.1,0.78,0.85]) #Sets the position of the layer on the canvas
  figure_ax2 = figure.add_axes([0.88,0.1,0.03,0.85]) #Sets the position of the layer on the canvas

  cm_colorbar = 'viridis'
  vmax = -1.0
  vmin = -5.0
  X1,Y1 = np.meshgrid(np.linspace(20,300,1000), np.linspace(1860,2300,1000))
  img1=figure_ax1.pcolormesh(X1,Y1-1832, err1.x[1] * np.log10(err0.x[0] * (X1*1000)**err0.x[1] * (Y1 - 1832)**err0.x[2]) + np.log10(err1.x[0]) ,cmap=cm_colorbar, vmin = vmin, vmax = vmax)
  img1_contour1 = figure_ax1.contour(X1,Y1-1832,err1.x[1] * np.log10(err0.x[0] * (X1*1000)**err0.x[1] * (Y1 - 1832)**err0.x[2]) + np.log10(err1.x[0]), colors = 'black', linewidths=2.0, linestyles='dashed')
  img1_contour2 = figure_ax1.contour(X1,Y1-1832,err1.x[1] * np.log10(err0.x[0] * (X1*1000)**err0.x[1] * (Y1 - 1832)**err0.x[2]) + np.log10(err1.x[0]),[np.log10(Q2D)], colors = 'tomato', linewidths=3.0, linestyles='-')
  figure_ax1.clabel(img1_contour2, img1_contour2.levels, inline_spacing=10, inline = True, fontsize = label_size)
  h21,ll=img1_contour1.legend_elements()
  h22,ll=img1_contour2.legend_elements()

  img12 = figure_ax1.scatter(Diam_3d/1000, Temp_3d, c='black', marker='*', s = 500, linewidth=1.0, edgecolors='white')
  img13 = figure_ax1.scatter(err5.x[0]/1000,T_2D_Prediction,c='none', marker='s', s = 300, linewidth=3.0, edgecolors='orangered')
  figure_ax1.annotate(f'3D Model:\n' + r'$d_{3D}=$ ' + f'{Diam_3d/1000:.1f} (km)\n' + r'$\Delta T_{3D}=$ ' + f'{Temp_3d:.1f} (K)', (Diam_3d/1000,Temp_3d), textcoords='offset points', xytext=(0,-20), ha='center',va='top', fontsize=ticks_size-3, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5, lw=1.0))
  figure_ax1.annotate(f'Optimal 2D Model:\n' + r'$d_{2D}=$ ' + f'{err5.x[0]/1000:.1f} (km)\n' + r'$\Delta T_{2D}=$' + f'{T_2D_Prediction:.1f} (K)', (err5.x[0]/1000,T_2D_Prediction), textcoords='offset points', xytext=(0,-20), ha='center',va='top', fontsize=ticks_size-3, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5, lw=1.0))

  figure_ax1.legend([h21[0], h22[0], img12, img13], [r'$Q_{2D}$ Contour: Analytical Predictions' ,r'$Q_{2D}$ Contour: Optimizing 2D Models', '3D Model', 'Optimal 2D Model'], loc='upper right', fontsize=ticks_size)

  figure_ax1.set_xlim([20,300])
  figure_ax1.set_ylim([30,460])

  figure_ax1.set_xlabel(r'Diameter of Plume Tail $d_{2D}$ (km)', fontsize = label_size)
  figure_ax1.set_ylabel(r'Initial Temperature Anomaly of Mantle Plume $\Delta T_{2D}$ (K)', fontsize = label_size)
  figure_ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
  figure_ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
  figure_ax1.tick_params(labelsize = ticks_size)
  figure_ax1.minorticks_on()

  cbar = figure.colorbar(img1, cax=figure_ax2, orientation = 'vertical',extend='both')
  cbar.set_label(r'$log_{10}(Q_{2D})$ $(m^2/s)$', fontsize=label_size, rotation=-90, labelpad=20)
  cbar.ax.tick_params(labelsize = ticks_size)
  cbar.minorticks_on()

  plt.savefig("./Calibration.png", dpi=300, format="png")