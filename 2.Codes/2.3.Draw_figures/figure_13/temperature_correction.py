################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import jn, struve, jn_zeros
from cmcrameri import cm
import math

ticks_size = 13
label_size = 15

# Create a canvas
figure = plt.figure(figsize=(8,18), dpi=300)
# Create a layer
figure_ax1 = figure.add_axes([0.11,0.70,0.80,0.27]) #Sets the position of the layer on the canvas
figure_ax2 = figure.add_axes([0.11,0.36,0.75,0.28]) #Sets the position of the layer on the canvas
figure_cbar = figure.add_axes([0.88,0.36,0.03,0.28])
figure_ax3 = figure.add_axes([0.11,0.04,0.80,0.27])
figure_ax1.annotate('(a)', xy=(-0.03,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax2.annotate('(b)', xy=(-0.03,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
figure_ax3.annotate('(c)', xy=(-0.03,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")

### figure a ###
Model_Name1 = ['D100', 'D125', 'RefModel', 'D175', 'D200']
Model_Name2 = ['T2000', 'T2050', 'RefModel ', 'T2150', 'T2200']

opt_i = [25,24,17,18,18,14,15,24,27] # 1-10Myr
opt_j = [6,7,12,14,16,9,11,11,11] # 1-10Myr
T_2D_Opt1 = np.array([2200,2190,2120,2130,2130])
T_2D_Opt2 = np.array([2090,2100,2120,2190,2220])

t_start = 1; t_end = 5
k = 1.3E-6
dl = (660*0.8-140)*1000
save_flux_3D = 1000 * 1000 * np.load(file="3D_flux.npy")
Q3D = np.sum(save_flux_3D[:,t_start:t_end+1],1)/(t_end-t_start+1)
save_flux_2D = 1000 * np.load(file="2D_flux.npy")
Q2D=(np.sum(save_flux_2D[opt_i,opt_j,t_start:t_end+1],1)/(t_end-t_start+1))

T3D = np.array([2100,2100,2100,2100,2100,2000,2050,2150,2200]) - 1832
T_3D1 = np.array([2100,2100,2100,2100,2100])
T_3D2 = np.array([2000,2050,2100,2150,2200])
R2D = np.array([35,40,65,80,85,45,65,60,60]) * 500
R3D = np.array([100,125,150,175,200,150,150,150,150]) * 500

A = np.zeros([9])
B = np.zeros([9])

t2D=dl/(Q2D/(R2D*2.0))
t3D=dl/(Q3D/(R3D*R3D*np.pi))

print(t2D/(1.0E+6 * 365.25 * 24 * 3600))
print(t3D/(1.0E+6 * 365.25 * 24 * 3600))

num_Fourier = 100
for _ in range(1,num_Fourier,2):
  A += 8.0/np.pi/np.pi * (np.exp(-k * np.pi * np.pi * _ * _ / 4.0 / R2D / R2D * t2D)) / _ / _

num_Bessel = 30
Bessel_Zero = jn_zeros(0, num_Bessel)
for _ in range(1,num_Bessel,1):
  B += np.pi/Bessel_Zero[_-1] * np.exp(-k * Bessel_Zero[_-1] * Bessel_Zero[_-1] / R3D / R3D * t3D)*struve(0,Bessel_Zero[_-1])


T_2D_Prediction = B/A*T3D + 1832
T_2D_Prediction1 = T_2D_Prediction[0:5]
T_2D_Prediction2 = np.array([T_2D_Prediction[5], T_2D_Prediction[6], T_2D_Prediction[2], T_2D_Prediction[7], T_2D_Prediction[8]])

# plot
img11,=figure_ax1.plot(Model_Name1,T_3D1, color='red',marker='o', markersize=7, linestyle=':', linewidth=2.0)
img11,=figure_ax1.plot(Model_Name2,T_3D2, color='red',marker='o', markersize=7, linestyle=':', linewidth=2.0)
img12,=figure_ax1.plot(Model_Name1,T_2D_Opt1, color='blue',marker='s', markersize=6, linestyle='--', linewidth=2.0)
img12,=figure_ax1.plot(Model_Name2,T_2D_Opt2, color='blue',marker='s', markersize=6, linestyle='--', linewidth=2.0)
img13=figure_ax1.scatter(Model_Name1,T_2D_Prediction1, color='black',marker='x', s = 150, linewidth=2.0)
img13=figure_ax1.scatter(Model_Name2,T_2D_Prediction2, color='black',marker='x', s = 150, linewidth=2.0)
figure_ax1.legend([img11,img12,img13], [r'$T_{3D}$ Tail Models', r'$T_{2D}$ Opt_Model', r'$T_{2D}$ Analytical Results'], handlelength=4,loc='lower left', fontsize=ticks_size)
figure_ax1.set_ylim([1970,2280])

### figure b ###
t = 1.0* 1.0E+6 * 365.25 * 24 * 3600
k = 1.3E-6
t2D = t
t3D = t
epsilon = 1.0

r1 = np.linspace(10,220,1000) * 1000
R2D, R3D = np.meshgrid(r1,r1)

A = np.zeros([1000,1000])
B = np.zeros([1000,1000])

num_Fourier = 200
for _ in range(1,num_Fourier,2):
  A += 8.0/np.pi/np.pi * (np.exp(-k * np.pi * np.pi * _ * _ / 4.0 / R2D / R2D * t2D)) / _ / _

num_Bessel = 30
Bessel_Zero = jn_zeros(0, num_Bessel)
for _ in range(1,num_Bessel,1):
  B += np.pi/Bessel_Zero[_-1] * np.exp(-k * Bessel_Zero[_-1] * Bessel_Zero[_-1] / R3D / R3D * t3D)*struve(0,Bessel_Zero[_-1])

cm_colorbar = cm.broc
img21 = figure_ax2.pcolormesh(R2D/500,R3D/500,B/A,vmax = 1.5, vmin=0.5,cmap=cm_colorbar)
img22 = figure_ax2.scatter(np.array([120,130,160,220,270,170,170,170,170]) * epsilon ,np.array([125,150,200,300,400,200,200,200,200]) * epsilon, c='red',marker='+', s = 150, linewidth=2)
img23 = figure_ax2.scatter([35,40,65,80,85,45,65,60,60],[100,125,150,175,200,150,150,150,150], c='blue',marker='x', s = 150, linewidth=2)
img24_contour = figure_ax2.contour(R2D/500,R3D/500,(B/A), [0.8,0.9,0.95,1.0,1.05,1.1,1.2], colors = 'black', linewidths=2, linestyles='--')
figure_ax1.clabel(img24_contour, img24_contour.levels, inline_spacing=10, inline = True, fontsize = label_size)
h24,ll=img24_contour.legend_elements()
figure_ax2.legend([img22, img23, h24[0]], ['Head-Only Models', 'Tail Models', r'$\left( \right. \Delta T_{2D}$ / $ \Delta T_{3D} \left. \right)$ Contour' ], handlelength=5, loc='lower center', fontsize=ticks_size)
cbar = figure.colorbar(img21, cax=figure_cbar, orientation = 'vertical')
cbar.set_label(r'$\left( \right. \Delta T_{2D}$ / $ \Delta T_{3D} \left. \right)$', fontsize=label_size, rotation=-90, labelpad=18)
cbar.ax.tick_params(labelsize = ticks_size)


### figure c ###
T0 = 1799
model_name = ['T_2000', 'T_2050', 'Ref_Model', 'T_2150', 'T_2200']
T_3D = np.array([2000,2050,2100,2150,2200]) - T0
T_2D = np.array([2000,2040,2080,2120,2140]) - T0
T_3D_drawLine = np.array([170,420])

# fitting curve
def func_res4(para, sign=1.0):
  return np.sum((T_3D * para[0] - T_2D)**2)
para4 = [0.9003]
err4 = minimize(func_res4, para4, args=(1,), method='Nelder-Mead',tol=1.0E-10)
print("minimize fitting d: ",err4)

# calculate R2
fit_line_4 = err4.x[0] * T_3D
SSR = np.sum((fit_line_4 - np.mean(T_2D))**2)
SSE = np.sum((fit_line_4 - T_2D)**2)
SST = SSR + SSE
r_squared = 1 - (SSE / SST)
print(r"$R^2$: ", r_squared)

# plot
fit_line_4 = err4.x[0] * T_3D_drawLine
img31=figure_ax3.scatter(T_3D, T_2D, color='black',marker='+', s = 200, linewidth=2)
for i, name in enumerate(model_name):
  figure_ax3.annotate(name, (T_3D[i],T_2D[i]), textcoords='offset points', xytext=(20,-20), ha='left', fontsize=ticks_size)

img32,=figure_ax3.plot(T_3D_drawLine, fit_line_4, color='black', linestyle='dashdot')
figure_ax3.text(180, 270, f'Slope: {err4.x[0]:.3e}\n$R^2$: {r_squared:.4f}',style = 'oblique', fontsize = label_size, color = "black")
figure_ax3.legend([img31,img32], ['Head-Only Model Results', 'Fit Line: y = ax (none intercept)'], handlelength=4,loc='upper left', fontsize=ticks_size)
figure_ax3.set_xlim([150,450])
figure_ax3.set_ylim([130,390])


figure_ax1.set_xlabel('Model Name', fontsize = label_size, labelpad=-5)
figure_ax1.set_ylabel('Temperature (K)', fontsize = label_size)
figure_ax1.tick_params(labelsize = ticks_size)
figure_ax1.grid(which='major', linestyle='--')
figure_ax1.tick_params(axis='x', labelrotation=-22.8, grid_linestyle='-')
figure_ax1.tick_params(axis='y', labelrotation=0)
figure_ax1.set_axisbelow(True)
figure_ax1.minorticks_off()

figure_ax2.set_xlabel(r'$d_{2D}$ (km)', fontsize = label_size)
figure_ax2.set_ylabel(r'$d_{3D}$ (km)', fontsize = label_size)
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.minorticks_on()

figure_ax3.set_xlabel(r'$T_{3D} - T_0$ (K)', fontsize = label_size)
figure_ax3.set_ylabel(r'$T_{2D} - T_0$ (K)', fontsize = label_size)
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax3.grid(which='major', linestyle='--')
figure_ax3.tick_params(axis='x', labelrotation=0)
figure_ax3.tick_params(axis='y', labelrotation=0)
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_on()


# Displays the drawing results
plt.savefig("./figure_13.png", dpi=300, format="png")