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
label_size = 16

# Create a canvas
figure = plt.figure(figsize=(15,5), dpi=300)
# Create a layer
figure_ax3 = figure.add_axes([0.01,0.03,0.57,0.90])
figure_ax4 = figure.add_axes([0.65,0.14,0.33,0.78])
figure_ax3.annotate('(a) Model with a Continous Plume Tail', xy=(-0.00,1.03), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax4.annotate('(b)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, va="center")

### figure c ###

Senario_2 = plt.imread('./Scenario_2.tif')

img3 = figure_ax3.imshow(Senario_2[:,:])
figure_ax3.get_xaxis().set_visible(False)
figure_ax3.get_yaxis().set_visible(False)
figure_ax3.spines[:].set_visible(False)


### figure d ###
model_name = ['D100', 'D125', 'RefModel', 'D175', 'D200', 'T2000', 'T2050', 'T2150', 'T2200']
opt_i = [21,21,17,16,17,18,13,24,26] # 5Myr
opt_j = [6,8,12,15,16,8,12,11,11] # 5Myr
t_start = 1; t_end = 5
save_flux_3D = 1000 * 1000 * np.load(file="3D_flux.npy")
q_3D = np.sum(save_flux_3D[:,t_start:t_end+1],1)/(t_end-t_start+1)
q_3D_drawLine = np.array([30,20000])
save_flux_2D = 1000 * np.load(file="2D_flux.npy")
q_2D__2=(np.sum(save_flux_2D[opt_i,opt_j,t_start:t_end+1],1)/(t_end-t_start+1))**2
print(q_2D__2)
print(q_3D)
# fitting curve
def func_res2(para, sign=1.0):
  return np.sum(((para[0] * q_3D - q_2D__2) / q_2D__2)**2)
para2 = [1.816e-09]
err2 = minimize(func_res2, para2, args=(1,), method='Nelder-Mead', tol=1.0E-6)
print("minimize fitting b: ",err2)

#calculate R2
fit_line_2 = err2.x[0] * q_3D
SSR = np.sum((fit_line_2 - np.mean(q_2D__2))**2)
SSE = np.sum((fit_line_2 - q_2D__2)**2)
SST = SSR + SSE
r_squared = 1 - (SSE / SST)
print(r"$R^2$: ", r_squared)

fit_line_2 = err2.x[0] * q_3D_drawLine
img42,=figure_ax4.loglog(q_3D_drawLine, fit_line_2, color='black', linestyle='dashdot')
img41=figure_ax4.scatter(q_3D, q_2D__2, color='black',marker='+', s = 200, linewidth=2)
for i, name in enumerate(model_name):
  figure_ax4.annotate(name, (q_3D[i],q_2D__2[i]), textcoords='offset points', xytext=(10,-10), ha='left', fontsize=ticks_size)

figure_ax4.text(50, 1.0E-6, f'Slope: {err2.x[0]:.3e}\n$R^2$: {r_squared:.4f}',style = 'oblique', fontsize = label_size, color = "black")
figure_ax4.legend([img41,img42], ['Tail Model Results', 'Fit Line: y = ax (none intercept)'], handlelength=4,loc='upper left', fontsize=ticks_size)
figure_ax4.set_xlim([30,20000])
figure_ax4.set_ylim([5.0E-8,3.0E-5]) # The y-axis is reversed


figure_ax4.set_xlabel(r'$\overline{Q_{3D}}$ $(m^3 s^{-1})$', fontsize = label_size)
figure_ax4.set_ylabel(r'$\overline{Q_{2D}}^2$ $(m^4 s^{-2})$', fontsize = label_size)
figure_ax4.tick_params(labelsize = ticks_size)
figure_ax4.grid(which='both', linestyle=':')
figure_ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
figure_ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
figure_ax4.set_axisbelow(True)
figure_ax4.minorticks_on()

plt.savefig("./figure_11.png", dpi=300, format="png")