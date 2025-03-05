################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.special import jn, struve, jn_zeros
from cmcrameri import cm
import math

ticks_size = 13
label_size = 16

# Create a canvas
figure = plt.figure(figsize=(12,12), dpi=300)
# Create a layer
figure_ax0 = figure.add_axes([0.00,0.02,1.00,0.97]) #Sets the position of the layer on the canvas
figure_ax1 = figure.add_axes([0.05,0.55,0.90,0.40]) 
figure_ax2 = figure.add_axes([0.06,0.07,0.40,0.40]) 
figure_ax3 = figure.add_axes([0.57,0.07,0.40,0.40]) 
figure_ax1.annotate('(a) Model with Only a Plume Head', xy=(-0.00,1.05), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax2.annotate('(b)', xy=(-0.05,1.05), xycoords="axes fraction", fontsize=label_size, va="center")
figure_ax3.annotate('(c)', xy=(-0.05,1.05), xycoords="axes fraction", fontsize=label_size, va="center")
### figure backgroud ###

figure_ax0.set_xlim([0,1.0])
figure_ax0.set_ylim([0,1.0])
figure_ax0.plot([0.0,1.0],[0.52,0.52],linewidth=2.0,c='black',linestyle='dashed')
figure_ax0.set_xticks([])
figure_ax0.set_yticks([])
figure_ax0.axis('off')

### figure a ###

Senario_1 = plt.imread('./Scenario_1.tif')

img1 = figure_ax1.imshow(Senario_1[:,:])
figure_ax1.get_xaxis().set_visible(False)
figure_ax1.get_yaxis().set_visible(False)
figure_ax1.spines[:].set_visible(False)

### figure b ###
model_name = ['D_125', 'D_150', 'Ref_Model', 'D_300', 'D_400']
d_3D = np.array([125.0, 150.0, 200.0, 300.0, 400.0])
d_2D = np.array([120.0, 130.0, 160.0, 220.0, 270.0])
d_2D_drawLine = np.array([100,300])
d3D_d2D__3 = (d_3D/d_2D)**3

# fitting curve
def func_res1(para, sign=1.0):
  return np.sum(((para[0] * d_2D + para[1] - d3D_d2D__3)/d3D_d2D__3)**2)
para1 = [1.299e-02,-2.562e-01]
err1 = minimize(func_res1, para1, args=(1,), method='Nelder-Mead')
print("minimize fitting a: ",err1)

# calculate R2
fit_line_1 = err1.x[0] * d_2D + err1.x[1]
SSR = np.sum((fit_line_1 - np.mean(d3D_d2D__3))**2)
SSE = np.sum((fit_line_1 - d3D_d2D__3)**2)
SST = SSR + SSE
r_squared = 1 - (SSE / SST)
print(r"$R^2$: ", r_squared)

# plot
fit_line_1 = err1.x[0] * d_2D_drawLine + err1.x[1]
img21=figure_ax2.scatter(d_2D, d3D_d2D__3, color='black',marker='+', s = 200, linewidth=2)
img22,=figure_ax2.plot(d_2D_drawLine, fit_line_1, color='black', linestyle='dashdot')
for i, name in enumerate(model_name):
  figure_ax2.annotate(name, (d_2D[i],d3D_d2D__3[i]), textcoords='offset points', xytext=(10,-10), ha='left', fontsize=ticks_size)

figure_ax2.text(120, 2.4, f'Slope: {err1.x[0]:.3e}\nIntercept: {err1.x[1]:.3e}\n$R^2$: {r_squared:.4f}',style = 'oblique', fontsize = label_size, color = "black")
figure_ax2.legend([img21,img22], ['Head-Only Model Results', 'Fit Line: y = ax + b (linear)'], handlelength=4,loc='upper left', fontsize=ticks_size)
figure_ax2.set_xlim([100,300])
figure_ax2.set_ylim([0.5,4.0])

### figure c ###

model_name = ['D_125', 'D_150', 'Ref_Model', 'D_300', 'D_400']
d_3D = np.array([125.0, 150.0, 200.0, 300.0, 400.0])
d_2D = np.array([120.0, 130.0, 160.0, 220.0, 270.0])
d_2D_drawLine = np.linspace(95,320,500)
d3D_d2D__3 = (d_3D/d_2D)**3

# fitting curve
def func_res1(para, sign=1.0):
  return np.sum(((para[0] * d_2D + para[1] - d3D_d2D__3)/d3D_d2D__3)**2)
para1 = [1.299e-02,-2.562e-01]
err1 = minimize(func_res1, para1, args=(1,), method='Nelder-Mead')
print("minimize fitting a: ",err1)

# calculate R2
fit_line_1 = err1.x[0] * d_2D + err1.x[1]
SSR = np.sum((fit_line_1 - np.mean(d3D_d2D__3))**2)
SSE = np.sum((fit_line_1 - d3D_d2D__3)**2)
SST = SSR + SSE
r_squared = 1 - (SSE / SST)
print(r"$R^2$: ", r_squared)

# plot
d_3D_drawLine = (err1.x[0] * d_2D_drawLine ** 4 + err1.x[1] * d_2D_drawLine ** 3)**(1.0/3.0)
img10,=figure_ax3.plot(d_3D_drawLine, d_3D_drawLine, color='black', linestyle='dotted', linewidth=2)
img11=figure_ax3.scatter(d_3D, d_2D, color='black',marker='+', s = 200, linewidth=2)
img12,=figure_ax3.plot(d_3D_drawLine, d_2D_drawLine, color='black', linestyle='dashdot', linewidth=2)
for i, name in enumerate(model_name):
  figure_ax3.annotate(name, (d_3D[i],d_2D[i]), textcoords='offset points', xytext=(10,-10), ha='left', fontsize=ticks_size)
img13=figure_ax3.scatter(158, 141, color='blue',marker='o', s = 120, linewidth=2)


x_tail = 340
y_tail = 330
x_head = 340
y_head = 250
dx = x_head - x_tail
dy = y_head - y_tail
arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), mutation_scale=30, color='gray')
figure_ax3.add_patch(arrow)
figure_ax3.annotate(' Diameter \nCalibration', (355,290), textcoords='offset points', xytext=(0,0), ha='left', fontsize=ticks_size, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7, lw=2.0))
figure_ax3.legend([img10,img11,img12,img13], [r'Same Diameter Line: $d_{2D}=d_{3D}$', 'Head-Only Model Results', r'Analytical Predictions','Burov and Gerya (2014)'], handlelength=4,loc='upper left', fontsize=ticks_size)
figure_ax3.set_xlim([50,550])
figure_ax3.set_ylim([70,450])


################
figure_ax2.set_xlabel(r'$d_{2D}$ (km)', fontsize = label_size)
figure_ax2.set_ylabel(r'$\left(d_{3D}\ /\ d_{2D}\right)^3$', fontsize = label_size)
figure_ax2.tick_params(labelsize = ticks_size)
figure_ax2.grid(which='major', linestyle='--')
figure_ax2.tick_params(axis='x', labelrotation=0)
figure_ax2.tick_params(axis='y', labelrotation=0)
figure_ax2.set_axisbelow(True)
figure_ax2.minorticks_on()

figure_ax3.set_xlabel(r'$d_{3D}$ (km)', fontsize = label_size)
figure_ax3.set_ylabel(r'$d_{2D}$ (km)', fontsize = label_size)
figure_ax3.tick_params(labelsize = ticks_size)
figure_ax3.grid(which='major', linestyle='--')
figure_ax3.tick_params(axis='x', labelrotation=0)
figure_ax3.tick_params(axis='y', labelrotation=0)
figure_ax3.set_axisbelow(True)
figure_ax3.minorticks_on()

# Displays the drawing results

plt.savefig("./figure_10.png", dpi=300, format="png")
#plt.ioff() # Open the interactive drawing. If closed, the program pauses while drawing 