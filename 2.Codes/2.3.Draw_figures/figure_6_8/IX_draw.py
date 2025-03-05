################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import math 
import os
from cmcrameri import cm

temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00")

i = 30
j = 5
print(temp[i])
print(size[j])

# Plot parameters

ticks_size = 12
label_size = 15
figure = plt.figure(figsize=(15,9), dpi=300)

cm_cmap_T = cm.vik
cm_cmap_eta = cm.cork_r

ax1 = figure.add_axes([0.06,0.57,0.26,0.4])
ax1.annotate('(a)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax2 = figure.add_axes([0.35,0.57,0.26,0.4])
ax2.annotate('(b)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax3 = figure.add_axes([0.64,0.57,0.26,0.4])
ax3.annotate('(c)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax4 = figure.add_axes([0.925,0.57,0.02,0.4])

ax5 = figure.add_axes([0.06,0.12,0.26,0.4])
ax5.annotate('(d)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax6 = figure.add_axes([0.35,0.12,0.26,0.4])
ax6.annotate('(e)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax7 = figure.add_axes([0.64,0.12,0.26,0.4])
ax7.annotate('(f)', xy=(-0.05,1.03), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax8 = figure.add_axes([0.925,0.12,0.02,0.4])
ax9 = figure.add_axes([0.04,0.0,0.94,0.05])


X_length = 4000
Y_length = 660
Z_length = 4000
I3_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I3_Plume/D100/"
I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I2_Plume/test_" + str(j  + i * len(size)) + "/"


################## 
## figure (a)
file_num = 2
file_type = 't'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))


# Set axis parameters
ax1.set_xlim([0,X_length])
ax1.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax1.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img1 = ax1.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],I2_data[0:half_column,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
img1 = ax1.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
ax1.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax1.text(800, -600, '2D', fontsize = label_size, color = "black")
ax1.text(2700, -600, '3D Slice', fontsize = label_size, color = "black")
ax1.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")


################## 
## figure (b)
file_num = 5
file_type = 't'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))


# Set axis parameters
ax2.set_xlim([0,X_length])
ax2.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax2.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img2 = ax2.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],I2_data[0:half_column,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
img2 = ax2.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
ax2.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax2.text(800, -600, '2D',  fontsize = label_size, color = "black")
ax2.text(2700, -600, '3D Slice',  fontsize = label_size, color = "black")
ax2.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")


################## 
## figure (c)
file_num = 10
file_type = 't'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))


# Set axis parameters
ax3.set_xlim([0,X_length])
ax3.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax3.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img3 = ax3.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],I2_data[0:half_column,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
img3 = ax3.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_T,vmin = 300, vmax = 2200)
ax3.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax3.text(800, -600, '2D',  fontsize = label_size, color = "black")
ax3.text(2700, -600, '3D Slice',  fontsize = label_size, color = "black")
ax3.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")


################## 
## figure (d)
file_num = 2
file_type = 'v'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

# Set axis parameters
ax5.set_xlim([0,X_length])
ax5.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax5.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img5 = ax5.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],np.log10(I2_data[0:half_column,]),cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
img5 = ax5.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
ax5.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax5.text(800, -600, '2D',  fontsize = label_size, color = "black")
ax5.text(2700, -600, '3D Slice',  fontsize = label_size, color = "black")
ax5.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")


################## 
## figure (e)
file_num = 5
file_type = 'v'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

# Set axis parameters
ax6.set_xlim([0,X_length])
ax6.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax6.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img6 = ax6.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],np.log10(I2_data[0:half_column,]),cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
img6 = ax6.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
ax6.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax6.text(800, -600, '2D',  fontsize = label_size, color = "black")
ax6.text(2700, -600, '3D Slice',  fontsize = label_size, color = "black")
ax6.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")



################## 
## figure (f)
file_num = 10
file_type = 'v'
file_name = file_type + '%03d' % file_num

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       =  I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

# Set axis parameters
ax7.set_xlim([0,X_length])
ax7.set_ylim([-Y_length,-10]) # The y-axis is reversed
ax7.tick_params(labelsize = ticks_size)

# Plot the temperature field
half_column = int(np.floor(grid[0]/2))
img5 = ax7.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],np.log10(I2_data[0:half_column,]),cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
img5 = ax7.pcolor(I3_X[half_column:,],I3_Y[half_column:,],I3_data[half_column:,],cmap = cm_cmap_eta,vmin = 17.8, vmax = 24.2)
ax7.vlines([2000], 0, -660, linestyles='dashed', colors='black')
ax7.text(800, -600, '2D',  fontsize = label_size, color = "black")
ax7.text(2700, -600, '3D Slice',  fontsize = label_size, color = "black")
ax7.text(200, -640, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")

# Plot Time arrow
ax9.set_xlim([0,1.0])
ax9.set_ylim([0,1.0])
ax9.arrow(0.0,0.75,1.0,0.0, width=0.08, head_width=0.4, head_length=0.03, length_includes_head=True, fc='k', ec='k')
ax9.text(0.154, 0.1, '2', fontsize = label_size, color = "black")
ax9.text(0.463, 0.1, '5', fontsize = label_size, color = "black")
ax9.text(0.772, 0.1, '10', fontsize = label_size, color = "black")
ax9.scatter([0.158,0.467,0.777],[0.75,0.75,0.75], c='black', marker='.', edgecolors='black', linewidths=2.0, s = 90, label="3D ref_model")
ax9.text(0.88, 0.20, 'Time (Myr)', fontsize = label_size, color = "black")

ax1.set_ylabel("y (km)", fontsize = label_size)
ax5.set_ylabel("y (km)", fontsize = label_size)
ax5.set_xlabel("x (km)", fontsize = label_size)
ax6.set_xlabel("x (km)", fontsize = label_size)
ax7.set_xlabel("x (km)", fontsize = label_size)

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()

ax5.minorticks_on()
ax6.minorticks_on()
ax7.minorticks_on()

ax1.set_xticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])
ax6.set_yticks([])
ax7.set_yticks([])
ax9.set_xticks([])
ax9.set_yticks([])
ax9.axis('off')

# Plot colorbar
cbar_1 = figure.colorbar(img3, cax=ax4, orientation = 'vertical')
cbar_1.set_label('Temperature (K)', fontsize=label_size, rotation=-90, labelpad=18)
cbar_1.ax.tick_params(labelsize = ticks_size)
cbar_1.minorticks_on()

cbar_2 = figure.colorbar(img5, cax=ax8, orientation = 'vertical')
cbar_2.set_label(r'Viscosity ($log_{10} \left( Pa \cdot s \right)$)', fontsize=label_size, rotation=-90, labelpad=18)
cbar_2.ax.tick_params(labelsize = ticks_size)
cbar_2.minorticks_on()

# Displays the drawing results
plt.savefig("./Figure_SI.png", dpi=300, format="png")