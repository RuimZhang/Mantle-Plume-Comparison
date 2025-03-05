################################################################################
## Author: Rui-Min Zhang, Zhong-Hai Li* @ UCAS                                ##
##         Wei Leng                     @ USTC                                ##
##         Ya-Nan Shi, Jason P. Morgan  @ SUSTech                             ##
## Email: zhangruimin22@mails.ucas.ac.cn                                      ##
## Encoding: UTF-8                                                            ##
################################################################################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import numpy as np

X_length = 1000
Y_length = 680
Z_length = 1000
I3_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/standard_Model/I3/"
I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/standard_Model/I2/"

ticks_size = 12
label_size = 15
figure = plt.figure(figsize=(16,12), dpi=300)

ax1 = figure.add_axes([0.10,0.6,0.25,0.35])
ax1.annotate('(a)', xy=(-0.2,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax2 = figure.add_axes([0.34,0.54,0.7,0.45])
ax2.annotate('(b)', xy=(-0.01,0.89), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax2.annotate('x ($km$)', xy=(0.3,0.06), xycoords="axes fraction", fontsize=label_size, ha="center", va="center", rotation=-7)
ax2.annotate('y ($km$)', xy=(0.00,0.45), xycoords="axes fraction", fontsize=label_size, ha="center", va="center", rotation=92)
ax2.annotate('z ($km$)', xy=(0.71,0.20), xycoords="axes fraction", fontsize=label_size, ha="center", va="center", rotation=72)
ax3 = figure.add_axes([0.10,0.07,0.25,0.45])
ax3.annotate('(c)', xy=(-0.2,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax4 = figure.add_axes([0.40,0.07,0.25,0.45])
ax4.annotate('(d)', xy=(-0.1,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax5 = figure.add_axes([0.70,0.07,0.25,0.45])
ax5.annotate('(e)', xy=(-0.1,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")

# Plot (a), 2D composition
file_type = 'c'
file_num = 1
file_name = file_type + '%03d' % file_num

work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split())))
num = 0
index = 0
I2_composition = np.zeros(int(grid[0])*int(grid[1]), dtype=np.int32)
while (num < I2_data.shape[0]):
  value = int(I2_data[num])
  if value == -2:
    num_colors = int(I2_data[num+1])
    material = int(I2_data[num+2])
    start_index = index
    end_index = index + num_colors
    index = index + num_colors 
    num = num + 3
  else:
    if value == -1:
      material = np.nan
    else:
      material = value
    start_index = index
    end_index = index + 1
    index = index + 1
    num = num + 1
  I2_composition[start_index:end_index] = material
I2_composition = I2_composition.reshape(int(grid[0]),int(grid[1]))  

X = np.linspace(0,1000,int(grid[0]))
Y = np.linspace(0,-680,int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

I_colors = ["#00aaff", "#008900", "#006400", "#ab0000", "#e65c00", "#e0e01e"]
I_bounds = [-0.5,4.5,5.5,8.5,9.5,28.5,29.5]
I_cmap = ListedColormap(I_colors)
I_norms = BoundaryNorm(I_bounds, I_cmap.N)
img1 = ax1.pcolormesh(I2_X,I2_Y,I2_composition,cmap=I_cmap,norm=I_norms)
ax1.set_xlabel('x ($km$)', fontsize = label_size)
ax1.set_ylabel('y ($km$)', fontsize = label_size); ax1.set_ylim(-660,0)
ax1.tick_params(axis='x', labelrotation=0, grid_linestyle='--')
ax1.tick_params(axis='y', labelrotation=0, grid_linestyle='--')
ax1.tick_params(labelsize = ticks_size)


# Plot (b), 3D composition
composition_3D = plt.imread('./Figure_1_2.tiff')

img2 = ax2.imshow(composition_3D[300:2160, 600:3800, :])
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.spines[:].set_visible(False)


# Plot (c), 2D & 3D temperature
file_type = 't'

file_num = 5
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

mid_column = int(np.floor(int(grid[0])/2))
ax3.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 5 Myr", color='red')
ax3.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 5 Myr", color='cyan')


file_num = 30
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

ax3.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 30 Myr", color='blue')
ax3.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 30 Myr", color='gold')

ax3.set_xlim(250, 2000); ax3.set_ylim(-660,0)
ax3.set_xlabel('Temperature ($K$)', fontsize = label_size)
ax3.set_ylabel('y ($km$)', fontsize = label_size)
ax3.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
ax3.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
ax3.tick_params(labelsize = ticks_size)
ax3.legend(loc='lower left', fontsize=ticks_size)
ax3.grid(linestyle='--', color='silver', linewidth=0.5)

# Plot (d), 2D & 3D density

file_type = 'd'

file_num = 5
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

mid_column = int(np.floor(int(grid[0])/2))
ax4.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 5 Myr", color='red')
ax4.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 5 Myr", color='cyan')

file_type = 'd'

file_num = 30
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

mid_column = int(np.floor(int(grid[0])/2))
ax4.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 30 Myr", color='blue')
ax4.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 30 Myr", color='gold')


ax4.set_xlim(0, 4300); ax4.set_ylim(-660,0)
ax4.set_xlabel('Density ($kg/m^3$)', fontsize = label_size)
ax4.set_ylabel('', fontsize = label_size)
ax4.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
ax4.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
ax4.tick_params(labelsize = ticks_size)
ax4.legend(loc='lower left', fontsize=ticks_size)
ax4.grid(linestyle='--', color='silver', linewidth=0.5)


# Plot (e), 2D & 3D viscosity

file_type = 'v'

file_num = 5
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.log10(np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1])))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

mid_column = int(np.floor(int(grid[0])/2))
ax5.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 5 Myr", color='red')
ax5.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 5 Myr", color='cyan')


file_num = 30
file_name = file_type + '%03d' % file_num
work_Path = I2_Path
with open(work_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.log10(np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1])))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')
# Read I3 data
os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

ax5.plot(I2_data[mid_column,:], Y, linewidth=4.0, linestyle='-', marker='', markersize=0, label="2D 30 Myr", color='blue')
ax5.plot(I3_data[mid_column,:], Y, linewidth=2.0, linestyle='--', marker='', markersize=0, label="3D 30 Myr", color='gold')

ax5.set_xlim(17.8, 24.2); ax5.set_ylim(-660,0)
ax5.set_xlabel(r'Viscosity ($log_{10} \left( Pa \cdot s \right)$)', fontsize = label_size)
ax5.set_ylabel('', fontsize = label_size)
ax5.tick_params(axis='x', labelrotation=0, grid_linestyle='-.')
ax5.tick_params(axis='y', labelrotation=0, grid_linestyle='-.')
ax5.tick_params(labelsize = ticks_size)
ax5.legend(loc='lower left', fontsize=ticks_size)
ax5.grid(linestyle='--', color='silver', linewidth=0.5)


ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()

plt.savefig("./Figure_1.png", dpi=300, format="png")