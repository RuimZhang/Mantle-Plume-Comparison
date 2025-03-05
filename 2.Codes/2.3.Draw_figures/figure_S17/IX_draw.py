import matplotlib.pyplot as plt
import numpy as np
import math 
import os
from cmcrameri import cm

temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("50000.00", "60000.00", "70000.00", "80000.00", "90000.00", "100000.00", "110000.00", "120000.00", "130000.00", "140000.00", "150000.00", "160000.00", "170000.00", "180000.00", "190000.00", "200000.00", "210000.00", "220000.00", "230000.00", "240000.00", "250000.00", "260000.00", "270000.00", "280000.00", "290000.00", "300000.00", "310000.00", "320000.00", "330000.00", "340000.00", "350000.00", "360000.00", "370000.00", "380000.00", "390000.00", "400000.00")



X_length = 2000
Y_length = 660
Z_length = 2000

ticks_size = 10
label_size = 12
figure = plt.figure(figsize=(6,10), dpi=300)

cm_cmap_T = cm.vik

ax1 = figure.add_axes([0.15,0.70,0.8,0.25])
ax1.annotate('(a)', xy=(-0.1,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax2 = figure.add_axes([0.15,0.42,0.8,0.25])
ax2.annotate('(b)', xy=(-0.1,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")
ax3 = figure.add_axes([0.15,0.14,0.8,0.25])
ax3.annotate('(c)', xy=(-0.1,1.0), xycoords="axes fraction", fontsize=label_size, ha="center", va="center")

ax4 = figure.add_axes([0.15,0.06,0.80,0.02])


draw_width = 600
draw_start = -60
draw_end = -300

################## 
## figure (a,b)
print("Figure b: ")
i = 19 ; print("Temp: ", temp[i])
j = 12 ; print("Size: ", size[j])
file_num = 2
file_type = 't'
file_name = file_type + '%03d' % file_num

I3_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I3_Plume_noTail/2200K/"
I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I2_Plume_noTail/test_" + str(j  + i * len(size)) + "/"

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

half_column = int(np.floor(grid[0]/2))
y_num_start = 20; y_num_end=35


ax1.set_xlim([X_length/2.0-draw_width,X_length/2.0])
ax1.set_ylim([draw_end,draw_start])
img1 = ax1.pcolor(I3_X[0:half_column,],I3_Y[0:half_column,],I3_data[0:half_column,],cmap = cm_cmap_T,vmin = 1500, vmax = 1900)
ax1.text(420, -260, '3D T_2200: \n2200 (K), 200 (km)', style = 'oblique', fontsize = label_size, color = "black")
ax1.text(420, -290, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")

ax2.set_xlim([X_length/2.0-draw_width,X_length/2.0])
ax2.set_ylim([draw_end,draw_start])
img2 = ax2.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],I2_data[0:half_column,],cmap = cm_cmap_T,vmin = 1500, vmax = 1900)
ax2.text(420, -260, '2D Opt_T_2200: \n2140 (K), 170 (km)', style = 'oblique', fontsize = label_size, color = "black")
ax2.text(420, -290, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")

################## 
## figure (c)
print("Figure c: ")
i = 25 ; print("Temp: ", temp[i])
j = 10 ; print("Size: ", size[j])
file_type = 't'
file_name = file_type + '%03d' % file_num
I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I2_Plume_noTail/test_" + str(j  + i * len(size)) + "/"

with open(I2_Path + file_name + '.txt', 'r') as I2_file:
  time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
  grid     = np.array(list(map(float, I2_file.readline().strip().split())))
  X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  Y        = -np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
  I2_file.readline()
  I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')



ax3.set_xlim([X_length/2.0-draw_width,X_length/2.0])
ax3.set_ylim([draw_end,draw_start])
img3 = ax3.pcolor(I2_X[0:half_column,],I2_Y[0:half_column,],I2_data[0:half_column,],cmap = cm_cmap_T,vmin = 1500, vmax = 1900)
ax3.text(420, -260, '2D Model: \n2200 (K), 150 (km)', style = 'oblique', fontsize = label_size, color = "black")
ax3.text(420, -290, '%.2f' % file_num + " (Myr)", fontsize = ticks_size, color = "black")

ax1.set_ylabel("y (km)", fontsize = label_size)
ax2.set_ylabel("y (km)", fontsize = label_size)
ax3.set_ylabel("y (km)", fontsize = label_size)
ax3.set_xlabel("x (km)", fontsize = label_size)

ax1.set_xticks([])
ax2.set_xticks([])

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()

# Plot colorbar
cbar_1 = figure.colorbar(img1, cax=ax4, orientation = 'horizontal')
cbar_1.set_label('Temperature (K)', fontsize=label_size)
cbar_1.ax.tick_params(labelsize = ticks_size)
cbar_1.minorticks_on()

# Displays the drawing results
plt.savefig("./Figure_S17.png", dpi=300, format="png")