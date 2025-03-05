import matplotlib.pyplot as plt
import numpy as np
import math 
import os
import sys
from scipy import integrate
from scipy.interpolate import interp1d
# Plume with tail
temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00")

print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 0;  j_end = len(size); print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 1;  t_end = 11;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end-1) + "Myr")

save_flux = np.zeros((num_i,num_j,num_time))

X_length = 4000
Y_length = 660
Z_length = 4000
I2_Path = "/Volumes/SSD_ZhangRuimin/I2_Plume/"
file_type = 'u'

for _ in range(t_start,t_end,1):
  file_num = _
  file_name = file_type + '%03d' % file_num
  fail_test_num = 0
  for i in range(i_start,i_end):
    for j in range(j_start,j_end):
        percent = int((1 + (j - j_start) + (j_end - j_start) * (i - i_start)) / ((j_end - j_start) * (i_end - i_start)) * 100)
        print("\r", end="")
        print("Comparation Progress: {}%: ".format(percent), "▋" * (percent // 2), end="")
        sys.stdout.flush()

        work_Path = I2_Path + "test_" + str(j + i * len(size)) + "/"
        try:
          with open(work_Path + file_name + '.txt', 'r') as I2_file:
            time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
            grid     = np.array(list(map(float, I2_file.readline().strip().split())))
            X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
            I2_file.readline()
            Y        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
            I2_file.readline()
            I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]*2),int(grid[1]))
            Vx = I2_data[0:int(grid[0]),:]
            Vy = I2_data[int(grid[0]):int(grid[0]*2),:]
        except FileNotFoundError:
          fail_test_num = fail_test_num + 1
          save_flux[i,j,_] = float('inf')
        else:
          Vy[Vy>0] = 0.0
          f = interp1d(X,(Vy[:,65]+Vy[:,66]+Vy[:,67])/3.0)
          flux,err= integrate.quad(f,X_length/2-float(size[j])/500, X_length/2+float(size[j])/500)
          save_flux[i,j,_] = -flux

  print("Totally, " + str(fail_test_num) + "files Not Found" + str(fail_test_num / ((j_end - j_start) * (i_end - i_start)) * 100) + "%")
  np.save(file="2D_flux.npy", arr=save_flux)

