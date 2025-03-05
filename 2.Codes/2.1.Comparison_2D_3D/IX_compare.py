import matplotlib.pyplot as plt
import numpy as np
import math 
import os
import sys
# Plume with tail
#temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
#size   = ("5000.00", "10000.00", "15000.00", "20000.00", "25000.00", "30000.00", "35000.00", "40000.00", "45000.00", "50000.00", "55000.00", "60000.00", "65000.00", "70000.00", "75000.00", "80000.00", "85000.00", "90000.00", "95000.00", "100000.00", "105000.00", "110000.00", "115000.00", "120000.00", "125000.00", "130000.00", "135000.00", "140000.00", "145000.00", "150000.00", "155000.00", "160000.00", "165000.00", "170000.00", "175000.00", "180000.00", "185000.00", "190000.00", "195000.00", "200000.00")

# Plume without tail
temp   = ("1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020", "2030", "2040", "2050.0", "2060.0", "2070.0", "2080.0", "2090.0", "2100.0", "2110.0", "2120.0", "2130.0", "2140.0", "2150.0", "2160.0", "2170.0", "2180.0", "2190.0", "2200.0", "2210.0", "2220.0", "2230.0", "2240.0", "2250.0")
size   = ("50000.00", "60000.00", "70000.00", "80000.00", "90000.00", "100000.00", "110000.00", "120000.00", "130000.00", "140000.00", "150000.00", "160000.00", "170000.00", "180000.00", "190000.00", "200000.00", "210000.00", "220000.00", "230000.00", "240000.00", "250000.00", "260000.00", "270000.00", "280000.00", "290000.00", "300000.00", "310000.00", "320000.00", "330000.00", "340000.00", "350000.00", "360000.00", "370000.00", "380000.00", "390000.00", "400000.00")

print(">>>> Program Start <<<<")
num_i = len(temp);   i_start = 0; i_end = len(temp); print("TempRange: " + str(temp[i_start]) + " ~ " + str(temp[i_end-1]) + " K")
num_j = len(size);   j_start = 0;  j_end = len(size); print("SizeRange: " + str(size[j_start]) + " ~ " + str(size[j_end-1]) + " m")
num_time = 21;       t_start = 0;  t_end = 21;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end-1) + "Myr")


save_i = []
save_j = []
save_err = np.zeros((num_i,num_j,num_time))
save_time = np.zeros((num_i,num_j))

X_length = 2000
Y_length = 660
Z_length = 2000
I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I2_Plume_noTail/"
I3_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I3_Plume_noTail/D_150/"
#I2_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I2_Plume/"
#I3_Path = "/Volumes/Mars_Base/9.Models/1.1.MPC/I3_Plume/T2200/"
file_type = 't'

for _ in range(t_start,t_end,1):
  file_num = _
  file_name = file_type + '%03d' % file_num

  # Read I2 data
  work_Path = I2_Path + "test_75/"
  with open(work_Path + file_name + '.txt', 'r') as I2_file:
    time     = np.array(list(map(float, I2_file.readline().strip().split())))[0]
    grid     = np.array(list(map(float, I2_file.readline().strip().split())))
    X        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
    I2_file.readline()
    Y        = np.array(list(map(float, I2_file.readline().strip().split())))/1000.0
    I2_file.readline()
    I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))
  I2_X, I2_Y = np.meshgrid(X, Y, indexing = 'ij')

  print('\n In %.2f' % _ + " (Myr) step, The comparison between I2VIS and I3VIS begin!")
  # Read I3 data
  os.system("pvpython makeSlice.py " + I3_Path + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
  I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
  I3_data    =  I3_file[:,0].reshape(int(grid[0]),int(grid[1]))
  I3_Y       = -I3_file[:,2].reshape(int(grid[0]),int(grid[1]))
  I3_X       = -I3_file[:,3].reshape(int(grid[0]),int(grid[1]))

  local_err0 = np.linalg.norm(x=(I2_data[3:-3,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None)
  local_err1 = np.linalg.norm(x=(I2_data[2:-4,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None) # left
  local_err2 = np.linalg.norm(x=(I2_data[4:-2,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None) # right
  local_err3 = np.linalg.norm(x=(I2_data[3:-3,19:39]-I3_data[3:-3,20:40]), ord=2, axis=None) # up
  local_err4 = np.linalg.norm(x=(I2_data[3:-3,21:41]-I3_data[3:-3,20:40]), ord=2, axis=None) # down

  min_err = np.min([local_err0, local_err1, local_err2, local_err3, local_err4])
  min_i = 2
  min_j = 3

  fail_test_num=0
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
            I2_data  = np.array(list(map(float, I2_file.readline().strip().split()))).reshape(int(grid[0]),int(grid[1]))

            local_err0 = np.linalg.norm(x=(I2_data[3:-3,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None)
            local_err1 = np.linalg.norm(x=(I2_data[2:-4,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None) # left
            local_err2 = np.linalg.norm(x=(I2_data[4:-2,20:40]-I3_data[3:-3,20:40]), ord=2, axis=None) # right
            local_err3 = np.linalg.norm(x=(I2_data[3:-3,19:39]-I3_data[3:-3,20:40]), ord=2, axis=None) # up
            local_err4 = np.linalg.norm(x=(I2_data[3:-3,21:41]-I3_data[3:-3,20:40]), ord=2, axis=None) # down
            save_time[i,j] = time
        except FileNotFoundError:
          fail_test_num = fail_test_num + 1
          save_err[i,j,_] = float('inf')
          save_time[i,j] = float('inf')
          #print("test" + str(j + i * num_j))
        else:
          save_err[i,j,_] = np.min([local_err0, local_err1, local_err2, local_err3, local_err4])
          if min_err > np.min([local_err0, local_err1, local_err2, local_err3, local_err4]):
            min_err = np.min([local_err0, local_err1, local_err2, local_err3, local_err4])
            min_i = i
            min_j = j
  print("Totally, " + str(fail_test_num) + "files Not Found" + str(fail_test_num / ((j_end - j_start) * (i_end - i_start)) * 100) + "%")
  save_i.append(min_i)
  save_j.append(min_j)
  
  print('\nAt %.2f' % _ + " (Myr), the most similar models are: " + str(temp[min_i]) + "K, " + str(size[min_j]) + "km, ")
  np.save(file="D_150.npy", arr=save_err)