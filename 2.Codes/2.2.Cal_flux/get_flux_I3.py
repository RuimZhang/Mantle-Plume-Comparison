import matplotlib.pyplot as plt
import numpy as np
import math 
import os
import sys
from scipy import integrate

model_name=["100km", "125km", "ref_model", "175km", "200km", "2000K", "2050K", "2150K", "2200K"]
size=[100,125,150,175,200,150,150,150,150]
print(">>>> Program Start <<<<")
num_test = len(model_name)
num_time = 21;     t_start = 1;  t_end = 11;  print("TimeRange: " + str(t_start) + " ~ " + str(t_end-1) + "Myr")

save_flux = np.zeros((num_test,num_time))

X_length = 4000
Y_length = 660
Z_length = 4000
I3_Path = "/Volumes/SSD_ZhangRuimin/I3_Plume/"
file_type = 'u'

for _ in range(t_start,t_end,1):
  file_num = _
  file_name = file_type + '%03d' % file_num
  for i_test in range(len(model_name)):
    os.system("pvpython makeSlice.py " + I3_Path + model_name[i_test] + "/" + file_name + ".vtk " + I3_Path + "tmp.txt "+ str(X_length))
    I3_file = np.loadtxt(I3_Path + "tmp.txt", skiprows=1, dtype='float64', delimiter=',')
    I3_data    =  I3_file[:,2].reshape(405,405)/365.25/24/3600/100
    I3_X       =  I3_file[:,4].reshape(405,405)
    X          =  I3_X[1,:]
    I3_Z       = -I3_file[:,6].reshape(405,405)
    Z          =  I3_Z[:,1]
    I3_data[I3_data <= 0] = 0.0
    integral_X  = integrate.romb(I3_data[169:234,169:234],dx=X[2]-X[1], show=False, axis=0)
    integral_XZ = integrate.romb(integral_X,dx=Z[2]-Z[1], show=True)
    print("Romberg: ", integral_XZ)
    save_flux[i_test,_] = integral_XZ
    np.save(file="3D_flux_bak.npy", arr=save_flux)
