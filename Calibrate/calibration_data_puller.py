# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:24:33 2019

@author: skidmore
"""
import h5py as h5
import os  
import matplotlib.pyplot as plt

f = ('K:/Data/Historical_calibration_data/Skidmore/2019/7-3-2019/5w')

name_set = set()
qpd_slope = []
datapath = '/mean_background_corrected_datum/optical_tweezer_calibration_parameters_for_experiment/QPD_slope_of_line'

#print(f[datapath][()])


for subdir, dirs, files in os.walk(f):
    for name in files:
        if name in name_set:
            pass
        else:
            name_set.add(name)
            absFile = os.path.abspath(os.path.join(subdir,name))
            if absFile.endswith('.h5'):
                h_open = h5.File(absFile, 'r')
                
                print(os.path.join(subdir, name))
                print(h_open[datapath][()])
                qpd_slope.append(h_open[datapath][()])
                h_open.close()
            else:
                pass
print(qpd_slope)


n, bins, patches = plt.hist(x=qpd_slope, bins='auto', color='#0504aa', rwidth=0.85)

plt.title('Xb_slope_Ben-2019')
plt.xlabel('Xb')
plt.ylabel('Count')
plt.savefig('Xb_slope'+'.png', bbox_inches='tight')
plt.show()


