# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:35:28 2019

@author: skidmore
"""
import h5py
import os



## Corrections for hdf5 files ##
calibration_of_optical_tweezers = 'This calibration calculates the spring constant (pN/nm) and XY displacement (V/V/nm) of an optically trapped sulphate-coated fluorescent bead (nominal radius 2 micrometers) in buffered saline at provided laser power.'
Stokes_Faxen_coefficient_units = 'pNs/nm'
optical_tweezers = 'A piece of apparatus, consisting of a laser, an optical microscope, a condenser and a quadrant photo detector, QPD sensor , that is used to manipulate microscopic particles. Adapted from irl: http://purl.obolibrary.org/obo/CHMO_0000946'



'''
count = 0
lines = 0
for dirname, dirs, files in os.walk('K:/Data/Historical_calibration_data/Skidmore/2019/3-5-2019'):
    for filename in files:
        filename_without_extension, extension = os.path.splitext(filename)
        if extension == '.h5':
            count +=1
            f1 = h5py.File(filename, 'r+')
            #with open(os.path.join(dirname, filename), 'r+') as f:
            data = f1[filename+'/calibration']
            data[...] = calibration
   '''         
#rootFolder = "K:/Data/Historical_calibration_data/Skidmore/2019/3-5-2019/"
#with open('K:/Data/Historical_calibration_data/Skidmore/2019/3-5-2019/bead1/calibration_optical_tweezers_03-05-2019.h5', 'r+') as f:
f = h5py.File('K:/Data/Historical_calibration_data/Skidmore/2019/3-5-2019/bead1/calibration_optical_tweezers_03-05-2019.h5', 'r+')
x1 = f['calibration']
x1.attrs['calibration of optical tweezers'] = calibration_of_optical_tweezers

#print(list(x1.keys()))
##THIS IS FOR USE ON FUTURE FILES BE SURE TO UNCOMMENT IT!
#x1.move('stokes_damping_constant', 'Stokes_Faxen_coefficient')
#x1_1 = x1['Stokes_Faxen_coefficient']
#x1_1.attrs['units'] = Stokes_Faxen_coefficient_units
#del x1_1['dimensions']


#SHOULD BE WORKING?? WHY WONT IT WRITE???
f.attrs['optical tweezers'] = 'A piece of apparatus, consisting of a laser, an optical microscope, a condenser and a quadrant photo detector, QPD sensor , that is used to manipulate microscopic particles. Adapted from irl: http://purl.obolibrary.org/obo/CHMO_0000946'

#Power at objective **NEW VARIABE**
##FOR BEN MEASUREMENTS
x1_1 = f['calibration']
x2 = x1_1.create_dataset('laser power at objective', data = 90)
##FOR VIVEK MEASUREMENTS
x1.create_dataset('laser power at objective', data = 130)
x2.attrs['laser power at objective'] = 'Power of the trapping laser at the trapping plane of the microsope objective'
x2.attrs['units'] = 'mW'



#del f.attrs['calibration/calibration of optical tweezers']
#print(list(f['calibration'].attrs))
    #data = f['calibration']
    #data[...] = calibration

