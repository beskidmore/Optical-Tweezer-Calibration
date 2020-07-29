# -*- coding: utf-8 -*-
### Code for calibration of optical tweezers                                                      ###
### Author: Benjamin Skidmore, based on matlab code by Vivek Rajasekharan                         ###
### Last edit Date: 7/16/2019                                                                     ###
### Written in Python 3.6                                                                         ###
### Referenced equations are from:                                                                ###
### Rajasekharan, V., Sreenivasan, V. K. A., & Farrell, B. (2017).                                ###
### Force Measurements for Cancer Cells. Methods in Molecular Biology (Clifton, N.J.),            ###
### 1530, 195–228. https://doi.org/10.1007/978-1-4939-6646-2_12                                   ###
### See the README for all HDF5 attribute definitions                                             ###

# Dependencies: matplotlib, math, numpy, plotly, h5py
# numpy, matplotlib, plotly, and h5py will need to be installed by user separately and in the path

# Imports libraries #
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics 
import os  
from plotly.offline import  plot
import plotly.graph_objs as go
# Expect depreciation warning from h5py. It is a known error and harmless for now
import h5py
import pandas as pd
import scipy.io
# Path to working directory 
#rootFolder = input('Enter path to Working directory:')
# if in set directory can utilize this 
rootFolder = "H:/Data/Skidmore/OT_testing/10-15-2019/calibration/bead1/SW"


# Load dark signal for incorporation. eq. 11
with open('H:/Data/Skidmore/OT_testing/10-15-2019/calibration/bead3/P/x1_properties.txt') as file2:
    next(file2)
    for lne in file2:
        lne.strip(' ')
        dark_x = float(lne.split('\t')[3])
        dark_y = float(lne.split('\t')[4])
        dark_sum = float(lne.split('\t')[5])
        break
file2.close()

# The stokes damping constant
# Set B based on height (pNsnm^-1)
# standardized at: 70 nm, Bead radius of 2 um      May be different.
Beta = 0.0000391 

# Set gain, this is almost always 10 but if different users are able to type it as input
#G = int(input('Enter Gain (1, 3, or 10): '))
G = 10

# Radius of Bead (micrometer (um)). Nominal diameter of 4um. Used in calculation of Beta. METADATA
R = 2

# Set variables
wavestart = 4000
file_iteration = 0
dis = []
dis_tup = []
Xp_points = []
Yp_points = []
theta_lst = []
acst_lst = []
Xb = []
Xt_predicted = []
name_set = set()

# Gather info for writing input data files to HDF5/matlab formats
measure_date = 0 
while measure_date == 0:
    try: 
        measure_date = str(input('What day was the experiment done (mm-dd-yyyy):'))
    except ValueError:
        print('That is not a number.')
    else:
        if len(measure_date) == 10 and len(measure_date.split('-')[0]) == 2 and len(measure_date.split('-')[1]) == 2 and len(measure_date.split('-')[2]) == 4:
            if int(measure_date.split('-')[0]) in range(13) and int(measure_date.split('-')[1]) in range(32) and int(measure_date.split('-')[2]) in range(1950,3000):
                break
            else:
                print('This is not a reasonable date, use mm-dd-yyyy OR mm_dd_yyyy format')
                measure_date = 0
        else:
            print('This is not a reasonable date, use mm-dd-yyyy OR mm_dd_yyyy format')
            measure_date = 0
beadnumber = 0            
while beadnumber == 0:
    try:
        beadnumber = str(input('What number bead is calibrated on this day (1-100):'))
    except ValueError:
        print('That is not a number.')


# Prepare to write to hdf5 format
h = h5py.File('calibration_optical_tweezers_'+measure_date+'_bead_'+beadnumber+'.h5', 'w')
calib = h.create_group('calibration')
dset_raw = h.create_group('measurement_datum')
dset_trans = h.create_group('mean_background_corrected_datum')
mean_ot_cal_param = dset_trans.create_group('optical_tweezer_calibration_parameters_for_experiment')
draw_dis = dset_raw.create_group('planned_displacement_of_trapped_bead')
draw_dark = dset_raw.create_group('mean_background_signal_in_darkness')
ddx = draw_dark.create_dataset('x_signal_darkness', data=dark_x)
ddy = draw_dark.create_dataset('y_signal_darkness', data=dark_y)
dds = draw_dark.create_dataset('sum_signal_darkness', data=dark_sum)
dsgain = calib.create_dataset('gain', data=G)
dsbeta = calib.create_dataset('Stokes_Faxen_coeffiecent', data=Beta)
#SET POWER HERE EVERY TIME! VIVEK =130 SKIDMORE = 90
dspower = calib.create_dataset('power at microscope objective', data = 90)

# Supply attributes for hdf5 file
h.attrs['measurement date'] = measure_date
h.attrs['data steward'] = 'Brenda Farrell'
h.attrs['researcher'] = 'Vivek Rajasekharan'
h.attrs['data scientist'] = 'Benjamin Skidmore'
h.attrs['objective'] = 'Calibration of optical tweezers'
h.attrs['funding'] = 'NIH-R21CA1152779, NIH S10 RR027549-01, and 2480050302'
h.attrs['bead number'] = beadnumber
calib.attrs['calibration'] = 'A planned process with the objective to establish the relationship between data produced by a measurement device and physical qualities. This is done by using the measurement device under defined conditions, and either tuning it to adjust the measured output, or record the output and use it as a reference in future measurements. irl: http://purl.obolibrary.org/obo/OBI_0000818'
calib.attrs['optical tweezers'] = 'A piece of apparatus, consisting of a laser, an optical microscope, a condenser and a quadrant photo detector, QPD sensor , that is used to manipulate microscopic particles. Adapted from irl: http://purl.obolibrary.org/obo/CHMO_0000946'
calib.attrs['calibration of optical tweezers'] = 'This calibration calculates the spring constant (pN/nm) and XY displacement (V/V/nm) of an optically trapped sulphate-coated fluorescent bead (nominal radius 2 micrometers) in buffered saline at provided laser power.'
#h.attrs['cell number'] = 'No cell number for calibration'
dset_raw.attrs['measurement_datum'] = 'A measurement datum is an information content entity that is a recording of the output of a measurement such as produced by a device. irl: http://purl.obolibrary.org/obo/IAO_0000109'
dset_raw.attrs['editor note'] = 'This is a collection of raw data before transformation'
dset_trans.attrs['mean_background_corrected_datum'] = 'An averaging data transformation is a data transformation that has objective averaging. A background correction data transformation (sometimes called supervised classification) is a data transformation that has the objective background correction. irl: http://purl.obolibrary.org/obo/OBI_0200170 | http://purl.obolibrary.org/obo/OBI_0000666'
dset_trans.attrs['editor note'] = 'Data after subtracting signal in dark and averaging over ~750 periods'
dset_trans.attrs['units'] ='Volts (V)'
draw_dis.attrs['planned_displacement_of_trapped_bead'] = 'Desired movement of the bead in x direction within the xy-trapping plane with AODs'
draw_dis.attrs['units'] ='Nanometers (nm)'
draw_dark.attrs['mean_background_signal_in_darkness'] = 'Average of the dark signal (background noise) measured before a calibration'
ddx.attrs['x_signal_dark'] = 'Mean proportion of signal in x direction measured in darkness, this is the background signal detected in the x direction'
ddx.attrs['units'] = 'Volts (V)'
ddy.attrs['y_signal_dark'] = 'Mean proportion of signal in y direction measured in darkness, this is the background signal detected in the y direction'
ddy.attrs['units'] = 'Volts (V)'
dds.attrs['sum_signal_darkness'] = 'Mean sum of signal in darkness'
dds.attrs['units'] = 'Volts (V)'
dsgain.attrs['definition'] = 'Gain is a measure of the ability of a two-port circuit (often an amplifier) to increase the power or amplitude of a signal from the input to the output port by adding energy converted from some power supply to the signal. Source: Graf, Rudolf F. (1999). Modern Dictionary of Electronics (7 ed.). Newnes. p. 314. ISBN 0080511988.'
dsgain.attrs['units'] = 'Volts/Volts (V/V)'
dsgain.attrs['gain of QPD'] = 'During data collection, gain is applied to boost the signal (background signal corrected current in each quadrant * QPD resistance (200 kΩ) ) from the first stage amplifier before it is processed by the second stage amplifier.'
dsbeta.attrs['definition'] = 'Stokes damping constant is a variable that helps approximate the relationship between fluid and solids, specifically those which exhibit simple harmonic motion at small amplitudes.' 
dsbeta.attrs['formula'] = 'β = 6πηR / [I - 9/16(R/h) + 1/8(R/h)^3 + 45/256(R/h)^4 - 1/16(R/h)^5], where R is the radius of the bead, h is the height above a dish the bead is suspended. The term in the denominator considers the additional effect on the hydrodynamic drag caused by proximity of the Petri dish to the bead and η is the viscosity of the solution.'
dsbeta.attrs['units'] = 'picoNewtons*second / Nanometers | (pNs/nm)'
dspower.attrs['definition'] = 'Laser power measured at the microscope objective'
dspower.attrs['units'] = 'milliWatts (mW)'

# First matlab file, measured data, experimental variable (gain, dark influence, etc.)
column_count = 1
df = np.zeros((5,9), dtype=np.object)
df[0,0] = measure_date
df[0,1] = 'Measured X (nm)'
df[0,2] = 'Measured Y (nm)'
df[0,3] = 'Measured Sum (nm)'
df[0,4] = 'Dark_X (nm)'
df[1,4] = dark_x
df[2,4] = dark_x
df[3,4] = dark_x
df[4,4] = dark_x
df[0,5] = 'Dark_Y (nm)'
df[1,5] =  dark_y
df[2,5] =  dark_y
df[3,5] =  dark_y
df[4,5] =  dark_y
df[0,6] = 'Dark_sum (nm)'
df[1,6] = dark_sum
df[2,6] = dark_sum
df[3,6] = dark_sum
df[4,6] = dark_sum
df[0,7] = 'Gain'
df[1,7] = G
df[2,7] = G
df[3,7] = G
df[4,7] = G
df[0,8] = 'Beta (stokes damping constant)'
df[1,8] = Beta
df[2,8] = Beta
df[3,8] = Beta
df[4,8] = Beta

# Second matlab file initialization and organization, transformed and calculated data
dft = np.zeros((5,8), dtype=np.object)
dft[0,0] = measure_date
dft[0,1] = 'Transformed Data'
#dft[0,2] = 'Actual Displacement'
dft[0,3] = 'Theta (Angle of Movement)'
dft[0,4] = 'Xt_voltage'
dft[0,5] = 'QPD Slope of line (V/V/nm)'
dft[0,6] = 'Time Constant'
dft[0,7] = 'Spring Constant kp'

# Iterate through all four files in working directory. Load data and implement algorithms
while file_iteration < 4:
    # Reads data input file and puts qpd dX, qpd dY, and qpd sum data into lists
    xs = []
    ys = []
    qpd_sum = []
    for root, dirs, files in os.walk(rootFolder):  
        for name in files:
            if name in name_set:
                pass
            else:
                name_set.add(name)
                if name.endswith('n.txt'):
                    displacement = -1*int(name[1:-5])
                    dis_title = 'n'+str(name[1:-5])
                elif name.endswith('p.txt'):
                    displacement = int(name[1:-5])
                    dis_title = str(name[1:-5])
                else:
                    continue
                
                # Identifies those files that end with .txt and makes the file path to these files
                shpName = os.path.splitext(name)[0]
                absFile = os.path.abspath(os.path.join(root,name))
                # Opens all files that end with .txt in the working directory
                if absFile.endswith('.txt'):    
                    # Write original measurement data to HDF5 file     
                    d = pd.read_table(absFile)
                    title = "{0}_{1}".format((measure_date), str(dis_title))
                    
                    # Load data for use in calculations/transformation, etc. 
                    file1 = open(absFile, 'r')
                    print('opened one')
                    for i in range(wavestart+1):
                        next(file1)
                    for line in file1:
                        if line != '\n':
                            xs.append(float(line.split('\t')[0]))
                            ys.append(float(line.split('\t')[1]))
                            qpd_sum.append(float(line.split('\t')[2]))
                    # As data is read from txt file write it into matlab readable cell arrays
                    df[column_count,0] = displacement
                    df[column_count,1] = xs
                    df[column_count,2] = ys
                    df[column_count,3] = qpd_sum
                    
                    # Read large raw data into HDF5 file. First create groups by displacement the insert by type
                    dis_draw_dis = draw_dis.create_group(str(displacement))
                    dis_draw_dis.attrs[str(displacement)]  = 'Planned displacement of the bead in the x direction within the xy-trapping plane with AODs.'
                    dis_draw_dis.attrs['units'] = 'Nanometers (nm)'
                    dddx = dis_draw_dis.create_dataset('displacement_x', data = xs)
                    dddy= dis_draw_dis.create_dataset('displacement_y', data = ys)
                    ddds = dis_draw_dis.create_dataset('sum_signal_in_light', data = qpd_sum)
                    dddx.attrs['displacement_x'] = 'Proportion of measured displacement in the x direction'
                    dddx.attrs['units'] = 'Volts (V)'
                    dddy.attrs['displacement_y'] = 'Proportion of measured displacement in the y direction'
                    dddy.attrs['units'] = 'Volts (V)'
                    ddds.attrs['sum_signal_in_light'] = 'Sum of signal in light (wavelength transmission between 530 nm - 590 nm, with excitation peak at 576 nm) '
                    ddds.attrs['units'] = 'Volts (V)'
                    column_count += 1
                    file1.close()
                    break

    print(displacement) 
       
    # Account for dark here.
    # Subtracts dark*G from every data point in the X / Y columns of read files. 
    #xs = [float(x) - (dark_x*G) for x in xs]
    #ys = [float(y) - (dark_y*G) for y in ys]
    # Subtracts dark from data points in sum column of read files then multiplies by gain
    # Notice difference in where gain is implemented.
    #qpd_sum = [(float(q) - dark_sum)*G for q in qpd_sum]
    #### 6-28-2019 Found that we already account for the dark in labview. Gain may or may not be taken care of?
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    qpd_sum = [(float(q))*G for q in qpd_sum]
    # Periodically  average every 16000 data points. 16000 is the number of data points in a period of the square wave.
    # Produces variables that will be reused.
    sw_period = 16000
    i = 0
    num_periods = 0
    L1 = len(xs)    
    sum_xs = [0] * sw_period
    sum_ys = [0] * sw_period
    sum_qpd_sum = [0] * sw_period
    x_avgs = []
    y_avgs = []
    s_avgs = []
        
    # Finds sum data points period wise -> [a,b,c] + [a',b',c'] ... -> [a+a',b+b',...]. 
    # Then find the average of each data point by dividing by # of periods.
    # DATA TRANSFORMATION BEGINS HERE
    while sw_period + i <= L1:
        sum_xs = [float(x) + float(y) for x,y in zip(sum_xs, xs[i:(i+sw_period)])]
        sum_ys = [float(x) + float(y) for x,y in zip(sum_ys, ys[i:(i+sw_period)])]
        sum_qpd_sum = [float(x) + float(y) for x,y in zip(sum_qpd_sum, qpd_sum[i:(i+sw_period)])]    
        i = i + sw_period
        num_periods += 1
    # Get average of points over every period
    x_avgs = [(x / num_periods) for x in sum_xs]
    y_avgs = [(y / num_periods) for y in sum_ys]
    s_avgs = [(s / num_periods) for s in sum_qpd_sum]

    ## Now start calculation of slope eq. 12: Mqpdx = DXl/Sl/Dx
    ## Units in V/V/nm. see fig 8a.
    ## Xp = DXL/Sl
    ## Yp - DYl/Sl
    ## Dx is known during calibration and corresponds to displacement (nm)
    
    # Use average of points to find true average of square wave points of interest
    # Makes list of -y[1-3000] and divides it by qpd_sum_avg, 10*s_avg[1:3000]
    # Finds ratio of -y/sum pointwise for wave zero points (p1) and high/low points (p2) _|--|_
    p1 = [(float(y)*-1) / (float(s)) for y,s in zip(y_avgs[1:3000], s_avgs[1:3000])]
    p2 = [(float(y)*-1) / (float(s)) for y,s in zip(y_avgs[6000:10000], s_avgs[6000:10000])]

    # The average is taken across these points first (p1) 1-3000, then (p2) 6000-10000
    p1 = float(sum(p1))/float(len(p1))
    p2 = float(sum(p2))/float(len(p2))

    # Find difference in average top of square wave and bottom of square wave. _|--|_
    Xp = (p2 - p1)

    # Makes list of -x[1-3000] and divides it by qpd_sum_avg, 10*s_avg[1:3000] 
    # Finds ratio of -x/sum pointwise for wave zero points (p1) and high/low points (p2) _|--|_
    yp1 = [float(x)*-1 / float(s) for x,s in zip(x_avgs[1:3000], s_avgs[1:3000])]
    yp2 = [float(x)*-1 / float(s) for x,s in zip(x_avgs[6000:10000], s_avgs[6000:10000])]
    
    # The average is taken across these points either 1-3000, or 6000-10000
    yp1 = float(sum(yp1))/float(len(yp1))
    yp2 = float(sum(yp2))/float(len(yp2))
    
    # Find difference in top of square wave and bottom of square wave. _|--|_
    Yp = (yp2 - yp1)
    print(Xp, Yp)
    # Now we move on to identifying theta, the angle corresponding to the distance the bead moved in circular 2D plane.
    # Get theta in degrees.
    #  
    theta = math.degrees(math.atan(Yp/Xp))
    
    ## The following portion is modeled after matlab:
    ## Acst = Xp / cosd(theta)
    ## qpd_signal2 = -y_avgs / (10*math.cos(theta)*s_avgs)
    ## qpd_sig = qpd_sig2 - mean(qpd_sig2[1000:3000])
    
    # Because we are interested in the hypotenuse, true distance, displacement we need to identify the QPD intensity (V/V) 
    # relative to the movement
    # This differs from the matlab version (cosd(theta)) because python handles input to cos as degrees. 
    # This is Xt_voltage
    # . 
    acst = Xp / math.cos(math.radians(theta))

    # Create list of normalized qpdsignal using the QPD intensity during the bead movement.
    # This will be used to identify the measured displacement.
    qpdsignal2 = []
    for i in range(0,len(s_avgs)):
        ## The matlab code has this as -y_avgs/ (10*cosd(theta)*s_avgs).
        qpdsignal2.append(-(y_avgs[i])/(math.cos(math.radians(theta))*s_avgs[i]))
    qpdsignal = []
    qpd_lower_avg = np.mean([i for i in qpdsignal2[1000:3000]])
    for i in range(0, len(qpdsignal2)):  
        qpdsignal.append(qpdsignal2[i] - qpd_lower_avg)
       
        
    # Now that we have Xp and Yp it is possible to determine slope Mqpdx/Mqpdy
    # Generate list of Xp and Yp from each data file for comparison against displacement (known in calibration)
    # Plot these Xps vs displacement and plot Yps against displacement
    # add displacements to displacements list in order of those seen. do same to XP_points.
    dis.append(displacement)
    
    dis_tup.append((displacement, Xp, Yp, theta, acst, qpdsignal, title))
    
    # Plot dx/dl/displacement and get slope, plots measured displacement using this slope, plots square waves, 
    # Allows for selection of two points for the identification of time constant slope.
    # Only plots if all four data files have been seen. Can always change this to a variable if necessary
    if len(dis) == 4:
        # Order displacement from least to greatest 
        dis.sort()
        count = 0
        # Creates corresponding y coordinate for (displacement, QPD intensity)
        # Make Xb list of lists. holding 16000 X 1 for each data file loaded (4)    
        for d, x, y, theta, acst, qpdsignal, title in dis_tup:
            count += 1
            if d in dis:
                Xp_points.insert(int(dis.index(d)), x)
                Yp_points.insert(int(dis.index(d)), y)
                theta_lst.insert(int(dis.index(d)), theta)
                acst_lst.insert(int(dis.index(d)), acst)
                Xb.insert(int(dis.index(d)), qpdsignal)
            else:
                continue
        #This is called Normalized_detected_displacement. It is the Y component used for identifying Xb slope 
        Xt_voltage = acst_lst

        # Plots Acst vs displacements
        # Uses least squares to fit the line to data points for dx/dl/displacement in V/V/nm
        # Will use this slope to determine the accuracy of movement via plot of: Xb*1/slope
        # Because we only get one graphical output for QPD slope and AOD slope:
        fit = np.polyfit(dis, acst_lst, 1)
        print(acst_lst)
        print('fit of line', fit)
        qpd_slope = fit[0]
        fit_fn = np.poly1d(fit)
        plt.plot(dis, acst_lst, 'ro', dis, fit_fn(dis), '-')
        plt.title('QPD slope and AOD slope')
        plt.xlabel('Displacement (nm)')
        plt.ylabel('QPD intensity (V/V)')
        plt.savefig(measure_date+'_QPD_slope'+'.png', bbox_inches='tight')
        plt.show()         
        
        # Write qpd slope to h5 file. Cannot do later as it won't write over
        dst_qpd = mean_ot_cal_param.create_dataset('QPD_slope_of_line', data = qpd_slope)
        dst_qpd.attrs['qpd_slope_of_line'] = 'Slope of a least squares line created from points of planned displacement in the x-plane vs. normalized detected displacement (V/V)'
        dst_qpd.attrs['units'] = 'Volts/Volts/Nanometers (V/V/nm)'
        # Xt_predicted. This is (Acst at each dis. - lineslop(2))/lineslope(1).  True Displacement.
        # That is Calculated hypotenuse for a given displacement  - y-intercept on plot of expected displacements (x) vs. the calculated hypotenuse displacement/ slope of line on this plot. 
        print('dis', dis)
        
        for i in range(len(dis)):
            print(acst_lst[i], fit[1], fit[0])
            print((acst_lst[i] - fit[1])/fit[0])
            Xt_predicted.append((acst_lst[i] - fit[1])/fit[0])

         
        # Identify time constant, that is slope of line between two selected points using least squares.
        # Use this slope (time constant), m, to identify the Spring constant, kp, by multiplying by 
        # Beta, found from Stoke's-Faxen Law, eq. 8. Multiplication: eq. 7 
        def fitstat(time, xb4):
            print('lengths', len(time), len(xb4))
            m,b = np.polyfit(time, xb4, 1)
            print('Slope of selected points', m)
            fitstat.m = m
            
        # Create a numpy array of Xb for easy list comprehension parsing and indexing     
        A = np.array(Xb)
        proplst = []
        spoint = []
        mlst = []
        sp_con_lst = []
        recp_time_lst = []
        # Plot of displacements in nm vs. data points (16000)
        # This is separate from next "for" loop because we plot square waves there and its syntax is the same
        for i in range(len(Xb)):
            plt.plot(list(range(len(A[i]))), [x*(1/qpd_slope) for x in A[i]])
        plt.title('Measured Displacement')
        plt.xlabel('Data Points')
        plt.ylabel('Displacement (nm)')
        plt.savefig(measure_date+'Measured_displacement'+'.png', bbox_inches='tight')
        plt.show()
        
        # This plots the Square waves as well as allows for selection of x-coordinates for slope identification
        for i in range(len(Xb)):
            trial1 = []
            # Access the numpy array of Xb's and find difference in averages of top and zeros point in waves
            acst2 = np.mean(A[i, 7000:9000]) - np.mean(A[i, 1000:3000])
            # trial1 becomes the plot where two points are selected for slope identification.
            # abs is used because negative numbers don't work with log.
            for j in A[i, 3500:6500]:    
                trial1.append(np.log(abs(1 - (float(j)/float(acst2)))))
        
            ylst = np.array(trial1)
            xlst = np.linspace(3500, 6500, 3000)
    
            # For visualizing and manually selecting points for identifying slope.            
            trace = go.Scatter(x = xlst, y = ylst)
            tra = [trace]    
            plot(tra, filename='Displacement_%s.html' % str(dis[i]))
            # Manual selection of points
            #x1 = int(input('What is the First x-coordinate?'))
            #x2 = int(input('What is the Second x-coordinate?'))
            ## Only manually select one time, may standardize answers but risk missing line if data is incorrect
            while i == 0:
                # Allows for selection of points only if they are within range and in correct positions
                try:
                    x1 = int(input('What is the First x-coordinate?'))
                    x2 = int(input('What is the Second x-coordinate?'))
                except ValueError:
                    print('That is not a number.')
                else:
                    if 4000 <= x1 <= 16000 and x1 < x2 <= 16000:        
                        break
                    else:
                        print('x2 must be greater than x1 and both x1 and x2 must be between 0 and 16000')
            y1 = ylst[x1-3500]
            y2 = ylst[x2-3500]
            
            # Fitting part of the function. Identify regression coefficient with time and selected points(x1,x2) 
            # Time is a variable that is a reversion back to seconds for the range of selected points. (measurements are taken at 200 kHz)
            time = np.arange(0, (x2-x1)*.000005, .000005)
            xb4 = []
            # Because we don't reference trial1 we must convert these using the same methodology (1-i/acst2)
            for j in A[i, x1:x2]:
                xb4.append(np.log(abs(1-float(j)/float(acst2))))
            # The way that np.arange works may give a different number of points than required.
            while len(time) != len(xb4):
                print('Those data points broke python, please choose data points further apart or closer together.')
                x1 = int(input('What is the new First x-coordinate?'))
                x2 = int(input('What is the new Second x-coordinate?')) 
                y1 = ylst[x1-3500]
                y2 = ylst[x2-3500]
                time = np.arange(0, (x2-x1)*.000005, .000005)
                xb4 = []
                for j in A[i, x1:x2]:
                    xb4.append(np.log(abs(1-float(j)/float(acst2))))
                print(len(time), len(xb4))
            else:
                fitstat(time, xb4)
            
            spring_constant = fitstat.m*Beta
            # Create lists of spring constant and reciprocal time constant for later averaging. 
            sp_con_lst.append(spring_constant)
            recp_time_lst.append(fitstat.m)
            
            # Write transformed data to the h5 file
            dtrans_dis = dset_trans.create_group(str(dis[i]))
            dtd_transformed = dtrans_dis.create_dataset('mean_background_corrected_datum', data = A[i])
            #dtd_actual = dtrans_dis.create_dataset('calculated_displacement', data = Xt_predicted[i])
            dtd_theta = dtrans_dis.create_dataset('theta', data = theta_lst[i])
            dtd_xt_volt = dtrans_dis.create_dataset('Normalized_detected_displacement', data = Xt_voltage[i])
            dtd_recp = dtrans_dis.create_dataset('reciprocal_time_constant', data = fitstat.m)
            dtd_spring = dtrans_dis.create_dataset('spring_constant', data = spring_constant)
            
            # Write attributes to hdf5 file
            dtrans_dis.attrs[str(dis[i])] = 'Planned displacement of the bead in the x direction'
            dtrans_dis.attrs['units'] = 'Nanometers (nm)'
            dtd_transformed.attrs['mean_background_corrected_datum'] = 'An averaging data transformation is a data transformation that has objective averaging. A background correction data transformation (sometimes called supervised classification) is a data transformation that has the objective background correction. irl: http://purl.obolibrary.org/obo/OBI_0200170 | http://purl.obolibrary.org/obo/OBI_0000666' 
            #dtd_actual.attrs['calculated_displacement'] = 'Calculated total displacement of the bead, independent of direction within the xy-plane'
            #dtd_actual.attrs['units'] = 'Nanometers (nm)'
            dtd_theta.attrs['theta'] = 'Angle of bead movement relative to the xy-plane, where 0 ≤ θ ≤ 2π'
            dtd_theta.attrs['calculations'] = 'Antitangent of the hypotenuse of bead displacement within the xy-plane, which is found by taking the voltage difference when bead displacement is occurring (top of square wave) vs. no displacement (bottom of square wave) for both X and Y components.'
            # The difference in voltage from mean_background_corrected_datum (volts) when displacement is occurring vs. the neurtal state for movement in both x and y directions relative to 0° on the xy-plane. Antitanget of this difference gives theta.'
            dtd_theta.attrs['units'] = 'Degrees'
            dtd_xt_volt.attrs['Normalized_detected_displacment'] = 'Detected displacement of the bead in the xy-plane'
            dtd_xt_volt.attrs['units'] = 'Volts/Volts (V/V)'
            dtd_xt_volt.attrs['calculations'] = 'The average difference in voltage when displacement is occurring (top of square wave) vs. when no displacement is occurring (bottom of square wave), relative to detected movement. That is: dX / (cos(theta)).'
            #'The voltage difference in mean_background_corrected_datum (volts) when displacement is occurring vs. neutral state. This difference is Xp and is placed over the length of the hypotenuse of movement (cos(theta)).' 
            dtd_xt_volt.attrs['units'] = 'Volts/nanometer (V/nm)'
            dtd_recp.attrs['reciprocal_time_constant'] = 'Reciprocal time constant for bead trajectory is the exponential decay or rise calculated between two user identified points on a mean background corrected square wave of this displacement datum'
            dtd_recp.attrs['units'] = 'per seconds (1/s)'
            dtd_spring.attrs['spring_constant_of_optical_tweezers_per_displacement'] = 'The Hookian spring constant of the trapping laser.'
            #'Utilizing the reciprocal time constant, the Hooke spring constant is calculated for this displacement. irl: http://purl.obolibrary.org/obo/NCIT_C50187 | http://purl.obolibrary.org/obo/NCIT_C64638'
            dtd_spring.attrs['units'] = 'nanoNewtons/micrometer (nN/μm)'
            # Iterate through calculated data and insert into numpy array for writing to matlab file.
            dft[i+1,0] = dis[i]
            dft[i+1,1] = A[i]
            dft[i+1,2] = Xt_predicted[i]
            dft[i+1,3] = theta_lst[i]
            dft[i+1,4] = Xt_voltage[i]
            dft[1,5] = qpd_slope
            dft[2,5] = qpd_slope
            dft[3,5] = qpd_slope
            dft[4,5] = qpd_slope
            dft[i+1,6] = fitstat.m
            dft[i+1,7] = spring_constant
                                           
            # This stores square wave info and prepares to plot it
            plt.plot(range(len(Xb[0])), Xb[i])
        # Plot the square waves        
        plt.title('Square Waves')
        plt.xlabel('Data Points')
        plt.ylabel('QPD Intensity (V/V)')
        plt.savefig(measure_date+'Square_waves'+'.png', bbox_inches='tight')
        plt.show()
    file_iteration += 1


# Write both matlab files
scipy.io.savemat('calibration_optical_tweezers'+measure_date+'_Raw.mat', {'raw_data': df}, oned_as='column')
scipy.io.savemat('calibration_optical_tweezers'+measure_date+'_Transformed.mat', {'Transformed_data': dft}, oned_as='column')

# Generate mean spring constant and mean reciprocal time constant
mean_spring = mean_ot_cal_param.create_dataset('mean_spring_constant', data = (statistics.mean(sp_con_lst)))
mean_spring.attrs['mean_spring_constant'] = 'Mean of spring constants calculated at the four expected displacements'
mean_spring.attrs['units'] = 'nanoNewtons/micrometer (nN/μm)'
mean_recp =  mean_ot_cal_param.create_dataset('mean_reciprocal_time_constant', data = (statistics.mean(recp_time_lst)))
mean_recpsd = mean_ot_cal_param.create_dataset('mean_reciprocal_time_constant_sd', data = np.std(recp_time_lst))
mean_recpsd.attrs['mean_reciprocal_time_constant_sd'] = "A standard deviation calculation is a descriptive statistics calculation defined as the square root of the variance. Also thought of as the average distance of each value to the mean. irl: http://purl.obolibrary.org/obo/OBI_0200121"
mean_recp.attrs['mean_reciprocal_time_constant'] = 'Mean of reciprocal time constants calculated at the four expected displacements'
mean_recp.attrs['units'] = 'per seconds (1/s)'
# Close and write to hdf5 file
h.close()

print('Done')