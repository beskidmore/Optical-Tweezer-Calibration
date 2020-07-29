# -*- coding: utf-8 -*-
### Code for calibration of optical tweezers                                                      ###
### Author: Benjamin Skidmore, based on matlab code by Vivek Rajasekharan                         ###
### Last edit Date: 09/2/2018                                                                    ###
### Writen in Pyton 3.6                                                                           ###
### Referenced equations are from:                                                                ###
###Rajasekharan, V., Sreenivasan, V. K. A., & Farrell, B. (2017).                                 ###
### Force Measurements for Cancer Cells. Methods in Molecular Biology (Clifton, N.J.),            ###
### 1530, 195–228. https://doi.org/10.1007/978-1-4939-6646-2_12                                   ###

## Dependencies: matplotlib, math, numpy, plotly, h5py
# numpy, matplotlib, plotly will need to be installed by user separatly and in the path

# Imports libraries #
import matplotlib.pyplot as plt
import numpy as np
import math
import os  
from plotly.offline import  plot
import plotly.graph_objs as go
import h5py
import pandas as pd
import scipy.io

# Load dark signal for incorporation. eq. 11
with open('C:/Users/skidmore/Desktop/calibrate/Data/10th/P/X1_properties.txt') as file2:
    next(file2)
    for lne in file2:
        lne.strip(' ')
        dark_x = float(lne.split('\t')[3])
        dark_y = float(lne.split('\t')[4])
        dark_sum = float(lne.split('\t')[5])
        break
file2.close()

# Set B based on height (pNsnm^-1)
# standardized at:      May be different.
Beta = 0.0000391 

# Set gain, this is almost always 10 but if different users are able to type it as input
#G = int(input('Enter Gain (1, 3, or 10): '))
G = 10

# Radius of Bead (micrometer (um)). Nominal diameter of 4um. NOT USED. METADATA
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

# Gather info for writing input data files to HDF5 format
#measure_date = str(input('What day was the experiment done (mm_dd_yyyy):'))
measure_date = str(10_10_2013)
df1 = pd.DataFrame({})

# Path to working directory 
#rootFolder = input('Enter path to Working directory:')
# if in set directory can utilize this  C:/Users/skidmore/Desktop/calibrate/Data/10th/SW
rootFolder = "C:/Users/skidmore/Desktop/calibrate/Data/10th/SW/"

# Prepare to write to hdf5 format
## Data structure: Date -> Raw|Transformed -> Raw.Displacement -> Raw.Displacement.*dis* -> .X|.Y|.S   X 4 (for each *dis)
##                                          -> Raw.Dark        -> Raw.Dark.X|.Y|.S
##                                          -> Raw.Gain
h = h5py.File(measure_date+'.h5', 'w')
# Write 1 dimensional arrays
dset_raw = h.create_group('Raw_Data')
dset_trans = h.create_group('Transformed_Data')
draw_dis = dset_raw.create_group('Displacement')
draw_dark = dset_raw.create_group('Dark')
draw_dark.create_dataset('Dark_X', data=dark_x)
draw_dark.create_dataset('Dark_Y', data=dark_y)
draw_dark.create_dataset('Dark_sum', data=dark_sum)
dset_raw.create_dataset('Gain', data=G)
##dset = h.create_dataset(measure_date, (5,8),
                        
                        
'''dtype=[('Date', '<f8'),
 ('Measured X', '<f8'),
 ('Measured Y', '<f8'),
 ('Measured Sum', '<f8'),
 ('Dark_X', '<f8'),
 ('Dark_Y', '<f8'),
 ('Dark_sum', '<f8'),
 ('Gain', '<f8')])'''
# First matlab file, measured data, experimental variable (gain, dark influence, etc.)
column_count = 1
df = np.zeros((5,8), dtype=np.object)
df[0,0] = measure_date
df[0,1] = 'Measured X'
df[0,2] = 'Measured Y'
df[0,3] = 'Measured Sum'
df[0,4] = 'Dark_X'
df[1,4] = dark_x
df[0,5] = 'Dark_Y'
df[1,5] =  dark_y
df[0,6] = "Dark_sum"
df[1,6] = dark_sum
df[0,7] = 'Gain'
df[1,7] = G
## Second matlab file initilization and organization, transformed and calculated data
dft = np.zeros((5,8), dtype=np.object)
dft[0,0] = measure_date
dft[0,1] = 'Transformed Data'
dft[0,2] = 'Actual Displacement'
dft[0,3] = 'Theta (angle of movement)'
dft[0,4] = 'Xt_voltage'
dft[0,5] = 'QPD Slope of line (V/V/nm)'
dft[0,6] = 'Time Constant'
dft[0,7] = 'Spring Constant kp'

# Iterate through all four files in working directory. load data and imlement algorithms
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
                
                # identifis those files that end with .txt and makes the filepath to these files
                shpName = os.path.splitext(name)[0]
                absFile = os.path.abspath(os.path.join(root,name))
                # opens all files that end with .txt in the working directory
                if absFile.endswith('.txt'):    
                    # Write original measurement data to HDF5 file     
                    d = pd.read_table(absFile)
                    title = "{0}_{1}".format((measure_date), str(dis_title))
                    #h = h5py.File(title+'.hdf5', 'w')
                    #dset = h.create_dataset('data', data=d)
                    #h.close()
                    
                    # Load data for use in calculations/transormation, etc. 
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
                    dis_draw_dis.create_dataset('Measured_X', data = xs)
                    dis_draw_dis.create_dataset('Measure_Y', data = ys)
                    dis_draw_dis.create_dataset('Measured_sum', data = qpd_sum)
                    #dset_raw.create_dataset('Measured_X', data = xs)
                    column_count += 1
                    file1.close()
                    break

    print(displacement) 
       
    # Account for dark here.
    # subtracts dark*G from every datapoint in the X / Y columns of read files. 
    xs = [float(x) - (dark_x*G) for x in xs]
    ys = [float(y) - (dark_y*G) for y in ys]
    # subtracts dark from datapoints in sum column of read files then mulitplies by gain
    # Notice difference in where gain is implemented.
    qpd_sum = [(float(q) - dark_sum)*G for q in qpd_sum]
    
    
    # periodicly average every 16000 datapoints. 16000 is the number of datapoints in a period of the square wave.
    # produces variables that will be reused.
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
        
    # Finds sum datapoints periodwise -> [a,b,c] + [a',b',c'] ... -> [a+a',b+b',...]. 
    # Then find the average of each datapoint by dividing by # of periods.
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

    ## Now start calculation of slope eq 12.: Mqpdx = DXl/Sl/Dx
    ## Units in V/V/nm. see fig 8a.
    ## Xp = DXL/Sl
    ## Yp - DYl/Sl
    ## Dx is known during calibration and corresponds to displacement (nm)
    
    # Use average of points to find true average of square wave points of interest
    # makes list of -y[1-3000] and divides it by qpd_sum_avg, 10*s_avg[1:3000]
    # Finds ratio of -y/sum pointwise for wave zeropoints (p1) and high/low points (p2) _|--|_
    p1 = [(float(y)*-1) / (float(s)) for y,s in zip(y_avgs[1:3000], s_avgs[1:3000])]
    p2 = [(float(y)*-1) / (float(s)) for y,s in zip(y_avgs[6000:10000], s_avgs[6000:10000])]

    # The average is taken across these points first (p1) 1-3000, then (p2) 6000-10000
    p1 = float(sum(p1))/float(len(p1))
    p2 = float(sum(p2))/float(len(p2))

    # Find difference in average top of square wave and bottom of square wave. _|--|_
    Xp = (p2 - p1)

    # makes list of -x[1-3000] and divides it by qpd_sum_avg, 10*s_avg[1:3000] 
    # Finds ratio of -x/sum pointwise for wave zeropoints (p1) and high/low points (p2) _|--|_
    yp1 = [float(x)*-1 / float(s) for x,s in zip(x_avgs[1:3000], s_avgs[1:3000])]
    yp2 = [float(x)*-1 / float(s) for x,s in zip(x_avgs[6000:10000], s_avgs[6000:10000])]
    
    #the average is taken across these points either 1-3000, or 6000-10000
    yp1 = float(sum(yp1))/float(len(yp1))
    yp2 = float(sum(yp2))/float(len(yp2))
    
    # find difference in top of square wave and bottom of square wave. _|--|_
    Yp = (yp2 - yp1)
    
    # Now we move on to identifiying theta, the angle corresponding to the distance the bead moved in circular 2D plane.
    # Get theta in degrees.
    theta = math.degrees(math.atan(Yp/Xp))
    
    ## the following portion is modeled after matlab:
    ## Acst = Xp / cosd(theta)
    ## qpd_signal2 = -y_avgs / (10*math.cos(theta)*s_avgs)
    ## qpd_sig = qpd_sig2 - mean(qpd_sig2[1000:3000])
    
    # because we are interested in the hypotenuse, true distance, displacement.
    # This differs from the matlab version (cosd(theta)) because python handles input to cos as degrees. 
    # This is Xt_voltage
    acst = Xp / math.cos(math.radians(theta))

    # create list of normalized qpdsignal using the hypotenuse distance the bead moved.
    # This will be used to identify the measured displacement.
    qpdsignal2 = []
    for i in range(0,len(s_avgs)):
        ### The matlab code has this as -y_avgs/ (10*cosd(theta)*s_avgs).
        qpdsignal2.append(-(y_avgs[i])/(math.cos(math.radians(theta))*s_avgs[i]))
    qpdsignal = []
    qpd_lower_avg = np.mean([i for i in qpdsignal2[1000:3000]])
    for i in range(0, len(qpdsignal2)):  
        qpdsignal.append(qpdsignal2[i] - qpd_lower_avg)
       
        
    ## Now that we have Xp and Yp it is possible to determine slope Mqpdx/Mqpdy
    ## Should generate list of Xp and Yp from each datafile for comparison against displacement (known in calibration)
    ## Plot these Xps vs displacement and plot Yps against distplacement
    # add displacements to displacements list in order of those seen. do same to XP_points.
    dis.append(displacement)
    
    dis_tup.append((displacement, Xp, Yp, theta, acst, qpdsignal, title))
    
    ## Plot dx/dl/displacement and get slope, plots measured displacement using this slope, plots square waves, 
    ## Allows for selection of two points for the identification of time constant slope.
    # only plots if all four data files have been seen. Can always change this to a variable if necessiary
    if len(dis) == 4:
        # order displacement from least to greatest 
        dis.sort()
        count = 0
        # creates corresponding y coordinate for (displacement, QPD intensisty)
        # make Xb list of lists. holding 16000 X 1 for each datafile loaded (4)    
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
        
        Xt_voltage = acst_lst
        
        # Plots Acst vs displacements
        # Uses least squares to fit the line to datapoints for dx/dl/displacment in V/V/nm
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
        plt.show()         
        
        ## Xt_predicted. This is (Acst at each dis. - lineslop(2))/lineslope(1)
        print('dis', dis)
        for i in range(len(dis)):
            Xt_predicted.append((acst_lst[i] - fit[1])/fit[0])
        # Begin writing datafile
        datafile = open('datafile.txt', 'w')
        datafile.write('Displacement'+'\t'+'Xt_predicted'+'\t'+
                       'Theta'+'\t'+'Xt_voltage'+'\t'+'QPD slope of line (V/V/nm)'+
                       '\t'+'Time constant'+'\t'+'Spring constant kp'+'\n')
         
        ## Identify time constant, that is slope of line between two selected points using least squares.
        # Use this slope (time constant), m, to identify the Spring constant, kp, by multiplying by 
        # Beta, found from Stoke's-Faxen Law, eq. 8. Multiplication: eq. 7 
        # Write values to correct columns in datafile
        def fitstat(time, xb4):
            print('lengths', len(time), len(xb4))
            m,b = np.polyfit(time, xb4, 1)
            print('Slope of selected points', m)
            datafile.write(str(dis[i])+'\t'+str(Xt_predicted[i])+'\t'+
                           str(theta_lst[i])+'\t'+str(Xt_voltage[i])+'\t'+str(qpd_slope)+
                           '\t'+str(m)+'\t'+ str(m*Beta)+'\n')
            fitstat.m = m
            
        # Create a numpy array of Xb for easy list comprehension parsing and indexing     
        A = np.array(Xb)
        proplst = []
        spoint = []
        mlst = []
        sp_con_lst =[]
        # Plot of displacements in nm vs. datapoints (16000)
        # This is separate from next "for" loop because we plot square waves there and its syntax is the same
        for i in range(len(Xb)):
            plt.plot(list(range(len(A[i]))), [x*(1/qpd_slope) for x in A[i]])
        plt.title('Measured Displacemnt')
        plt.xlabel('Data Points')
        plt.ylabel('Displacement (nm)')
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
    
            # For visulaizing and manualy slecting points for identifying slope.            
            trace = go.Scatter(x = xlst, y = ylst)
            tra = [trace]
            plot(tra, filename='Displacement_%s.html' % str(dis[i]))
            # Manual selection of points
            #x1 = int(input('What is the First x-coordinate?'))
            #x2 = int(input('What is the Second x-coordinate?'))
            ## Only manually select one time, may standardize answers but risk missing line if data is incorrect
            if i == 0:
                x1 = 4004
                x2 = 4114
                #x1 = int(input('What is the First x-coordinate?'))
                #x2 = int(input('What is the Second x-coordinate?')) 
            y1 = ylst[x1-3500]
            y2 = ylst[x2-3500]
            
            # Fitting part of the function. Identify regression coefficent with time and selected points(x1,x2) 
            # Time is a variable that is a reversion back to Hz for the range of selected points.
            time = np.arange(0, (x2-x1)*.000005, .000005)
            xb4 = []
            # Because we don't reference trial1 we must convert these using the same methodology (1-i/acst2)
            for j in A[i, x1:x2]:
                xb4.append(np.log(abs(1-float(j)/float(acst2))))
            # The way that np.arange works may give a different number of points than required.
            while len(time) != len(xb4):
                print('Those datapoints broke python, please choose datapoints further apart or closer together.')
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
                
            
            # Iterate through calculated data and insert into numpy array for writting to matlab file.
            dft[i+1,0] = dis[i]
            dft[i+1,1] = A[i]
            dft[i+1,2] = Xt_predicted[i]
            dft[i+1,3] = theta_lst[i]
            dft[i+1,4] = Xt_voltage[i]
            dft[1,5] = qpd_slope
            dft[i+1,6] = fitstat.m
            print(fitstat.m*Beta)
            spring_constant = fitstat.m*Beta
            print(spring_constant)
            dft[i+1,7] = spring_constant
                                           
            # This stores square wave info and prepares to plot it
            plt.plot(range(len(Xb[0])), Xb[i])
        # Plot the square waves        
        plt.title('Square Waves')
        plt.xlabel('Data Points')
        plt.ylabel('QPD Intensity (V/V)')
        plt.show()
        
    file_iteration += 1
    
    

#Write both matlab files
scipy.io.savemat(measure_date+'_Raw', {'raw_data': df}, oned_as='column')
scipy.io.savemat(measure_date+'_Transformed', {'Transformed_data': dft}, oned_as='column')

#h = h5py.File(measure_date+'_Raw'+'.hdf5', 'w')
#dset = h.create_dataset('Raw Data', data=df)
h.close()

#h2 = h5py.File(measure_date+'_Transformed'+'.hdf5', 'w')
#dset = h2.create_dataset('Transformed Data', data = dft)
#h2.close()
print(df.shape, dft.shape)
datafile.close()
print('Done')
