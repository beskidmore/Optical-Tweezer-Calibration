 %% Determine time when force is zero 
Stime =25; % begining of range time
Etime= 50; % end of range time
initialtime=0;
[cslope] = cumslope(QPDnmXc,Stime,Etime,period1,initialtime); % call function to calculate cslope cumslope
%% Plot time when force is zero This uses interactive object 
Stime=25;
Etime=50;
close all;
[dcm_obj,zeroforcetime]= plot_cslope(cslope, period1,Stime,Etime);
zeroTime=zeroforcetime.Position(1,1);