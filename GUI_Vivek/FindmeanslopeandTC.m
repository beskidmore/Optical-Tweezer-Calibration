load('AvgSW.mat')
load('SquareWave.mat')
%%


for i=1:4
    
trialtc(i)=Avg2SW(i).TimeConstant;

end

TC=mean(trialtc)

for i=1:12
    
trialslp(i)=SWl(i).slope(1);

end

m=mean(trialslp)