function [cslope] = cumslope(QPDnmXc,Stime,Etime,period1, initialtime )
clear eventname; clear cslope;
eventname=QPDnmXc(Stime*1/period1:1:Etime*1/period1,1); % Just look at section
%translating from time to counter period 1 was 0.0005 here 

cslope = zeros(200000-1,1); % set up matrix this was for 20 seconds may need larger array

    for i=1:1:size(eventname)-1;
        t= (1:1:i+1)./2000+initialtime-1/period1; % calculate time
        t=t';
        q=polyfit(t,eventname(1:1:i+1,1),1);  % fit to linear function polyfit is function in Matlab
        cslope(i,1)=q(1);
    end
    clear q;
end