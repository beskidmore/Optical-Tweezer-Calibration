function varargout = stiffness_GUI_FINAL_newdata(varargin)
% STIFFNESS_GUI_FINAL_NEWDATA MATLAB code for stiffness_GUI_FINAL_newdata.fig
%      STIFFNESS_GUI_FINAL_NEWDATA, by itself, creates a new STIFFNESS_GUI_FINAL_NEWDATA or raises the existing
%      singleton*.
%
%      H = STIFFNESS_GUI_FINAL_NEWDATA returns the handle to a new STIFFNESS_GUI_FINAL_NEWDATA or the handle to
%      the existing singleton*.
%
% 
% # ITEM1
% # ITEM2
% 
%      STIFFNESS_GUI_FINAL_NEWDATA('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in STIFFNESS_GUI_FINAL_NEWDATA.M with the given input arguments.
%
%      STIFFNESS_GUI_FINAL_NEWDATA('Property','Value',...) creates a new STIFFNESS_GUI_FINAL_NEWDATA or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before stiffness_GUI_FINAL_newdata_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to stiffness_GUI_FINAL_newdata_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help stiffness_GUI_FINAL_newdata

% Last Modified by GUIDE v2.5 19-Jun-2013 15:23:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @stiffness_GUI_FINAL_newdata_OpeningFcn, ...
                   'gui_OutputFcn',  @stiffness_GUI_FINAL_newdata_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
%

% --- Executes just before stiffness_GUI_FINAL_newdata is made visible.
function stiffness_GUI_FINAL_newdata_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to stiffness_GUI_FINAL_newdata (see VARARGIN)

% Choose default command line output for stiffness_GUI_FINAL_newdata
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes stiffness_GUI_FINAL_newdata wait for user response (see UIRESUME)
% uiwait(handles.figure1);
%%


% --- Outputs from this function are returned to the command line.
function varargout = stiffness_GUI_FINAL_newdata_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
dname = uigetdir('K:\Data\2013-test\10_October\10_9_2013\SW\1');
cd(dname)


% --- Executes on button press in Xt_Xb.
function Xt_Xb_Callback(hObject, eventdata, handles)
% hObject    handle to Xt_Xb (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  


fNeg=fieldnames (handles.negStruct);
fPos=fieldnames (handles.posStruct);
q=1;
for q=1:length(fNeg)
    
SW_period=16000; %no of datapoints per period of the square wave
Xdata=handles.negStruct.(fNeg{q}); %Choose the data to be used
L1=length(Xdata); %Length of data to be used for periodic averaging
psum_data=zeros(SW_period,3,'double');
 

 t=1;
 %NEGATIVE%Code for centering the wave: >6.25 for p and <6.15 for n
        

 wavestart=4001;
Xdatanew=Xdata(wavestart:L1,1:3);
 L=length(Xdatanew);
 % code for averaging
i=uint32(1);
k=double(0);
while (i+SW_period-1)<=L
    psum_data=psum_data+Xdatanew(i:(i+SW_period-1),:);
    i=i+SW_period;
    k=k+1;
end
pavg_data=psum_data/(k);

 
 p1=mean(-pavg_data(1:3000,2)./(10.*(pavg_data(1:3000,3))));
p2=mean(-pavg_data(6000:10000,2)./(10.*(pavg_data(6000:10000,3))));
 
 Xp_b.(fNeg{q})=p2-p1;
 
  Yp1=mean(-pavg_data(1:3000,1)./(10.*(pavg_data(1:3000,3))));
Yp2=mean(-pavg_data(6000:10000,1)./(10.*(pavg_data(6000:10000,3))));
 
 Yp_b.(fNeg{q})=Yp2-Yp1;
  Theta.(fNeg{q})=atand(Yp_b.(fNeg{q})/Xp_b.(fNeg{q}));

 
 Acst.(fNeg{q})=Xp_b.(fNeg{q})/cosd(Theta.(fNeg{q}));
QPDsignal2=-(pavg_data(:,2))./(10*cosd(Theta.(fNeg{q})).*((pavg_data(:,3))));
QPDsignal=QPDsignal2-mean (QPDsignal2(1000:3000));
Xb.(fNeg{q})=QPDsignal;

end

r=1;

for r=1:length(fPos)
SW_period=16000; %no of datapoints per period of the square wave
Xdata=handles.posStruct.(fPos{r}); %Choose the data to be used
L1=length(Xdata); %Length of data to be used for periodic averaging
psum_data=zeros(SW_period,3,'double');
 t=1;
 
 %POSITIVE%Code for centering the wave: >6.25 for p and <6.15 for n
 

 
 wavestart=4001;
Xdatanew=Xdata(wavestart:L1,1:3);
 L=length(Xdatanew);
 
  % code for averaging
i=uint32(1);
k=double(0);
while (i+SW_period-1)<=L
    psum_data=psum_data+Xdatanew(i:(i+SW_period-1),:);
    i=i+SW_period;
    k=k+1;
end
pavg_data=psum_data/(k);

 
 p1=mean(-pavg_data(1:3000,2)./(10.*(pavg_data(1:3000,3))));
p2=mean(-pavg_data(6000:10000,2)./(10.*(pavg_data(6000:10000,3))));
 
 Xp_b.(fPos{r})=p2-p1;
 
  Yp1=mean(-pavg_data(1:3000,1)./(10.*(pavg_data(1:3000,3))));
Yp2=mean(-pavg_data(6000:10000,1)./(10.*(pavg_data(6000:10000,3))));
 
 Yp_b.(fPos{r})=Yp2-Yp1;
  Theta.(fPos{r})=atand(Yp_b.(fPos{r})/Xp_b.(fPos{r}));

 
 Acst.(fPos{r})=Xp_b.(fPos{r})/cosd(Theta.(fPos{r}));
 QPDsignal2=-(pavg_data(:,2))./(10*cosd(Theta.(fPos{r})).*((pavg_data(:,3))));
QPDsignal=QPDsignal2-mean (QPDsignal2(1000:3000));
Xb.(fPos{r})=QPDsignal;
end

handles.x2axis=[-800,-500, 500,800];
handles.x2axis
handles.y2axis=[Acst.f800n,Acst.f500n ,Acst.f500p,Acst.f800p];
handles.y2axis
linslope=polyfit(handles.x2axis, handles.y2axis, 1);


fieldname = fieldnames (Xb);
z=1;

if isfield(handles,'SWl')>0
    k=length(handles.SWl);
    disp(['Existing structure has ' num2str(length(handles.SWl)) ' data points'])
else
    k=0;
end


for z=1:length(fieldname)
    
handles.SWl(z+k).Xt=(fieldname{z});
handles.SWl(z+k).Xt_predicted=(Acst.(fieldname{z})-(linslope(2) ))/(linslope(1) );

handles.SWl(z+k).Theta=Theta.(fieldname{z});


handles.SWl(z+k).Power_mW= handles.var1;
handles.SWl(z+k).bead_no = handles.var2;

handles.SWl(z+k).Xb=Xb.(fieldname{z});

handles.SWl(z+k).Xt_voltage=Acst.(fieldname{z});

handles.SWl(z+k).slope=linslope;



end
SWl=handles.SWl;
save('SquareWave.mat',  'SWl');
disp ('Found Xb and Xt!')
guidata(hObject, handles);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Power_Callback(hObject, eventdata, handles)
% hObject    handle to Power (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



handles.var1=get(handles.Power, 'string');
handles.var1=str2double (handles.var1);
disp(['Laser power at the objective is ' num2str(handles.var1) ' mW'])
guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of Power as text
%        str2double(get(hObject,'String')) returns contents of Power as a double


% --- Executes during object creation, after setting all properties.
function Power_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Power (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in loadexistingSW.
function loadexistingSW_Callback(hObject, eventdata, handles)
% hObject    handle to loadexistingSW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('SquareWave.mat');
handles.SWl=SWl;
disp(['Loaded SW structure with ' num2str(length(SWl)) ' data points'])

guidata(hObject, handles);


% --- Executes on button press in plotlinslope.
function plotlinslope_Callback(hObject, eventdata, handles)
% hObject    handle to plotlinslope (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


axes(handles.linslope_plot);
set(handles.linslope_plot,'NextPlot','add');

plot(handles.linslope_plot,handles.x2axis,handles.y2axis, 'r', 'marker' , 'o'); 

title ('QPD slope (Blue) and AOD slope(Red))')

xlabel ('displacement (nm)')
ylabel ('QPD intensity (V/V)')




% --- Executes on button press in loadfiles.
function loadfiles_Callback(hObject, eventdata, handles)
% hObject    handle to loadfiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 files = dir('*.txt');
for i=1:length(files)
    fileName = [files(i).name];
    [fname, trash]= strtok (files(i).name, '.');
    [number, indicator]= strtok (fname, 'p | n');
    if indicator =='p'
    handles.posStruct.(fname) = dlmread(files(i).name, '\t', 1,0);
    elseif indicator=='n'
       handles.negStruct.(fname) = dlmread(files(i).name, '\t', 1,0); 
    end
    
    
end
disp ('Loaded all files into workspace!')
guidata(hObject, handles);



function bead_no_Callback(hObject, eventdata, handles)
% hObject    handle to bead_no (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.var2=get(handles.bead_no, 'string');
%handles.var2=str2double (handles.var2);
disp(['Bead number is ' (handles.var2) '.'])

guidata(hObject, handles);
% Hints: get(hObject,'String') returns contents of bead_no as text
%        str2double(get(hObject,'String')) returns contents of bead_no as a double


% --- Executes during object creation, after setting all properties.
function bead_no_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bead_no (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton1.
function pushbutton1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function square_waves_plot_CreateFcn(hObject, eventdata, handles)
% hObject    handle to square_waves_plot (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



% Hint: place code in OpeningFcn to populate square_waves_plot


% --- Executes on button press in plotsquarewaves.
function plotsquarewaves_Callback(hObject, eventdata, handles)
% hObject    handle to plotsquarewaves (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.square_waves_plot);
set(handles.square_waves_plot,'NextPlot','add');





title ('Square Waves')

xlabel ('Data Points')
ylabel ('QPD intensity (V/V)')

nbr=length(handles.SWl)/4;

for i=1:length(handles.SWl)
    q=floor(1+((i-0.1)/4));
    p=length(handles.SWl)-q;
    
    plot(handles.square_waves_plot, handles.SWl(i).Xb, 'Color', [nbr*q/length(handles.SWl),0, p/length(handles.SWl)])
    
    
 
end
guidata(hObject, handles);


% --- Executes on button press in FindAvgs.
function FindAvgs_Callback(hObject, eventdata, handles)
% hObject    handle to FindAvgs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

jj=1;
    storage(:,4)=zeros(16000,1);
    
    
while jj<length(handles.SWl)
    storage(:,1)=storage(:,1)+handles.SWl(jj).Xb;
    storage(:,2)=storage(:,2)+handles.SWl(jj+1).Xb;
    storage(:,3)=storage(:,3)+handles.SWl(jj+2).Xb;
   storage(:,4)=storage(:,4)+handles.SWl(jj+3).Xb;
    jj=jj+4;
    
    
    
end
nbr=length(handles.SWl)/4;
ii=1;
for ii=1:4
    
    handles.Avg2SW(ii).XbAVG=storage(:,ii)/nbr
    handles.Avg2SW(ii).Power= handles.SWl(1).Power_mW;
    plot(handles.square_waves_plot,  handles.Avg2SW(ii).XbAVG, 'g');
    if ii==1
        handles.Avg2SW(ii).SWname= handles.SWl(ii).Xt;
    elseif ii==2
         handles.Avg2SW(ii).SWname= handles.SWl(ii).Xt ;
    elseif ii==3
         handles.Avg2SW(ii).SWname= handles.SWl(ii).Xt;
    elseif ii==4
          handles.Avg2SW(ii).SWname= handles.SWl(ii).Xt; 
    end
end
Avg2SW=handles.Avg2SW;
save('AvgSW.mat',  'Avg2SW');

 guidata(hObject, handles);


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in EstimateTC.
function EstimateTC_Callback(hObject, eventdata, handles)
% hObject    handle to EstimateTC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

for i=1:4

 
QPDsignal=handles.Avg2SW(i).XbAVG;% load averaged SW from structure

Acst=mean (handles.Avg2SW(i).XbAVG(7000:9000))-mean (handles.Avg2SW(i).XbAVG(1000:3000)); %Find Xt for this averaged SW
    
 



start1=handles.var3;
stop1=handles.var4;
 Xb1=(QPDsignal(start1:stop1));
 Xb1=Xb1';
 
 Xb3=1-(Xb1./Acst);
 Xb4=log(Xb3);
 
 %
 
 %Fitting part of the function
 time1 = [0.000000:0.000005:(length(Xb4)-1)*0.000005];
 fitstats=regstats (Xb4,time1, 'linear');

Estimatesn=fitstats.beta(2);

handles.Avg2SW(i).STATS=fitstats;
%Save TC to same structure
handles.Avg2SW(i).TimeConstant=Estimatesn;
end
Avg2SW=handles.Avg2SW;
save('AvgSW.mat',  'Avg2SW');
 guidata(hObject, handles);
 disp(['Updated structure to include time constants!'])



function startpt_Callback(hObject, eventdata, handles)
% hObject    handle to startpt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of startpt as text
%        str2double(get(hObject,'String')) returns contents of startpt as a double
handles.var3=get(handles.startpt, 'string');
handles.var3=str2double (handles.var3);
disp(['Start point is set to ' num2str(handles.var3) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function startpt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to startpt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end





% --- Executes during object creation, after setting all properties.
function stoppt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stoppt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in loadavgSW.
function loadavgSW_Callback(hObject, eventdata, handles)
% hObject    handle to loadavgSW (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

load('AvgSW.mat');
handles.Avg2SW=Avg2SW;
disp(['Loaded AVG SW structure with ' num2str(length(Avg2SW)) ' data points'])

guidata(hObject, handles);



function stppt_Callback(hObject, eventdata, handles)
% hObject    handle to stppt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of stppt as text
%        str2double(get(hObject,'String')) returns contents of stppt as a double
handles.var4=get(handles.stppt, 'string');
handles.var4=str2double (handles.var4);
disp(['Stop point is set to ' num2str(handles.var4) '.'])

guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function stppt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stppt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Estisep.
function Estisep_Callback(hObject, eventdata, handles)
% hObject    handle to Estisep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

QPDsignal=handles.Avg2SW(handles.var5).XbAVG;% load averaged SW from structure

Acst=mean (handles.Avg2SW(handles.var5).XbAVG(7000:9000))-mean (handles.Avg2SW(handles.var5).XbAVG(1000:3000)); %Find Xt for this averaged SW
    
 
 %To find start point and stop point,within the linear region; Plot the
 %function and click on start point first, then on end point, holding shift
 %key. Hit 'enter' when done. Start point is saved to start2 and stop point
 %is saved to stop2. 
 trial1=QPDsignal(3501:6500);
 trial1=trial1';
 trial2=trial1-trial1(1);
 trial3=1-(trial2./Acst)
 trial4=log(trial3)

 figure1=figure (11); hold on
  h = plot([1:3000],trial4 , 'r', 'marker','d');
  %xlim ([480 900])
  %ylim ([-6 0.3])
 dcm_obj = datacursormode(figure1);
set(dcm_obj,'DisplayStyle','datatip',...
    'SnapToDataVertex','off','Enable','on')

disp('Click on a line to display a data tip, then press Return.')
pause                            % Wait while the user does this.

Cursorposition = getCursorInfo(dcm_obj);
guidata(hObject, handles);
% %%
% 
% 
% 
handles.start2=3500+Cursorposition(1,2).DataIndex ;
handles.stop2=3500+Cursorposition(1,1).DataIndex;
disp(['User selected start point is ' num2str(handles.start2) ' and user selected stop point is ' num2str(handles.stop2) '.' ])

function avgswno_Callback(hObject, eventdata, handles)
% hObject    handle to avgswno (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of avgswno as text
%        str2double(get(hObject,'String')) returns contents of avgswno as a double

handles.var5=get(handles.avgswno, 'string');
handles.var5=str2double (handles.var5);
disp(['Plotting log (1-Xb/Xt) VS time for AVGed squarewave number ' num2str(handles.var5) ' from structure'])

guidata(hObject, handles);
% --- Executes during object creation, after setting all properties.
function avgswno_CreateFcn(hObject, eventdata, handles)
% hObject    handle to avgswno (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
