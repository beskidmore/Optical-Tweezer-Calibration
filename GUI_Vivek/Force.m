function varargout = Force(varargin)
% FORCE MATLAB code for Force.fig
%      FORCE, by itself, creates a new FORCE or raises the existing
%      singleton*.
%
%      H = FORCE returns the handle to a new FORCE or the handle to
%      the existing singleton*.
%
%      FORCE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FORCE.M with the given input arguments.
%
%      FORCE('Property','Value',...) creates a new FORCE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Force_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Force_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Force

% Last Modified by GUIDE v2.5 07-Aug-2013 14:35:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Force_OpeningFcn, ...
                   'gui_OutputFcn',  @Force_OutputFcn, ...
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


% --- Executes just before Force is made visible.
function Force_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Force (see VARARGIN)

% Choose default command line output for Force
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Force wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Force_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function var_visc_Callback(hObject, eventdata, handles)
% hObject    handle to var_visc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of var_visc as text
%        str2double(get(hObject,'String')) returns contents of var_visc as a double
handles.var1=get(handles.var_visc, 'string');
handles.var1=str2double (handles.var1);
disp(['Viscocity is ' num2str(handles.var1) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function var_visc_CreateFcn(hObject, eventdata, handles)
% hObject    handle to var_visc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function var_a_Callback(hObject, eventdata, handles)
% hObject    handle to var_a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of var_a as text
%        str2double(get(hObject,'String')) returns contents of var_a as a double
handles.var2=get(handles.var_a, 'string');
handles.var2=str2double (handles.var2);
disp(['Radius of the bead is ' num2str(handles.var2) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function var_a_CreateFcn(hObject, eventdata, handles)
% hObject    handle to var_a (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function var_h_Callback(hObject, eventdata, handles)
% hObject    handle to var_h (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of var_h as text
%        str2double(get(hObject,'String')) returns contents of var_h as a double
handles.var3=get(handles.var_h, 'string');
handles.var3=str2double (handles.var3);
disp(['Height of the bead is ' num2str(handles.var3) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function var_h_CreateFcn(hObject, eventdata, handles)
% hObject    handle to var_h (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function var_TC_Callback(hObject, eventdata, handles)
% hObject    handle to var_TC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of var_TC as text
%        str2double(get(hObject,'String')) returns contents of var_TC as a double
handles.var4=get(handles.var_TC, 'string');
handles.var4=str2double (handles.var4);
disp(['Time constant is ' num2str(handles.var4) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function var_TC_CreateFcn(hObject, eventdata, handles)
% hObject    handle to var_TC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Stiffness.
function Stiffness_Callback(hObject, eventdata, handles)
% hObject    handle to Stiffness (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.k= 6*pi*handles.var1*handles.var2*handles.var4*((1-(9/16)*(handles.var2/handles.var3)+(1/8)*((handles.var2/handles.var3)^3)+(-45/256)*((handles.var2/handles.var3)^4)+(-1/16)*((handles.var2/handles.var3)^5))^-1)*(10^12/10^9);
disp(['Stiffness of the trap is ' num2str(handles.k) 'pN/nm.'])
guidata(hObject, handles);


% --- Executes on button press in TVisc.
function TVisc_Callback(hObject, eventdata, handles)
% hObject    handle to TVisc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.TempVisc);




set(handles.TempVisc,'NextPlot','add');

title ('Temperature Vs Viscocity')

xlabel ('Temperature (C)')
ylabel ('Viscocity (microPa/s)')
xlim ([22 25]);
grid on;
Te=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70];
Vi=[1306.9, 1138.2, 1002.0, 890.3, 797.5, 719.5, 653.5, 596.3, 547.1, 504.2, 466.6, 433.4, 403.9];
 plot(handles.TempVisc, Te, Vi)


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
files = dir('*.txt');
for i=1:length(files)
    fileName = [files(i).name];
    [fname, trash]= strtok (files(i).name, '.');   
    handles.dataStruct.(fname) = dlmread(files(i).name, '\t', 1,0);

end
disp (['Opened all files in folder!'])
guidata(hObject, handles);

function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double
handles.var5=get(handles.edit5, 'string');
handles.var5=str2double (handles.var5);
disp(['Slope is ' num2str(handles.var5) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Add2Struct.
function Add2Struct_Callback(hObject, eventdata, handles)
% hObject    handle to Add2Struct (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  
    if isfield(handles,'ForceData')>0
    handles.jk=length(handles.ForceData);
    disp(['Existing structure has ' num2str(length(handles.ForceData)) ' data points'])
    else
    
    handles.jk=0;
    end

    fNames=fieldnames (handles.dataStruct);
    
i=1;

for i=1:length(fNames)
    QPDdata=handles.dataStruct.(fNames{i});
     handles.ForceData(i+handles.jk).Stiffness=(handles.k);
   handles.ForceData(i+handles.jk).Slope=(handles.var5);
    handles.ForceData(i+handles.jk).CellInfo=(handles.var6);
    handles.ForceData(i+handles.jk).Filename=fNames(i);
    handles.ForceData(i+handles.jk).QPDdata=QPDdata;
    
    
    
end


   
    
    

    ForceData=handles.ForceData;
save('ForceData.mat',  'ForceData');
disp (['Saved new data element to Structure! Structure now has ' num2str(length(ForceData)) ' elements.'])

guidata(hObject, handles);


    



function Date_Callback(hObject, eventdata, handles)
% hObject    handle to Date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Date as text
%        str2double(get(hObject,'String')) returns contents of Date as a double
handles.var6=get(handles.Date, 'string');

disp(['Cell Info entered by the user is ' (handles.var6) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function Date_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Date (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function exptnum_Callback(hObject, eventdata, handles)
% hObject    handle to exptnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of exptnum as text
%        str2double(get(hObject,'String')) returns contents of exptnum as a double
handles.var7=get(handles.exptnum, 'string');
handles.var7=str2double (handles.var7);
disp(['Experiment number is ' num2str(handles.var7) '.'])
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function exptnum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to exptnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in LoadSTR.
function LoadSTR_Callback(hObject, eventdata, handles)
% hObject    handle to LoadSTR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
load('ForceData.mat');
handles.ForceData=ForceData;
disp(['Loaded data structure with ' num2str(length(ForceData)) ' elements'])

guidata(hObject, handles);


% --- Executes on button press in AppendProp.
function AppendProp_Callback(hObject, eventdata, handles)
files = dir('*.txt');
for i=1:length(files)




%# read lines
fid = fopen(files(i).name,'rt');
C = textscan(fid, '%s ', 'Delimiter','\n'); Datadump = C{1};
fclose(fid);


DD1=textscan(Datadump{1,1}, '%s', 'Delimiter', '\t');DD1= DD1{1};DD1T=transpose(DD1);
DD2=textscan(Datadump{2,1}, '%s', 'Delimiter', '\t');DD2= DD2{1};DD2T=transpose(DD2);
DD3=textscan(Datadump{3,1}, '%s', 'Delimiter', '\t');DD3= DD3{1};DD3T=transpose(DD3);
DD4=textscan(Datadump{4,1}, '%s', 'Delimiter', '\t');DD4= DD4{1};DD4T=transpose(DD4);
DD5=textscan(Datadump{5,1}, '%s', 'Delimiter', '\t');DD5= DD5{1};DD5T=transpose(DD5);
DD6=textscan(Datadump{6,1}, '%s', 'Delimiter', '\t');DD6= DD6{1};DD6T=transpose(DD6);
TitleData=horzcat (DD1T,DD3T,DD5T);
NumericData= horzcat(DD2T, DD4T, DD6T);
AllData=vertcat (TitleData, NumericData);
handles.ForceData(i+handles.jk).Properties=AllData;
handles.ForceData(i+handles.jk).PropertiesFileName=files(i).name;

end
ForceData=handles.ForceData;
save('ForceData_incProp.mat',  'ForceData');
disp(['Added Properties to the Structure ' num2str(length(ForceData)) ' elements'])

guidata(hObject, handles);
% hObject    handle to AppendProp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
