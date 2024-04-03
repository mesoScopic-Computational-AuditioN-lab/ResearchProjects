% Screen('Preference', 'SkipSyncTests', 1);
%-------------- parameters for the lab setup --------------------%
screen_window=[1280 1024];% size in pixels
screen_size=[47.2 30];% size in cm
pix_per_deg=round(mean(screen_window./screen_size)); %pixels per degree

data_name = input('ptc number: ','s');
% data_name= 'tst';
filename=[ data_name '.mat'];

eye_tracker = 0; % 1 if eyetracker is present, 0 for testing purposes

if eye_tracker
    vpx_Initialize; %start up arrington
    %        vpx_Calibrate(12);

    % make sure that we get gaze data from the Eyelink
    vpx_GetDataQuality %think about saving somthing about this

    global eye
    eye = 0;

    % open file to record data to
    formatOut = 'yymmdd_HHMMSS';
    %svstring = ['dataFile_NewUniqueExtension' ' "EYEMOVPAT_s' num2str(subject) '_g' num2str(group) '.txt"'];
    svstring = ['dataFile_NewName "Data\' num2str(datestr(now, formatOut))  '.txt"'];

    vpx_SendCommandString('setPriority High');
    vpx_SendCommandString('dataFile_includeRawData Yes');
    vpx_SendCommandString('dataFile_Pause True');
    vpx_SendCommandString(svstring); % how to save the timestamp when file is started

    vpx_SendCommandString('dataFile_asynchStringData NO') ; %record data synchronoously
    vpx_SendCommandString('dataFile_Pause False');
    %vpx_SendCommandString('dataFile_InsertString "CALSTART"'); %send onset of whole task
end

%-------------- setup psychtoolbox --------------------%
sca;    % Setup PTB with some default values
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 2);

screenNumber = max(Screen('Screens'));
screen_back=0.5; %black screen
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, screen_back, [], pix_per_deg, 2,[], [],  kPsychNeed32BPCFloat);

vbl = Screen('Flip', window);
% Numer of frames to wait before re-drawing
waitframes = 1;

fix_spot_color=[0 0 0];

% Flip to clear
Screen('Flip', window);
% Query the frame duration

[xCenter, yCenter] = RectCenter(windowRect);

ifi = Screen('GetFlipInterval', window);

stimTimeSecs = 2;
stimTimeFrames = round(stimTimeSecs / ifi);

gazePoint=struct('x',[],'y',[],'t',[],'Px',[],'Py',[]);


% improve portability of your code acorss operating systems 
KbName('UnifyKeyNames');

% Define the keyboard keys that are listened for.
escapeKey = KbName('ESCAPE');% the escape key as a exit/reset key

% specify key names of interest in the study
activeKeys = [KbName('LeftArrow') KbName('RightArrow')];

start_trial_time=cputime;
RestrictKeysForKbCheck(activeKeys);
% suppress echo to the command line for keypresses
ListenChar(2);

% for frame = 1:stimTimeFrames - 1
flip = 1;
while flip == 1
    [ keyIsDown, keyTime, keyCode ] = KbCheck; 
    if(keyIsDown), break; end
    
    % Flip to the screen
    vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);

    if eye_tracker
        [gazePoint.x(end+1),gazePoint.y(end+1)]=vpx_GetGazePoint(eye);
        [gazePoint.Px(end+1),gazePoint.Py(end+1)]=vpx_GetGazePointSmoothed(eye);
    end
    gazePoint.t(end+1)=cputime-start_trial_time;
    
%     Screen('DrawDots', window, [xCenter; yCenter], 10, fix_spot_color, [], 2);
%     vbl = Screen('Flip', window);
    %%%%% check for keyboard response
%     [keyIsDown,secs, keyCode] = KbCheck;
%     if keyCode(escapeKey) % use escape to end the trial
%         if eye_tracker
%                 vpx_SendCommandString('dataFile_Close');
%         end
%         dips("Esc pressed")
%         flip = 0;
%         ShowCursor;
%         sca;
%         return
%     end
    [ keyIsDown, keyTime, keyCode ] = KbCheck; 
    if(keyIsDown), break; end
end
RestrictKeysForKbCheck;
ListenChar(1)
sca

save(join(['eyetracking_data/' filename]),'gazePoint')
