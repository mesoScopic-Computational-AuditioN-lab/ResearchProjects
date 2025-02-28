%% SETTINGS - PUPILLOMETRY - EYE-TRACKER

function SETUP_eyetracker(eyetracking)
    %----------------------------------------------------------------------
    %                       set up Arrington
    %----------------------------------------------------------------------
    
    arguments
        eyetracking (1,1) logical = true;
    end
    
    if eyetracking == 1
        mainpath = pwd;
        mainpath = mainpath(1:end-36);
        addpath(genpath([mainpath '/Toolboxes/SacLab/']))

        vpx_Initialize; %start up arrington
        % vpx_Calibrate(12); % calibrate

        % make sure that we get gaze data from the Eyelink
        vpx_GetDataQuality;

        global eye
        eye = 0;

        % open file to record data to
        formatOut = 'yymmdd_HHMMSS';
        %svstring = ['dataFile_NewUniqueExtension' ' "EYEMOVPAT_s' num2str(subject) '_g' num2str(group) '.txt"'];
        % ADD PPNUM AND RUN 'S02_R2_'
        svstring = ['dataFile_NewName "Data\pred_pupil\' num2str(datestr(now, formatOut))  '.txt"'];  % update: added \pred_pupil\ see if folder creation is automatic or we have to manually add

        vpx_SendCommandString('setPriority High');
        vpx_SendCommandString('dataFile_includeRawData Yes');
        vpx_SendCommandString('dataFile_Pause True');
        vpx_SendCommandString(svstring); % how to save the timestamp when file is started

        vpx_SendCommandString('dataFile_asynchStringData NO') ; %record data synchronoously
        vpx_SendCommandString('dataFile_Pause False');
        %vpx_SendCommandString('dataFile_InsertString "CALSTART"'); %send onset of whole task
        vpx_SendCommandString ('parallaxCorrection_Slope 0.8'); % Pupil Diameter Calibration
    end

return

% %% following commands for reference only
% % send commands
% vpx_SendCommandString(['dataFile_insertmarker O']);
% vpx_SendCommandString(['dataFile_insertmarker T']);
% vpx_SendCommandString(['dataFile_insertmarker P']);
% vpx_SendCommandString(['dataFile_insertmarker F']);
% vpx_SendCommandString(['dataFile_insertmarker' num2str(cnd_matrix(trial,1))]);
% 
% % save eye-tracking data
% gazePoint=struct('x',[],'y',[],'t',[],'Px',[],'Py',[]); % Get the gaze point from ViewPoint.
% 
% if eye_tracking
%     [gazePoint.x(end+1),gazePoint.y(end+1)]=vpx_GetGazePoint(eye);
%     [gazePoint.Px(end+1),gazePoint.Py(end+1)]=vpx_GetGazePointSmoothed(eye);
%     gazePoint.t(end+1)=toc(start_trial_time);
% end
% 
% if eye_tracking
%     eye_data{trial}=gazePoint;
% end
% 
% save(filename,'eye_data')
% 
% %% close datafile
% vpx_SendCommandString('dataFile_Close');

