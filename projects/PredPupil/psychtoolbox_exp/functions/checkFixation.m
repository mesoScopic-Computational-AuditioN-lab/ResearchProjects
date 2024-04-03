function fixBol = checkFixation( fixateBull, bullrect, time2check, onlineCall, eyetracking)
    %% checkFixation
    %   
    %	Checks fixation of subjects gaze using an arrington eye-tracker.
    %
    %   USAGE 
    %		checkFixation(time, false);
    %
    %   IMPORTANT: 
    %       - returns boolean value, indicating whether enough data was
    %       collected during eyetracking
    %       - time should be given in seconds
    %       - performs calibration if not enough data is collected
    %       - indicate with boolean whether to use online calibration
    %   
    %------------------------------------------------------
    
    arguments
        fixateBull;
        bullrect;
        time2check (1,1) double {mustBePositive} = 0;
        onlineCall (1,1) logical = false;
        eyetracking (1,1) logical = true;
    end
    
    if eyetracking == 1
    if time2check == 0
        time2check = 5;
    end

    addpath(genpath([pwd '/functions/']));
    
    % variable assigment
    fixBol = false;
    gazePoint=struct('x',[],'y',[],'t',[],'Px',[],'Py',[]);
    vpx_SendCommandString(['dataFile_insertmarker O']); % send  

    
    % check if eye variable exists
    global eye % <- your need this!
    existing = ~isempty(eye) && eye;
    if existing
        eye = 0;
    end

    waitframes = 1;
    
    % define 
    % Flip to clear
    [window, screenrect]  =   Screen( 'OpenWindow', 0, round(0.5*(ones(1,3)*255)));
    vbl = Screen('Flip', window);
    Screen('DrawTexture',window,fixateBull,[],bullrect);
    % Query the frame duration
    ifi = Screen('GetFlipInterval', window);
    frames2check = round(time2check / ifi);
    
    % go through frames
%     vbl =[];
    for frm = 1:frames2check - 1
        Screen('DrawTexture',window,fixateBull,[],bullrect);
        vbl = Screen('Flip', window, vbl + (waitframes - 0.5) * ifi);
        
        % get eye-data
        [gazePoint.x(end+1),gazePoint.y(end+1)]=vpx_GetGazePoint(eye);
        [gazePoint.Px(end+1),gazePoint.Py(end+1)]=vpx_GetGazePointSmoothed(eye);
    end
    
    % check if enough data collected
    val2match = size(gazePoint.x,2)*0.75;
    count = 0;
    % go through value
    for i = 1:size(gazePoint.x,2)
        if gazePoint.x(i) >0 && gazePoint.y(i) >0
            count = count + 1;  % increase as fixation exists
        end
    end
    % check if enough data
    if count >= val2match
        fixBol = true;          % set boolean to true
    end
    
    % re-calibrate eyetracking
    if onlineCall ==  true
        disp("Online Calibration")
        if ~fixBol
            vpx_Calibrate(12);
        end
    end
    else
        fixBol = false;
    end
    
    sca;
return