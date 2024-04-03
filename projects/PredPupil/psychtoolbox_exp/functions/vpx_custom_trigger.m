function [time,trigger,command_delay] = vpx_custom_trigger(trigger, eyetracking, closing)
    %% vxp_custom_trigger
    %   
    %	Checks fixation of subjects gaze using an arrington eye-tracker.
    %
    %   USAGE 
    %		checkFixation(time, false);
    %
    %   IMPORTANT: 
    %       - returns time and trigger
    %   
    %------------------------------------------------------
    
    arguments
        trigger string;
        eyetracking (1,1) logical = true;
        closing (1,1) logical = false;
    end
    if ~closing
        if eyetracking == true
%             vpx_SendCommandString(['dataFile_insertmarker T']);
            [time, command_delay] = vpx_SendCommandString_clocked(join(['dataFile_insertmarker ', trigger]));
        else
            time = nan;
            command_delay = nan;
        end
    else
        vpx_SendCommandString('dataFile_Close');
    end
return