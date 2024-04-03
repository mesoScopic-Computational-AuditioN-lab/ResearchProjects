function [time,trigger,command_delay] = vpx_custom_trigger2(trigger, eyetracking)
%% vxp_custom_trigger
%   
%	No check, just sends trigger if eyetracking
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
end

if eyetracking == true
    [~, time, command_delay] = vpx_SendCommandString_clocked('dataFile_insertmarker %c', trigger);
else
    time = nan;
    command_delay = nan;
end
return