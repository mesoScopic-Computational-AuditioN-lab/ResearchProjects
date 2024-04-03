%% sending triggers

run_trigger = join(['R' ,num2str(crun)]);
if eyetracking
    [~, online_trigger_tracker.t(end+1), online_trigger_tracker.timedelay] = vpx_SendCommandString_clocked(join(['R' ,num2str(crun)]);
end


%% testing return value of sendcommandstring

r = vpx_SendCommandString('dataFile_insertmarker %c', trigger);
disp(r);