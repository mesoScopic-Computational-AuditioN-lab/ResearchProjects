clear all; close all; clc; commandwindow;
checkvers=version;
if str2double(checkvers(end-5:end-2))>2013;
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));
else
    rand('twister',sum(100*clock)); %#ok<RAND>
end
Screen('Preference', 'VisualDebuglevel',3);
oldEnableFlag = Screen('Preference', 'SuppressAllWarnings', 1);
HideCursor;ListenChar(0);