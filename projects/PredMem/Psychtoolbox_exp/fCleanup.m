function fCleanup()
% Cleanup routine:
commandwindow;
KbQueueRelease;
ShowCursor;
Screen('CloseAll'); %or sca
ListenChar(0); % Restore keyboard output to Matlab

Screen('Preference', 'VisualDebuglevel', 4);

% close audio buffers
PsychPortAudio('Close')
end