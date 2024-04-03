function [pahandle,expWin,rect]=fInitialisePTB(design)

Screen('Preference', 'SkipSyncTests',1) % set to 1 for debugging
dummymode=0;   % set to 1 to run in dummymode (using mouse as pseudo-eyetracker)

%Perform basic initialization of the sound driver:
InitializePsychSound();
% InitializePsychSound(1); % start psychtoolbox sound module

devices = PsychPortAudio('GetDevices', [], []);

%Open the default audio device [], with default mode [] (==Only playback),
%and a required latencyclass of 1 == low-latency mode, as well as
%a frequency of freq and nrchannels sound channels.
%This returns a handle to the audio device:

if design.EEG %need 3 channels for EEG (1x TRIGGER, 2x AUDIO)
    nchans = 3;
    pahandle = PsychPortAudio('Open', [], [], 3, design.fs, nchans, [], [], [design.trigchan design.audchan(1) design.audchan(2)]); % open handle
    PsychPortAudio('RunMode', pahandle, 1); % the audio hardware and processing don't shut down at the end of audio playback. Instead, everything remains active in a ''hot standby'' state. This allows to very quickly (with low latency) restart sound playback via the 'RescheduleStart' function.
    %PsychPortAudio('UseSchedule', pahandle, 1, 1);
else
    pahandle = PsychPortAudio('Open', [], [], 2);  %%%%add 4 in 2nd argument when with soundcard
end

% % make volume lower so it doesnt deafen the ppt
% [oldMasterVolume, oldChannelVolumes] = PsychPortAudio('Volume',pahandle);
% 
% 


%Play welcome sound
wavdata = audioread('_tone1000Hz.wav');
if design.EEG == 0
    PsychPortAudio('FillBuffer', pahandle, [wavdata'; wavdata']);
elseif design.EEG ==1
    data = zeros(size(wavdata,2),size(wavdata,1));
    %PsychPortAudio('CreateBuffer', pahandle, [wavdata'; wavdata'; data]);
    PsychPortAudio('FillBuffer', pahandle, [wavdata'; wavdata'; data]);
end
PsychPortAudio('Start', pahandle, 1, 0, 1);

%Set higher DebugLevel, so that you don't get all kinds of messages flashed
%at you each time you start the experiment:
Screen('Preference', 'VisualDebuglevel', 2);

%Choosing the display with the highest display number is
%a best guess about where you want the stimulus displayed.
%usually there will be only one screen with id = 0, unless you use a
%multi-display setup:
screens=Screen('Screens');
screenNumber=max(screens);

% STEP 1
%open an (the only) onscreen Window, if you give only two input arguments
%this will make the full screen white (=default)
if design.fullScreen
[expWin,rect]=Screen('OpenWindow',screenNumber,0);
else
%Alternative: replace the above with smaller window for testing
[expWin,rect]=Screen('OpenWindow',screenNumber,0,[50 50 400 400]);
%NOTE that smaller windows can induce synchronisation problems
%and other issues, so they're not suitable for running real experiment
%sessions. See >> help SyncTrouble
end

%Get rid of the mouse cursor, we don't have anything to click at anyway
if ~dummymode, HideCursor; end

%Preparing and displaying the welcome screen
%We choose a text size of 20 pixels - Well readable on most screens:
Screen('TextSize', expWin, 20);
Screen('TextFont', expWin, 'Geneva');

%Enable unified mode of KbName, so KbName accepts identical key names on
%all operating systems (not absolutely necessary, but good practice):
KbName('UnifyKeyNames');

%Start recording keypress events
KbQueueCreate; %create queue
KbQueueStart; %start recording keypresses

%Steal keyboard (after this line all the keypresses will NOT be
%'heard' by MATLAB. Only keys available will be CTRL+C to abort the
%sound
% ListenChar(2);

end