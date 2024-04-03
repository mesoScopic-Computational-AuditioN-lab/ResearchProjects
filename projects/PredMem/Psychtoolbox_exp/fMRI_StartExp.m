%% START FMRI EXPERIMENT - fast event related design
%
% - make sure you've generated the stimuli using fMRI_GenerateStim.m for
%   desired subject (system will throw an error if you have not)
% - design settings are fatched from the `design` file generated in
%   'fMRI_GenerateStim.m'
% - other settings can be changed in the `fMRI_settings.m` file, which is
%   always loaded before the experiment (audiodriver settings, display
%   settings etc.)
%
%______________________________Jorie van Haren, last updated June 2023 

clear mex;
addpath(genpath([pwd '/functions/'])); startup1;

%% QUICK PARAMETERS
fullScreen       = 1;            % 1 for full screen 
typeExp          = 'fastFMRI';  
system           = 2;            % 0:windows, 1:MAC, 2:fMRI

%% PARTICIPANT INFORMATION INPUT
ppnum       = input('Write down subject number : ','s'); 
wrun        = input('Run number : ', 's');

%% CHECK AND LOAD PREGENERATED INFORMATION

% get stimulus directory
stimDir     = sprintf('../DATA/Subject_%s/', ppnum);
stimFn      = sprintf('sub%s_run%s_set1.mat', ppnum, wrun);
designFn    = sprintf('design_sub%s_set1', ppnum);

% check if preloaded file exists
if ~exist(stimDir, 'file')
    error('No stimuli found: [%s], \n- please run `fMRI_GenerateStim.m` first. (sub: %s, run: %s)', stimDir, ppnum, wrun);
end

% load design struct with settings
load([stimDir designFn], 'design');

%% HOUSEKEEPING: APPEND QUICKSETTINGS TO STRUCT
design.fullScreen       = fullScreen;
design.typeExp          = typeExp;
design.system           = system;

%% START THE EXPERIMENT
fMRI_ExpMain(design,ppnum,wrun);  %start the experiment

%% SHUT DOWN AND CLEANUP
ShowCursor;     ListenChar;     sca;   clearvars;