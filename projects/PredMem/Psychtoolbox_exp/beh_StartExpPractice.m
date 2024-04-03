%% Example experiment config - reg detection task
%
% 1. generate a structure 'design' containing experiment parameters
% 2. get subject details
% 3. design stimuli
% 4. start experiment
% PRESS Q TO QUIT, PRESS sca to close psychtoolbox
%______________________________Roberta Bianco, last updated Oct 2017


clearvars;
close all;
%% Parametersq
design.fullScreen = 0;    % 1 for full screen
design.typeExp          = '';  
Screen('Preference', 'SkipSyncTests', 0); % if problems with psychtoolbox screen sync, set to 1

%% Exposure
design.nBlocks          = 1; % number of blocks total - needed for generating conditions
design.nBlocksRun       = 1; % number of blocks to run - e.g. set to 1 and re-start matlab before each block

design.stimPerBlock     = 12; % sum of nEachCond/nBlocks (stimPerBlock*nBlocks must divide by length of Condition)
design.nEachCond        = [5 5 1 1];  % repeats throughout experiment, must divqqide by n blocks
design.targetPerBlock   = 0; % items to repeat, must be divisor of N stimuli per condition (nSets = (stimPerBlock*nBlock)/nCond)
design.targetRepPerBlock = 0; % how many times target items repeat within a block

design.conds            = {'RANREG', 'RAND','STEP', 'CONT'};   % condition labels
design.condsSize        = [20 20 2 1]; % how many tones in pool for the respective conditions
design.condsRand        = [1 1 0 0];    % which conditions are/start as RAND sequences (0 REG, 1 RAND)
design.condsTrans       = [1 0 1 0];    % which conditions transit to other
design.condsContr       = [0 0 1 1];    % which conditions is the control 
design.condsTarget      = [0 0 0 0];    % which conditions is the target 
design.condi            = [1 2 3 4];    % assign each condition with  unique number to use throughout
design.familiarityBlock  = 0;

%% general features  
design.fs               = 44100; % audio sampling rate  
design.toneDur          = 50;    % pip duration (ms) 
design.seqDur           = 7000;  % length of stimulus total, ms, this is tonDur*seqLength
design.transitionTime   = [3000 4000];% possible range of times where transition can occur; needs to be multiple of toneDur
design.seqLength        = design.seqDur/design.toneDur;  % number of pips in sequence
design.freqPool         = [222;250;280;315;354;397;445;500;561;630;707;793;890;1000;1122;1259;1414;1587;1781;2000];

design.drawReplace      = 0; % 1: freqs drawn with (1) or without (0) replacement
design.noRep            = 1; % 0: allow adjacent tones to repeat
design.ISI              = 1100:1500;   	% ISI (ms)
design.endWait          = 500; % must be less than ISI+250 % how many milliseconds to wait at endof trial for accepting further button presses
% silence following stimulus = 250ms in stimulus + ISI
design.passive          = 0; % 0 = record responses // 1 = passive
design.EEG              = 0; % EEG requires triggers specified below...

%%
clc
design.subnum       = input('Write down subject number : ','s');
whichBlock          = input('Block number : ');

design.resultsDir = ['../DATA/practice/Subject_' num2str(design.subnum) '/'];
mkdir(design.resultsDir);

%% Load design structure or generate one if not found

if whichBlock ==1
    design.whichBlock=1;
    try
        load([design.resultsDir 'design_sub' num2str(design.subnum) '.mat']);
        design.whichBlock=1;
        
    catch %
        disp('No design file generated, making one now, please be patient! \n')
        [design, stimuli] = pDesignStimuli(design);  %enter the function that create design of the stimuli
        save([design.resultsDir 'design_sub' num2str(design.subnum) '.mat'],'design', 'stimuli');
    end
else
    try
        load([design.resultsDir 'design_sub' num2str(design.subnum) '.mat']);
        design.whichBlock = whichBlock; % overwrite design.whichBlock as we are now on the next one
        
    catch % just generate again for remaining blocks
        disp('*** ERROR - no design file found for previous blocks, generating again! \nPress any letter to continue')      
        design.nBlocksRun=design.nBlocks-whichBlock+1;
        design.whichBlock=1; % technically start from block 1
        [design, stimuli, stimuliF] = fDesignStimuli(design);
        save([design.resultsDir 'design_sub' num2str(design.subnum) '.mat'],'design', 'stimuli');  
    end
end

pExpMain(design,stimuli);  %start the experiment