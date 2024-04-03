%% Example experiment config - REGr behaviour only
%
% 1. generate a structure 'design' containing experiment parameters
% 2. get subject details (change subj id to generate new design matrix, subj-specific stim matrix is not overwritten)
% 3. design stimuli, REGr are stored in a regr matrix with a 20-tone
% pattern repetaed 6 times
% 4. start experiment, input subj number, input block number
% PRES Q to ABORT, sca to close psychtoolbox
%
% StartExp.m line 17: set full screen = 1 for real experiment
% fInitialisePTB.m line 23: set 2nd argumentset 4 when running with soundcard, empty when with laptop
% 1st DAY: run 5 learning blocks + block 6 which is the familiarity block

%______________________________Roberta Bianco, last updated Aprile 2022 


clearvars;
close all;
%% Parameters
design.fullScreen = 1;    % 1 for full screen 
design.typeExp          = 'REGr';  
Screen('Preference', 'SkipSyncTests', 0); % if problems with psychtoolbox screen sync, set to 1

%% Training
design.nBlocks          = 5; % number of blocks total - needed for generating conditions
design.nBlocksRun       = 1; % number of blocks to run - e.g. set to 1 and re-start matlab before each block
design.nEachCond        = [30 30 30 5 5]*design.nBlocks;  % repeats throughout experiment, must divide by n blocks
design.targetPerBlock   = 10; % items to repeat, must be divisor of N stimuli per condition (nSets = (stimPerBlock*nBlock)/nCond)
design.stimPerBlock     = sum(design.nEachCond/design.nBlocks); % N tot in a block
design.targetRepPerBlock = 3; % how many times target items are presented within a block
design.startTrial       = 1; % start from 1 for real experiment

design.conds            = {'RANREGr', 'RANREG', 'RAN','STEP', 'CONT'};   % condition labels
design.condsSize        = [20 20 20 2 1]; % how many tones in pool for the respective conditions
design.condsRand        = [1 1 1 0 0];    % which conditions are/start as RAND sequences (0 REG, 1 RAND)
design.condsTrans       = [1 1 0 1 0];    % which conditions transit to other
design.condsContr       = [0 0 0 1 1];    % which conditions is the control 
design.condsTarget      = [1 0 0 0 0];    % which conditions is the target 
design.condi            = [1 2 3 4 5];    % assign each condition with  unique number to use throughout
design.rtedges          = [2.200 2.550];  % required RTs for positive or negative feedback
design.rtedgesctr       = [0.400 0.700];  % required RTs for positive or negative feedback

%% familiarity (if run only familiarity block, set design.nBlocksRun above to 0)
design.familiarityBlock  = 1;   %logical: 1 if you want to run it
design.FnBlocks          = 1;  % number of blocks total - needed for generating conditions
design.FnEachCond        = [10 40]*design.FnBlocks;  % repeats throughout experiment, must divide by n blocks
design.FstimPerBlock     = sum(design.FnEachCond/design.FnBlocks); % sum of nEachCond/nBlocks (stimPerBlock*nBlocks must divide by length of Condition)
design.Fconds            = {'REGr','REG'};   % condition labels
design.FcondsSize        = [20 20];  % how many tones in pool for the respective conditions
design.FcondsRand        = [0 0];    % which conditions are/start as RAND sequences (0 REG, 1 RAND)
design.FcondsTrans       = [0 0];    % which conditions transit to other
design.FcondsContr       = [0 0];    % which conditions is the control 
design.FcondsTarget      = [1 0];    % which conditions is the target 
design.Fcondi            = [6 7];    % assign each condition with  unique number to use throughout
   
%% general features  
design.fs               = 44100; % audio sampling rate
design.toneDur          = 50;    % pip duration (ms) 
design.seqDur           = 5000;  % length of stimulus total, ms, this is tonDur*seqLength
design.transitionTime   = [1500 2000];% possible range of times where transition can occur; needs to be multiple of toneDur
design.seqLength        = design.seqDur/design.toneDur;  % number of pips in sequence
design.seqLengthFAM     = 3000/design.toneDur; %% 3000 ms/toneduration in ms
design.freqPool         = [222;250;280;315;354;397;445;500;561;630;707;793;890;1000;1122;1259;1414;1587;1781;2000];
design.ISI              = 500:900;   	% ISI (ms)
design.endWait          = 500; % must be less than ISI+250 % how many milliseconds to wait at endof trial for accepting further button presses
design.passive          = 0; % 0 = record responses // 1 = passive
design.EEG              = 0; % EEG requires triggers specified below...

%%
clc
design.subnum       = input('Write down subject number : ','s');
whichBlock          = input('Block number : ');

design.resultsDir = ['../DATA/behavior/Subject_' num2str(design.subnum) '/'];
mkdir(design.resultsDir);

design.fmriDir = ['../DATA/Subject_' num2str(design.subnum) '/'];
set1 = load([design.fmriDir 'design_sub' num2str(design.subnum) '_set1.mat']);
set2 = load([design.fmriDir 'design_sub' num2str(design.subnum) '_set2.mat']);

%% change ID to regr of set 2 (from 5 to 10)
regr = set1.regr;
for i = 1:5
    set2.regr(i).setID = i+5;
    regr(i+5).sequence = set2.regr(i).sequence;
    regr(i+5).setID = set2.regr(i).setID;
end

%% Load design structure 
if whichBlock ==1
    design.whichBlock=1;
        disp('No design file generated, making one now, please be patient! \n')
        [design, stimuli, stimuliF] = fDesignStimuli(design, regr);  %enter the function that create design of the stimuli
        save([design.resultsDir 'design_sub' num2str(design.subnum) '.mat'],'design', 'stimuli', 'stimuliF');
else
    try
        load([design.resultsDir 'design_sub' num2str(design.subnum) '.mat']);
        design.whichBlock = whichBlock; % overwrite design.whichBlock as we are now on the next one
        
    catch % just generate again for remaining blocks
        disp('*** ERROR - no design file found for previous blocks, generating again! \nPress any letter to continue')      
        design.nBlocksRun=design.nBlocks-whichBlock+1;
        design.whichBlock=1; % technically start from block 1
        [design, stimuli, stimuliF] = fDesignStimuli(design, regr);
        save([design.resultsDir 'design_sub' num2str(design.subnum) '.mat'],'design', 'stimuli', 'stimuliF');  
    end  
end

fExpMain(design,stimuli,stimuliF);  %start the experiment