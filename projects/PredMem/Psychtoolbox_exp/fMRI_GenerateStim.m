%% Example experiment config - REGr for fMRI
%
% 1. generate a structure 'design' containing experiment parameters
% 2. get subject details (change subj id to generate new design matrix, subj-specific stim matrix is not overwritten)
% 3. design stimuli, REGr and RANr are stored in a regr or ranr matrix with a 20-tone pattern repeated into x n cycles
% 4. start experiment, input subj number, input block number
%
% 1st DAY: run 5 learning blocks + block 6 which is the familiarity block

%______________________________Roberta Bianco, last updated Aprile 2022


clearvars;
close all;
%%
clc
design.subnum       = input('Write down subject number : ','s');
design.resultsDir = ['../DATA/Subject_' num2str(design.subnum) '/'];
mkdir(design.resultsDir);


for set = 1:2  % there is a set 1 of regr and ranr for block 1 and a set 2 for block 2

    design.nBlocks          = 8; % number of blocks total - needed for generating conditions
    design.nEachCond        = [5 5 5 25]*design.nBlocks;  % repeats throughout experiment, must divide by n blocks
    design.stimPerBlock     = sum(design.nEachCond/design.nBlocks); % N tot in a block
    design.targetPerBlock   = 5; % items to repeat, must be divisor of N stimuli per condition (nSets = (stimPerBlock*nBlock)/nCond)
    design.targetRepPerBlock = 1; % how many times target items are presented within a block
    design.startTrial       = 1; % start from 1 for real experiment
    design.conds            = {'RANREGr', 'RANREG', 'RANRANr','RANRAN'};   % condition labels DON'T CHANGE LABELS for reoccurring condition
    design.condsSize        = [20 20 20 20]; % how many tones in pool for the respective conditions
    design.condsRand        = [1 1 1 1];      % which conditions are/start as RAND sequences (0 REG, 1 RAND)
    design.condsTrans       = [1 1 1 0];      % which conditions transit to other
    design.condsContr       = [0 0 0 0];      % which conditions is the control one frequency or step frequency
    design.condsTarget      = [1 0 1 0];      % which conditions is the target that repeats
    design.condi            = [1 2 3 4];    % assign each condition with  unique number to use throughout

    %% general features
    design.fs               = 44100; % audio sampling rate
    design.toneDur          = 50;    % pip duration (ms)
    design.seqDur           = 5000;  % length of stimulus total, ms, this is tonDur*seqLength
    design.transitionTime   = [3000 3000];% possible range of times where transition can occur; needs to be multiple of toneDur
    design.seqLength        = design.seqDur/design.toneDur;  % number of pips in sequence
    design.seqLengthFAM     = 3000/design.toneDur; %% 3000 ms/toneduration in ms
    design.freqPool         = [222;250;280;315;354;397;445;500;561;630;707;793;890;1000;1122;1259;1414;1587;1781;2000];

    %% Load design structure or generate one if not found
    [design, stimuli, regr, ranr] = fMRI_funDesignStimuli(design);  %enter the function that create design of the stimuli
    save([design.resultsDir 'design_sub' num2str(design.subnum) '_set' num2str(set) '.mat'],'design', 'stimuli', 'ranr', 'regr');

    fMRI_funSaveWav(design,stimuli, set)
end